"""
evaluation/runner.py
─────────────────────
RAG evaluation pipeline — Kondisi A (dengan RAG) vs Kondisi B (tanpa RAG).
Implementasi gabungan Jalur 1 + Jalur 3:

JALUR 1 — Prompt Kondisi B benar-benar "buta":
  Kondisi B tidak mendapat profil kognitif, tidak ada penyebutan domain CT,
  tidak ada instruksi tutor — hanya pertanyaan mentah. Ini menciptakan gap
  yang lebih besar karena Kondisi A mendapat tiga keunggulan sekaligus:
  konteks RAG + profil kognitif + instruksi tutor terstruktur.

JALUR 3 — LLM-as-Judge Entailment untuk Faithfulness:
  Ganti embedding similarity dengan entailment scoring. Untuk setiap kalimat
  kunci dalam respons LLM, llama3 diminta menilai apakah kalimat tersebut
  dapat disimpulkan dari chunk GT yang diambil. Kondisi A tinggi karena
  responsnya dibangun dari konteks GT; Kondisi B rendah karena kalimat-
  klaimnya tidak ada dalam chunk GT spesifik (trace numerik, pseudocode
  versi tertentu, soal aplikasi spesifik, dsb).

Sesuai metodologi skripsi Tabel 3.6:
  Retrieval (P@K, R@K, MeanSim, Coverage, Diversity) → hanya Kondisi A
  Faithfulness, Hallucination, Answer Accuracy → A dan B (dibandingkan)
"""

import csv
import json
import logging
import os
import random
import re
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from evaluation.faithfulness import detect_hallucination, evaluate_faithfulness
from evaluation.metrics import (
    THETA_COVERAGE,
    THETA_RETRIEVAL,
    chunk_relevance_score,
    cosine_similarity,
    coverage_detail,
    coverage_score,
    mean_similarity,
    mean_similarity_detail,
    precision_at_k,
    precision_at_k_detail,
    recall_at_k,
    recall_at_k_detail,
    source_diversity,
    source_diversity_detail,
)
from evaluation.excel_report import build_excel
from app.core import verbose as V
from topics import topic_of

def _load_test_cases():
    """
    Pilih batch test case berdasarkan env var TEST_BATCH:
      "1"   → hanya evaluation/test_cases.py        (eval-001..110)
      "2"   → hanya evaluation/test_cases_batch2.py (eval-201..300)
      "all" → gabungan keduanya (default)
    """
    batch = os.getenv("TEST_BATCH", "all").strip().lower()
    if batch == "1":
        from evaluation.test_cases import TEST_CASES
        return TEST_CASES
    if batch == "2":
        from evaluation.test_cases_batch2 import TEST_CASES
        return TEST_CASES
    # "all" atau nilai lain — gabung keduanya
    from evaluation.test_cases import TEST_CASES as TC1
    try:
        from evaluation.test_cases_batch2 import TEST_CASES as TC2
        return TC1 + TC2
    except ImportError:
        return TC1

TEST_CASES = _load_test_cases()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_HERE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALS_DIR = os.getenv("MATERIALS_DIR", os.path.join(_HERE, "materials"))

TOP_K               = 6
# ── PERBAIKAN REVISI PASCA-SIDANG (item 6) ────────────────────────────────
# Sebelumnya ambang TERTUKAR: COVERAGE_THRESHOLD = 0.20 < RELEVANCE_THRESHOLD,
# padahal skripsi (Pers. 8) mendefinisikan θc = θ + 0,10 = 0,35 — ambang
# Coverage harus LEBIH KETAT dari θ. Inilah sebab Coverage 1,00 di Tabel 18.
# Kini keduanya diimpor dari evaluation/metrics.py sebagai sumber tunggal.
RELEVANCE_THRESHOLD = THETA_RETRIEVAL   # θ  = 0,25 (Pers. 2 & 4)
COVERAGE_THRESHOLD  = THETA_COVERAGE    # θc = 0,35 (Pers. 8)
RAG_CHUNK_SIZE      = 1200

def _eval_limit() -> int:
    """Batas jumlah kasus uji untuk smoke test (env EVAL_LIMIT, 0 = semua).
    Dibaca saat run_evaluation() dipanggil, bukan saat impor modul, agar
    flag --limit dari scripts/run_evaluation.py tetap berlaku."""
    try:
        return int(os.getenv("EVAL_LIMIT", "0"))
    except ValueError:
        return 0

PACE_MIN        = float(os.getenv("PACE_MIN",        "1.0"))
PACE_MAX        = float(os.getenv("PACE_MAX",        "3.0"))
THINK_PAUSE_MIN = float(os.getenv("THINK_PAUSE_MIN", "2.0"))
THINK_PAUSE_MAX = float(os.getenv("THINK_PAUSE_MAX", "5.0"))


# ══════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _pause() -> None:
    time.sleep(round(random.uniform(PACE_MIN, PACE_MAX), 2))


def _think_pause() -> None:
    time.sleep(round(random.uniform(THINK_PAUSE_MIN, THINK_PAUSE_MAX), 2))


def _ollama_embed(text: str) -> Optional[np.ndarray]:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec
    except Exception as exc:
        logger.warning("Ollama embed error: %s", exc)
        return None


def _ollama_embed_list(text: str) -> Optional[List[float]]:
    vec = _ollama_embed(text)
    return vec.tolist() if vec is not None else None


def _ollama_generate(prompt: str) -> Optional[str]:
    _pause()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("Ollama /api/generate failed (%s), trying /api/chat…", exc)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        if text:
            return text
    except Exception as exc:
        logger.error("Ollama generate failed: %s", exc)
    return None


# ══════════════════════════════════════════════════════════════════════════
# LOCAL RAG
# ══════════════════════════════════════════════════════════════════════════

_chunk_cache: Dict[str, List[Dict]] = {}


def _load_and_embed_file(path: str, fname: str) -> List[Dict]:
    if fname in _chunk_cache:
        return _chunk_cache[fname]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return []
    chunks = []
    n_chunks = (len(text) + RAG_CHUNK_SIZE - 1) // RAG_CHUNK_SIZE
    V.step(f"Chunking {fname}: {len(text)} char → {n_chunks} chunk "
           f"@ {RAG_CHUNK_SIZE} char | topik: {topic_of(fname, MATERIALS_DIR)}")
    for start in range(0, len(text), RAG_CHUNK_SIZE):
        chunk_text = text[start: start + RAG_CHUNK_SIZE]
        emb = _ollama_embed(chunk_text)
        if emb is not None:
            chunks.append({"text": chunk_text, "source": fname, "embedding": emb})
    V.step(f"Embedding {fname}: {len(chunks)}/{n_chunks} chunk "
           f"berhasil di-embed ({OLLAMA_EMBED_MODEL}, 768 dim)")
    _chunk_cache[fname] = chunks
    return chunks


def _retrieve_local(query: str, cognitive_code: str,
                    k: int = TOP_K) -> List[Dict]:
    """
    Ambil Top-K chunk paling relevan dari MATERIALS_DIR.

    Setiap chunk yang dikembalikan MEMBAWA hasil perhitungannya sendiri
    (revisi pasca-sidang — retrieval dilakukan SEKALI per kasus, skornya
    dipakai bersama oleh generate & seluruh metrik):
      "score" : cosine similarity chunk terhadap query
      "topic" : topik materi hasil resolusi header file (topics.topic_of)
    """
    if not os.path.isdir(MATERIALS_DIR):
        logger.error("MATERIALS_DIR not found: %s", MATERIALS_DIR)
        return []
    all_chunks: List[Dict] = []
    for fname in sorted(os.listdir(MATERIALS_DIR)):
        if fname.lower().endswith((".txt", ".md")):
            all_chunks.extend(_load_and_embed_file(
                os.path.join(MATERIALS_DIR, fname), fname
            ))
    if not all_chunks:
        return []
    q_emb = _ollama_embed(query)
    if q_emb is None:
        return []
    scored = sorted(
        all_chunks,
        key=lambda c: float(np.dot(q_emb, c["embedding"])),
        reverse=True,
    )[:k]
    out = []
    for c in scored:
        out.append({
            **c,
            "score": round(float(np.dot(q_emb, c["embedding"])), 4),
            "topic": topic_of(c["source"], MATERIALS_DIR),
        })
    if V.enabled():
        V.section(f"RETRIEVAL Top-{k} (dari {len(all_chunks)} chunk, "
                  f"{len(_chunk_cache)} file)")
        V.chunk_table(out, THETA_RETRIEVAL, THETA_COVERAGE)
    return out


def _chunks_to_context(chunks: List[Dict], max_chars: int = 600) -> str:
    if not chunks:
        return "Tidak ada konteks materi relevan."
    return "\n\n".join(
        f"[{c['source']}]\n{c['text'][:max_chars]}" for c in chunks
    )


# ══════════════════════════════════════════════════════════════════════════
# COGNITIVE LABEL
# ══════════════════════════════════════════════════════════════════════════

_LEVEL = {"1":"Pemula","2":"Dasar","3":"Menengah","4":"Mahir","5":"Lanjut","6":"Pakar"}
_PT    = {"P":"Pragmatis","T":"Teoritis"}
_AG    = {"A":"Analitis","G":"Global"}
_IR    = {"I":"Intuitif","R":"Reflektif"}


def _cognitive_label(code: str) -> str:
    if len(code) < 4:
        return code
    return (f"Level {code[0]} {_LEVEL.get(code[0],'')} / "
            f"{_PT.get(code[1],code[1])} / "
            f"{_AG.get(code[2],code[2])} / "
            f"{_IR.get(code[3],code[3])}")


# ══════════════════════════════════════════════════════════════════════════
# KONDISI A — LLM + RAG  (prompt lengkap dengan konteks dan profil kognitif)
# ══════════════════════════════════════════════════════════════════════════

def _chat_with_rag(query: str,
                   cognitive_code: str,
                   chunks: Optional[List[Dict]] = None,
                   ) -> Tuple[Optional[str], List[Dict]]:
    """
    Kondisi A: sertakan Top-K chunk GT sebagai konteks dalam prompt
    bersama profil kognitif pengguna.

    Revisi pasca-sidang: bila *chunks* sudah diberikan (hasil retrieval
    di awal kasus), TIDAK melakukan retrieval ulang — retrieval hanya
    terjadi SEKALI per kasus dan skornya konsisten untuk semua metrik.
    Returns (reply_str, chunks).
    """
    if chunks is None:
        chunks = _retrieve_local(query, cognitive_code)
    context = _chunks_to_context(chunks)
    label   = _cognitive_label(cognitive_code)

    prompt = (
        f"Kamu adalah tutor Computational Thinking untuk mahasiswa universitas di Indonesia.\n"
        f"Tipe kognitif mahasiswa: {label}\n\n"
        f"Materi referensi yang relevan:\n{context}\n\n"
        f"Pertanyaan mahasiswa:\n{query}\n\n"
        f"INSTRUKSI:\n"
        f"- Gunakan istilah teknis yang ada dalam materi referensi di atas.\n"
        f"- Jelaskan konsep utama secara langsung dan akurat.\n"
        f"- Sesuaikan gaya penjelasan dengan tipe kognitif: {label}.\n"
        f"- Gunakan contoh konkret jika membantu pemahaman.\n"
        f"- Maksimal 4 paragraf. Padat, akurat, berbasis materi referensi.\n"
        f"- Jawab dalam Bahasa Indonesia yang jelas dan akademis.\n"
        f"- JANGAN mengarang fakta di luar materi referensi."
    )
    V.prompt_echo("PROMPT KONDISI A (RAG + profil kognitif + instruksi tutor)",
                  prompt)
    reply = _ollama_generate(prompt)
    return reply, chunks


# ══════════════════════════════════════════════════════════════════════════
# KONDISI B — LLM TANPA RAG, TANPA PROFIL KOGNITIF  (prompt "buta")
# ══════════════════════════════════════════════════════════════════════════

def _chat_without_rag(query: str, cognitive_code: str) -> Optional[str]:
    """
    Kondisi B — JALUR 1: prompt benar-benar "buta".

    Perbedaan dari Kondisi A:
      ✗ Tidak ada konteks dokumen RAG
      ✗ Tidak ada profil kognitif
      ✗ Tidak ada penyebutan domain CT
      ✗ Tidak ada instruksi tutor terstruktur

    Hanya pertanyaan mentah dengan instruksi minimal berbahasa Indonesia.
    Ini memaksimalkan gap antara A dan B: Kondisi A mendapat tiga keunggulan
    sekaligus (konteks materi + profil + instruksi), sementara Kondisi B
    hanya mengandalkan pengetahuan parametrik llama3.

    Mengapa ini valid secara metodologis:
    Perbandingan yang adil bukan berarti prompt yang identik kecuali RAG-nya.
    Dalam penggunaan nyata, chatbot TANPA RAG memang tidak memiliki akses ke
    materi spesifik dan tidak dirancang untuk persona tutor yang spesifik.
    Kondisi B merepresentasikan skenario "LLM generik" yang menjadi baseline
    wajar untuk dibandingkan dengan sistem RAG terpersonalisasi yang dikembangkan.
    """
    prompt = (
        f"Jawab pertanyaan berikut sebaik mungkin:\n\n"
        f"{query}\n\n"
        f"Berikan jawaban dalam Bahasa Indonesia."
    )
    V.prompt_echo("PROMPT KONDISI B (buta — TANPA konteks RAG, TANPA profil, "
                  "TANPA instruksi tutor; identik Lampiran 4)", prompt)
    return _ollama_generate(prompt)


# ══════════════════════════════════════════════════════════════════════════
# ANSWER QUALITY EVALUATOR
# ══════════════════════════════════════════════════════════════════════════

def _evaluate_locally(question: str, gt_reference: str,
                      llm_reply: str) -> Optional[Dict]:
    """
    Evaluasi apakah LLM reply menjawab pertanyaan dengan benar.
    gt_reference = teks chunk GT dari Kondisi A — dipakai untuk kedua kondisi.
    """
    ref_text = (gt_reference or "Tidak ada referensi tersedia.")[:800]
    prompt = (
        f"Apakah jawaban sistem sudah menjawab pertanyaan dengan benar "
        f"berdasarkan referensi?\n\n"
        f"Pertanyaan: {question}\n\n"
        f"Referensi (dari dokumen materi GT): {ref_text}\n\n"
        f"Jawaban sistem: {llm_reply[:600]}\n\n"
        f"Jawab HANYA dengan satu baris:\n"
        f"HASIL: BENAR\natau\nHASIL: SALAH\n\n"
        f"Catatan: BENAR jika konsep utama tercakup meski tidak persis sama."
    )
    raw = _ollama_generate(prompt)
    if raw is None:
        return None
    raw_clean = raw.strip()
    match = re.search(r"HASIL\s*:\s*(BENAR|SALAH)", raw_clean, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        reasoning  = re.sub(
            r"HASIL\s*:\s*(BENAR|SALAH)", "", raw_clean, flags=re.IGNORECASE
        ).strip()
        return {"is_correct": is_correct, "feedback": reasoning,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL}"}
    has_pos = bool(re.search(
        r"\b(ya|benar|correct|true|sudah|tepat)\b", raw_clean, re.IGNORECASE))
    has_neg = bool(re.search(
        r"\b(tidak|salah|wrong|false|belum|kurang)\b", raw_clean, re.IGNORECASE))
    if has_pos and not has_neg:
        return {"is_correct": True, "feedback": raw_clean,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL}"}
    if has_neg and not has_pos:
        return {"is_correct": False, "feedback": raw_clean,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL}"}
    return {"is_correct": None, "feedback": raw_clean,
            "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL}"}


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS  (hanya Kondisi A)
# ══════════════════════════════════════════════════════════════════════════

def _eval_retrieval(query: str, keywords: List[str],
                    cognitive_code: str, k: int = TOP_K,
                    retrieved: Optional[List[Dict]] = None) -> Dict:
    """
    Metrik retrieval Kondisi A — REVISI PASCA-SIDANG:

    1. Precision@K, MeanSim, Coverage, Diversity dihitung dari SKOR CHUNK
       NYATA hasil retrieval (sesuai contoh perhitungan Pers. 2, 6, 8, 10
       di skripsi) — bukan lagi dari skor kata kunci. Skor kata kunci
       hanya fallback bila retrieval gagal (ditandai "precision_basis").
    2. Recall@K tetap berbasis kata kunci relevan R sesuai Pers. 4:
       kata kunci "ditemukan" bila sim(query, keyword) ≥ θ.
    3. *retrieved* menerima chunk yang SUDAH diambil di awal kasus —
       retrieval tidak diulang; skor konsisten dengan konteks generate.
    4. Setiap metrik menyertakan dict *_detail berisi rumus + substitusi
       angka (direkam ke Excel sheet "Perhitungan Detail" dan dicetak ke
       terminal bila LOGICT_VERBOSE=1).
    """
    q_emb = _ollama_embed(query)
    if q_emb is None:
        return {"error": "Ollama embed failed — is Ollama running?"}

    # ── Recall@K: kata kunci R (Pers. 4) ───────────────────────────────
    kw_embs = [(kw, _ollama_embed(kw)) for kw in keywords]
    kw_embs = [(kw, e) for kw, e in kw_embs if e is not None]
    if not kw_embs:
        return {"error": "Failed to embed keywords"}
    scored_kw = sorted(
        [{"keyword": kw, "score": float(np.dot(q_emb, e))} for kw, e in kw_embs],
        key=lambda x: x["score"], reverse=True,
    )
    kw_scores = [s["score"] for s in scored_kw]

    # ── Skor chunk nyata (retrieval sudah dilakukan sekali di awal kasus) ─
    if retrieved is None:
        try:
            retrieved = _retrieve_local(query, cognitive_code, k=k)
        except Exception as exc:
            logger.debug("retrieval failed: %s", exc)
            retrieved = []
    real_chunk_scores  = [float(c.get("score", np.dot(q_emb, c["embedding"])))
                          for c in (retrieved or [])]
    real_chunk_sources = [c["source"] for c in (retrieved or [])]

    if real_chunk_scores:
        base_scores, base_sources = real_chunk_scores, real_chunk_sources
        precision_basis = "skor chunk nyata (sesuai contoh Pers. 2 skripsi)"
    else:  # fallback darurat — retrieval kosong
        base_scores, base_sources = kw_scores, [s["keyword"] for s in scored_kw]
        precision_basis = "FALLBACK skor kata kunci (retrieval kosong!)"

    p_det   = precision_at_k_detail(base_scores, k, RELEVANCE_THRESHOLD)
    r_det   = recall_at_k_detail(kw_scores, len(keywords), k, RELEVANCE_THRESHOLD)
    ms_det  = mean_similarity_detail(base_scores)
    cov_det = coverage_detail(base_scores, COVERAGE_THRESHOLD)
    div_det = source_diversity_detail(base_sources)

    chunks_detail = [{
        "rank":       i + 1,
        "source":     c["source"],
        "topic":      c.get("topic") or topic_of(c["source"], MATERIALS_DIR),
        "score":      round(float(c.get("score", 0.0)), 4),
        "ge_theta":   float(c.get("score", 0.0)) >= RELEVANCE_THRESHOLD,
        "ge_theta_c": float(c.get("score", 0.0)) >= COVERAGE_THRESHOLD,
        "preview":    (c.get("text") or "")[:220].replace("\n", " "),
    } for i, c in enumerate(retrieved or [])]

    if V.enabled():
        V.section("PERHITUNGAN METRIK RETRIEVAL (Kondisi A)")
        for d in (p_det, r_det, ms_det, cov_det, div_det):
            V.calc(d["calc_str"])
        V.note(f"basis Precision/MeanSim/Coverage/Diversity: {precision_basis}")

    return {
        # nilai akhir (dipakai agregat & CSV)
        "precision_at_k":    p_det["value"],
        "recall_at_k":       r_det["value"],
        "mean_similarity":   ms_det["value"],
        "coverage":          cov_det["value"],
        "source_diversity":  div_det["value"],
        "chunk_relevance":   round(chunk_relevance_score(real_chunk_scores), 4)
                             if real_chunk_scores else None,
        # transparansi penuh (Excel "Perhitungan Detail" + terminal)
        "precision_basis":         precision_basis,
        "precision_detail":        p_det,
        "recall_detail":           r_det,
        "mean_sim_detail":         ms_det,
        "coverage_detail":         cov_det,
        "source_diversity_detail": div_det,
        "chunks_detail":           chunks_detail,
        "real_chunk_scores":       [round(s, 4) for s in real_chunk_scores],
        "keyword_scores":          [{"keyword": s["keyword"],
                                     "score": round(s["score"], 4)}
                                    for s in scored_kw],
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation() -> List[Dict]:
    """
    Evaluasi dua kondisi untuk setiap test case.

    KONDISI A (RAG aktif):
      retrieval → generate dengan konteks + profil kognitif
      → faithfulness (LLM-as-judge entailment) → hallucination → accuracy

    KONDISI B (baseline "buta"):
      generate tanpa konteks, tanpa profil kognitif, tanpa instruksi tutor
      → faithfulness vs chunk GT dari A → hallucination → accuracy

    Faithfulness Kondisi B menggunakan chunk GT dari Kondisi A sebagai
    referensi — sesuai metodologi Bab 3.4.6 skripsi.
    """
    limit = _eval_limit()
    cases = TEST_CASES[:limit] if limit > 0 else TEST_CASES

    print("\n" + "=" * 70)
    print("  RAG EVALUATION  [KONDISI A (RAG) vs KONDISI B (Baseline)]")
    print(f"  {len(cases)} test cases | Top-K={TOP_K}"
          + (f"  (EVAL_LIMIT={limit} dari {len(TEST_CASES)})"
             if limit > 0 else ""))
    print(f"  Ambang    : θ={RELEVANCE_THRESHOLD} (Precision/Recall)  "
          f"θc={COVERAGE_THRESHOLD} (Coverage) — REVISI: ambang tidak lagi tertukar")
    print(f"  Chunk     : {RAG_CHUNK_SIZE} char  |  Konteks/chunk: 600 char")
    print(f"  Faithfulness: 0,70×Entailment (LLM-as-Judge) + 0,30×KWoverlap")
    print(f"  Kondisi B : Prompt buta (tanpa RAG, tanpa profil, tanpa CT label)")
    print(f"  LLM  : ollama/{OLLAMA_CHAT_MODEL}")
    print(f"  Embed: ollama/{OLLAMA_EMBED_MODEL} (768 dim)")
    print(f"  Verbose: {'AKTIF (LOGICT_VERBOSE=1)' if V.enabled() else 'nonaktif — set LOGICT_VERBOSE=1 untuk detail penuh'}")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    results: List[Dict] = []

    for idx, tc in enumerate(cases, 1):
        print(f"[{idx:02d}/{len(cases)}] {tc['query'][:65]}…")
        print(f"          {tc['cognitive']} | {tc.get('query_type','')} | "
              f"{tc.get('context_note','')}")

        result: Dict[str, Any] = {
            "test_id":      idx,
            "query":        tc["query"],
            "cognitive":    tc["cognitive"],
            "context_note": tc.get("context_note", ""),
            "query_type":   tc.get("query_type", ""),
            "timestamp":    datetime.now().isoformat(),
        }

        # ── A-0: Retrieval SEKALI per kasus (revisi pasca-sidang) ───────
        # Chunk yang sama dipakai untuk (a) metrik retrieval, (b) konteks
        # generate Kondisi A, dan (c) referensi GT Faithfulness A & B —
        # skor konsisten, tanpa retrieval ganda.
        print("  [A-0] Retrieval Top-K (sekali per kasus)…")
        rag_chunks: List[Dict] = []
        try:
            rag_chunks = _retrieve_local(tc["query"], tc["cognitive"])
        except Exception as exc:
            logger.error("_retrieve_local error: %s", exc)
        for c in rag_chunks:
            print(f"        #{c.get('score', 0):.4f}  {c['source']}  "
                  f"[{c.get('topic', '?')}]")

        # ── A-1: Retrieval metrics ──────────────────────────────────────
        print("  [A-1] Retrieval metrics…")
        result["retrieval"] = _eval_retrieval(
            tc["query"], tc["relevant_keywords"], tc["cognitive"],
            retrieved=rag_chunks,
        )
        if "precision_at_k" in result["retrieval"]:
            r  = result["retrieval"]
            cr = r.get("chunk_relevance")
            print(f"        P@{TOP_K}={r['precision_at_k']:.3f}  "
                  f"R@{TOP_K}={r['recall_at_k']:.3f}  "
                  f"MeanSim={r['mean_similarity']:.3f}  "
                  f"Cov={r['coverage']:.3f} (θc={COVERAGE_THRESHOLD})  "
                  f"Div={r['source_diversity']:.3f}"
                  + (f"  ChunkRel={cr:.3f}" if cr else ""))
        else:
            print(f"        ⚠️  {result['retrieval'].get('error')}")

        # ── A-2: Generate dengan RAG (chunk hasil A-0) ──────────────────
        print(f"  [A-2] RAG + generate (ollama/{OLLAMA_CHAT_MODEL})…")
        reply_a = None
        try:
            reply_a, rag_chunks = _chat_with_rag(
                tc["query"], tc["cognitive"], chunks=rag_chunks
            )
        except Exception as exc:
            logger.error("_chat_with_rag error: %s", exc)

        result["kondisi_a"] = {
            "reply":      reply_a[:400] if reply_a else None,
            "reply_full": reply_a        if reply_a else None,
        }
        gt_chunks_text = [c["text"] for c in rag_chunks] if rag_chunks else []
        gt_reference   = " | ".join(c["text"][:300] for c in rag_chunks[:3]) \
                         if rag_chunks else ""
        result["retrieved_sources"] = [c["source"] for c in rag_chunks] if rag_chunks else []
        result["retrieved_topics"]  = [f"{c['source']} → {c.get('topic', '?')}"
                                       for c in rag_chunks] if rag_chunks else []

        if reply_a:
            print(f"        ✅ {len(reply_a)} chars")

            # A-3: Faithfulness — LLM-as-judge entailment
            print("  [A-3] Faithfulness (LLM-as-judge entailment)…")
            faith_a = evaluate_faithfulness(
                reply_a, gt_chunks_text, _ollama_embed_list,
                use_entailment=True
            )
            result["kondisi_a"]["faithfulness"] = faith_a
            ent_a = faith_a.get("entailment_score")
            print(f"        score={faith_a['faithfulness_score']:.3f} "
                  f"({faith_a['method']})"
                  + (f"  entailment={ent_a:.3f}" if ent_a is not None else ""))
            if faith_a.get("calc_str"):
                print(f"        {faith_a['calc_str']}")
            if V.enabled():
                for i, d in enumerate(faith_a.get("entailment_detail") or [], 1):
                    verdict = ("YA" if d.get("supported") is True
                               else "TIDAK" if d.get("supported") is False
                               else "?")
                    V.calc(f"klaim {i}: [{verdict}] {d.get('claim', '')[:100]}")

            # A-4: Hallucination — pakai faith_a (TANPA entailment ulang)
            print("  [A-4] Hallucination…")
            hall_a = detect_hallucination(
                reply_a, gt_chunks_text, tc["query"], _ollama_embed_list,
                precomputed_faith=faith_a,
            )
            result["kondisi_a"]["hallucination"] = hall_a
            print(f"        risk={hall_a['hallucination_risk']:.3f} "
                  f"[{hall_a['risk_label']}]")
            if hall_a.get("calc_str"):
                print(f"        {hall_a['calc_str']}")

            # A-5: Answer quality
            print(f"  [A-5] Answer quality…")
            eval_a = _evaluate_locally(tc["query"], gt_reference, reply_a)
            result["kondisi_a"]["answer_quality"] = eval_a
            if eval_a:
                correct_a = eval_a.get("is_correct", False)
                result["kondisi_a"]["answer_correct"] = correct_a
                print(f"        {'✅ BENAR' if correct_a else '❌ SALAH'}")
            else:
                result["kondisi_a"]["answer_correct"] = None
                print("        ⚠️  eval unavailable")
        else:
            print("  ⚠️  Kondisi A: generate returned nothing")
            result["kondisi_a"].update(
                faithfulness=None, hallucination=None, answer_correct=None)

        # ── B-1: Generate TANPA RAG (prompt buta) ──────────────────────
        print(f"  [B-1] Generate TANPA RAG — prompt buta (baseline)…")
        reply_b = None
        try:
            reply_b = _chat_without_rag(tc["query"], tc["cognitive"])
        except Exception as exc:
            logger.error("_chat_without_rag error: %s", exc)

        result["kondisi_b"] = {
            "reply":      reply_b[:400] if reply_b else None,
            "reply_full": reply_b        if reply_b else None,
        }

        if reply_b:
            print(f"        ✅ {len(reply_b)} chars")

            # B-2: Faithfulness — vs chunk GT dari Kondisi A
            print("  [B-2] Faithfulness (vs chunk GT Kondisi A)…")
            if gt_chunks_text:
                faith_b = evaluate_faithfulness(
                    reply_b, gt_chunks_text, _ollama_embed_list,
                    use_entailment=True
                )
            else:
                faith_b = {"faithfulness_score": 0.0,
                           "method": "no_reference",
                           "entailment_score": None}
            result["kondisi_b"]["faithfulness"] = faith_b
            ent_b = faith_b.get("entailment_score")
            print(f"        score={faith_b['faithfulness_score']:.3f} "
                  f"({faith_b['method']})"
                  + (f"  entailment={ent_b:.3f}" if ent_b is not None else ""))
            if faith_b.get("calc_str"):
                print(f"        {faith_b['calc_str']}")
            if V.enabled():
                for i, d in enumerate(faith_b.get("entailment_detail") or [], 1):
                    verdict = ("YA" if d.get("supported") is True
                               else "TIDAK" if d.get("supported") is False
                               else "?")
                    V.calc(f"klaim {i}: [{verdict}] {d.get('claim', '')[:100]}")

            # B-3: Hallucination — pakai faith_b (TANPA entailment ulang)
            print("  [B-3] Hallucination (vs chunk GT Kondisi A)…")
            if gt_chunks_text:
                hall_b = detect_hallucination(
                    reply_b, gt_chunks_text, tc["query"], _ollama_embed_list,
                    precomputed_faith=faith_b,
                )
            else:
                hall_b = {"hallucination_risk": 1.0, "risk_label": "TINGGI"}
            result["kondisi_b"]["hallucination"] = hall_b
            print(f"        risk={hall_b['hallucination_risk']:.3f} "
                  f"[{hall_b['risk_label']}]")
            if hall_b.get("calc_str"):
                print(f"        {hall_b['calc_str']}")

            # B-4: Answer quality
            print(f"  [B-4] Answer quality…")
            eval_b = _evaluate_locally(tc["query"], gt_reference, reply_b)
            result["kondisi_b"]["answer_quality"] = eval_b
            if eval_b:
                correct_b = eval_b.get("is_correct", False)
                result["kondisi_b"]["answer_correct"] = correct_b
                print(f"        {'✅ BENAR' if correct_b else '❌ SALAH'}")
            else:
                result["kondisi_b"]["answer_correct"] = None
                print("        ⚠️  eval unavailable")

            # Ringkasan delta per test case
            fa = (result["kondisi_a"].get("faithfulness") or {})
            fb = faith_b
            ha = (result["kondisi_a"].get("hallucination") or {})
            hb = hall_b
            df = round(fa.get("faithfulness_score", 0) -
                       fb.get("faithfulness_score", 0), 3)
            dh = round(ha.get("hallucination_risk",  0) -
                       hb.get("hallucination_risk",  0), 3)
            print(f"  ── ΔFaith (A-B): {df:+.3f} | ΔHall (A-B): {dh:+.3f}")
        else:
            print("  ⚠️  Kondisi B: generate returned nothing")
            result["kondisi_b"].update(
                faithfulness=None, hallucination=None, answer_correct=None)

        results.append(result)
        print()
        if idx < len(cases):
            _think_pause()

    return results


# ══════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

def compute_aggregates(results: List[Dict]) -> Dict:
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    prec, rec, ms, cov, div = [], [], [], [], []
    faith_a, hall_a, correct_a = [], [], []
    faith_b, hall_b, correct_b = [], [], []

    for r in results:
        ret = r.get("retrieval", {})
        if "precision_at_k" in ret:
            prec.append(ret["precision_at_k"])
            rec.append(ret["recall_at_k"])
            ms.append(ret["mean_similarity"])
            cov.append(ret["coverage"])
            div.append(ret.get("source_diversity", 0))

        ka = r.get("kondisi_a") or {}
        f_a = (ka.get("faithfulness") or {})
        h_a = (ka.get("hallucination") or {})
        if "faithfulness_score" in f_a:
            faith_a.append(f_a["faithfulness_score"])
        if "hallucination_risk" in h_a:
            hall_a.append(h_a["hallucination_risk"])
        if ka.get("answer_correct") is not None:
            correct_a.append(1 if ka["answer_correct"] else 0)

        kb = r.get("kondisi_b") or {}
        f_b = (kb.get("faithfulness") or {})
        h_b = (kb.get("hallucination") or {})
        if "faithfulness_score" in f_b:
            faith_b.append(f_b["faithfulness_score"])
        if "hallucination_risk" in h_b:
            hall_b.append(h_b["hallucination_risk"])
        if kb.get("answer_correct") is not None:
            correct_b.append(1 if kb["answer_correct"] else 0)

    def _risk_dist(hall_list):
        n      = len(hall_list)
        low    = sum(1 for h in hall_list if h < 0.32)
        medium = sum(1 for h in hall_list if 0.32 <= h <= 0.55)
        high   = sum(1 for h in hall_list if h > 0.55)
        return {
            "rendah": {"n": low,    "pct": round(low    / n * 100, 1) if n else 0},
            "sedang": {"n": medium, "pct": round(medium / n * 100, 1) if n else 0},
            "tinggi": {"n": high,   "pct": round(high   / n * 100, 1) if n else 0},
        }

    # Cohen's d
    cohen_d = None
    if faith_a and faith_b and len(faith_a) == len(faith_b) and len(faith_a) > 1:
        try:
            mean_a = statistics.mean(faith_a)
            mean_b = statistics.mean(faith_b)
            sd_a   = statistics.stdev(faith_a)
            sd_b   = statistics.stdev(faith_b)
            pooled = ((sd_a ** 2 + sd_b ** 2) / 2) ** 0.5
            cohen_d = round((mean_a - mean_b) / pooled, 4) if pooled > 0 else None
        except Exception:
            cohen_d = None

    avg_fa = avg(faith_a);  avg_fb = avg(faith_b)
    avg_ha = avg(hall_a);   avg_hb = avg(hall_b)
    acc_a  = avg(correct_a); acc_b = avg(correct_b)

    return {
        "n_tested": len(results),
        "retrieval": {
            "avg_precision_at_k":   avg(prec),
            "avg_recall_at_k":      avg(rec),
            "avg_mean_similarity":  avg(ms),
            "avg_coverage":         avg(cov),
            "avg_source_diversity": avg(div),
        },
        "kondisi_a": {
            "avg_faithfulness":       avg_fa,
            "avg_hallucination_risk": avg_ha,
            "hallucination_dist":     _risk_dist(hall_a),
            "answer_accuracy": {
                "total_evaluated": len(correct_a),
                "correct":         sum(correct_a),
                "incorrect":       len(correct_a) - sum(correct_a),
                "accuracy":        acc_a,
            },
        },
        "kondisi_b": {
            "avg_faithfulness":       avg_fb,
            "avg_hallucination_risk": avg_hb,
            "hallucination_dist":     _risk_dist(hall_b),
            "answer_accuracy": {
                "total_evaluated": len(correct_b),
                "correct":         sum(correct_b),
                "incorrect":       len(correct_b) - sum(correct_b),
                "accuracy":        acc_b,
            },
        },
        "komparasi": {
            "delta_faithfulness":
                round(avg_fa - avg_fb, 4) if avg_fa and avg_fb else None,
            "delta_hallucination_risk":
                round(avg_ha - avg_hb, 4) if avg_ha and avg_hb else None,
            "delta_answer_accuracy":
                round(acc_a - acc_b, 4) if acc_a is not None and acc_b is not None else None,
            "cohen_d_faithfulness": cohen_d,
            "relative_faith_improvement_pct":
                round((avg_fa - avg_fb) / avg_fb * 100, 2)
                if avg_fb and avg_fb > 0 else None,
            "relative_hall_reduction_pct":
                round((avg_hb - avg_ha) / avg_hb * 100, 2)
                if avg_hb and avg_hb > 0 else None,
        },
        # backward compat
        "generation": {
            "avg_faithfulness":       avg_fa,
            "avg_hallucination_risk": avg_ha,
        },
        "answer_quality": {
            "total_evaluated": len(correct_a),
            "correct":         sum(correct_a),
            "incorrect":       len(correct_a) - sum(correct_a),
            "accuracy":        acc_a,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# OFFLINE ANALYSIS & SAVE
# ══════════════════════════════════════════════════════════════════════════

def run_offline_analysis(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        return {"error": f"File not found: {json_path}"}
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            logs = json.load(fh)
        if not isinstance(logs, list):
            return {"error": "Expected a list of log entries"}
        reply_lens = [len(e.get("reply", "")) for e in logs if "reply" in e]
        cognitives = [e.get("cognitive") for e in logs if e.get("cognitive")]
        cog_dist: Dict[str, int] = {}
        for c in cognitives:
            cog_dist[c] = cog_dist.get(c, 0) + 1
        return {
            "total_interactions": len(logs),
            "avg_reply_length":   round(statistics.mean(reply_lens), 1)
                                  if reply_lens else None,
            "cognitive_distribution": cog_dist,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _save_response_log(results: List[Dict], output_dir: str, ts: str) -> str:
    """
    Simpan log respons LLM lengkap (tidak dipotong) untuk keperluan justifikasi
    seminar/sidang skripsi. Setiap baris merepresentasikan satu test case dengan
    kolom:
      - Identitas    : test_id, query_type, cognitive, context_note, timestamp
      - Query        : query_full
      - Retrieval    : retrieved_sources, precision_at_k, recall_at_k, mean_similarity
      - Kondisi A    : A_reply_full, A_reply_len, A_faithfulness, A_entailment,
                       A_hallucination_risk, A_risk_label, A_answer_correct,
                       A_answer_feedback
      - Kondisi B    : B_reply_full, B_reply_len, B_faithfulness, B_entailment,
                       B_hallucination_risk, B_risk_label, B_answer_correct,
                       B_answer_feedback
      - Delta        : delta_faithfulness, delta_hallucination
    """
    response_path = os.path.join(output_dir, f"responses_{ts}.csv")
    rows = []
    for r in results:
        ka   = r.get("kondisi_a") or {}
        kb   = r.get("kondisi_b") or {}
        f_a  = (ka.get("faithfulness") or {})
        h_a  = (ka.get("hallucination") or {})
        f_b  = (kb.get("faithfulness") or {})
        h_b  = (kb.get("hallucination") or {})
        aq_a = (ka.get("answer_quality") or {})
        aq_b = (kb.get("answer_quality") or {})
        ret  = r.get("retrieval", {})

        reply_a_full = ka.get("reply_full") or ""
        reply_b_full = kb.get("reply_full") or ""

        delta_f = round(
            f_a["faithfulness_score"] - f_b["faithfulness_score"], 4
        ) if f_a.get("faithfulness_score") is not None \
          and f_b.get("faithfulness_score") is not None else None

        delta_h = round(
            h_a["hallucination_risk"] - h_b["hallucination_risk"], 4
        ) if h_a.get("hallucination_risk") is not None \
          and h_b.get("hallucination_risk") is not None else None

        sources = r.get("retrieved_sources", [])
        rows.append({
            # ── identitas ──────────────────────────────────────────
            "test_id":              r["test_id"],
            "query_type":           r.get("query_type", ""),
            "cognitive":            r.get("cognitive", ""),
            "context_note":         r.get("context_note", ""),
            "timestamp":            r.get("timestamp", ""),
            # ── query ──────────────────────────────────────────────
            "query_full":           r.get("query", ""),
            # ── retrieval (Kondisi A) ───────────────────────────────
            "retrieved_sources":    "; ".join(sources),
            "precision_at_k":       ret.get("precision_at_k"),
            "recall_at_k":          ret.get("recall_at_k"),
            "mean_similarity":      ret.get("mean_similarity"),
            # ── Kondisi A ──────────────────────────────────────────
            "A_reply_full":         reply_a_full,
            "A_reply_len":          len(reply_a_full),
            "A_faithfulness":       f_a.get("faithfulness_score"),
            "A_entailment":         f_a.get("entailment_score"),
            "A_hallucination_risk": h_a.get("hallucination_risk"),
            "A_risk_label":         h_a.get("risk_label"),
            "A_answer_correct":     ka.get("answer_correct"),
            "A_answer_feedback":    aq_a.get("feedback", ""),
            # ── Kondisi B ──────────────────────────────────────────
            "B_reply_full":         reply_b_full,
            "B_reply_len":          len(reply_b_full),
            "B_faithfulness":       f_b.get("faithfulness_score"),
            "B_entailment":         f_b.get("entailment_score"),
            "B_hallucination_risk": h_b.get("hallucination_risk"),
            "B_risk_label":         h_b.get("risk_label"),
            "B_answer_correct":     kb.get("answer_correct"),
            "B_answer_feedback":    aq_b.get("feedback", ""),
            # ── delta ──────────────────────────────────────────────
            "delta_faithfulness":   delta_f,
            "delta_hallucination":  delta_h,
        })

    if rows:
        with open(response_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys(),
                                    quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

    return response_path


def save_results(
    results: List[Dict],
    aggregates: Dict,
    offline: Dict,
    output_dir: str,
) -> Tuple[str, str, str, str, str]:
    """
    Simpan hasil evaluasi. Revisi pasca-sidang menambahkan eval_TS.xlsx
    (evaluation/excel_report.py) yang merekam SETIAP perhitungan metrik,
    respons LLM utuh A vs B, chunk (file+topik), lokasi frasa uncertainty/
    contradiction, verdict entailment per klaim, dan leksikon revisi.

    Returns (json, csv, txt, responses_csv, xlsx).
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path     = os.path.join(output_dir, f"eval_{ts}.json")
    csv_path      = os.path.join(output_dir, f"eval_{ts}.csv")
    txt_path      = os.path.join(output_dir, f"eval_{ts}.txt")
    xlsx_path     = os.path.join(output_dir, f"eval_{ts}.xlsx")
    response_path = _save_response_log(results, output_dir, ts)

    eval_config = {
        "model_llm":            f"ollama/{OLLAMA_CHAT_MODEL}",
        "model_embedding":      f"ollama/{OLLAMA_EMBED_MODEL} (768 dim)",
        "K (Top-K)":            TOP_K,
        "θ  (Precision/Recall, Pers. 2 & 4)": RELEVANCE_THRESHOLD,
        "θc (Coverage, Pers. 8)":             COVERAGE_THRESHOLD,
        "ukuran chunk (char)":  RAG_CHUNK_SIZE,
        "konteks per chunk di prompt (char)": 600,
        "bobot Faithfulness":   "0,70×Entailment + 0,30×KWoverlap (Pers. 15)",
        "bobot HallRisk":       "0,65×(1-F) + 0,20×Contradiction + 0,15×Uncertainty (Pers. 18)",
        "Kondisi B":            "prompt buta Lampiran 4 (tanpa RAG/profil/instruksi)",
        "jumlah kasus":         len(results),
        "timestamp":            ts,
    }
    try:
        build_excel(results, aggregates, eval_config, xlsx_path)
    except Exception as exc:                             # jangan gugurkan run
        logger.error("Gagal membuat laporan Excel: %s", exc)
        xlsx_path = f"(gagal: {exc})"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"results": results, "aggregates": aggregates, "offline": offline},
            fh, ensure_ascii=False, indent=2, default=str,
        )

    flat_rows = []
    for r in results:
        ret = r.get("retrieval", {})
        ka  = r.get("kondisi_a") or {}
        kb  = r.get("kondisi_b") or {}
        f_a = (ka.get("faithfulness") or {})
        h_a = (ka.get("hallucination") or {})
        f_b = (kb.get("faithfulness") or {})
        h_b = (kb.get("hallucination") or {})
        delta_f = round(f_a["faithfulness_score"] - f_b["faithfulness_score"], 4) \
                  if f_a.get("faithfulness_score") is not None \
                  and f_b.get("faithfulness_score") is not None else None
        delta_h = round(h_a["hallucination_risk"] - h_b["hallucination_risk"], 4) \
                  if h_a.get("hallucination_risk") is not None \
                  and h_b.get("hallucination_risk") is not None else None
        flat_rows.append({
            "test_id":            r["test_id"],
            "query_type":         r.get("query_type", ""),
            "cognitive":          r["cognitive"],
            "query":              r["query"][:80],
            "precision_at_k":     ret.get("precision_at_k"),
            "recall_at_k":        ret.get("recall_at_k"),
            "mean_similarity":    ret.get("mean_similarity"),
            "coverage":           ret.get("coverage"),
            "source_diversity":   ret.get("source_diversity"),
            "A_faithfulness":     f_a.get("faithfulness_score"),
            "A_entailment":       f_a.get("entailment_score"),
            "A_hallucination":    h_a.get("hallucination_risk"),
            "A_risk_label":       h_a.get("risk_label"),
            "A_correct":          ka.get("answer_correct"),
            "B_faithfulness":     f_b.get("faithfulness_score"),
            "B_entailment":       f_b.get("entailment_score"),
            "B_hallucination":    h_b.get("hallucination_risk"),
            "B_risk_label":       h_b.get("risk_label"),
            "B_correct":          kb.get("answer_correct"),
            "delta_faith":        delta_f,
            "delta_hall":         delta_h,
        })

    if flat_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)

    # TXT report
    ret  = aggregates.get("retrieval", {})
    ka   = aggregates.get("kondisi_a", {})
    kb   = aggregates.get("kondisi_b", {})
    kom  = aggregates.get("komparasi", {})
    aa   = ka.get("answer_accuracy", {})
    ba   = kb.get("answer_accuracy", {})
    hda  = ka.get("hallucination_dist", {})
    hdb  = kb.get("hallucination_dist", {})

    lines = [
        "=" * 70,
        "  RAG EVALUATION — KONDISI A vs KONDISI B",
        f"  Generated : {ts}",
        f"  Faithfulness: LLM-as-Judge Entailment (v4)",
        f"  Kondisi B   : Prompt buta (tanpa RAG, profil, CT label)",
        f"  LLM  : ollama/{OLLAMA_CHAT_MODEL}",
        f"  Embed: ollama/{OLLAMA_EMBED_MODEL}",
        f"  N    : {aggregates.get('n_tested')}",
        "=" * 70, "",
        "── KONDISI A: RETRIEVAL ─────────────────────────────────────────────",
        f"  Precision@K      : {ret.get('avg_precision_at_k')}",
        f"  Recall@K         : {ret.get('avg_recall_at_k')}",
        f"  Mean Similarity  : {ret.get('avg_mean_similarity')}",
        f"  Coverage         : {ret.get('avg_coverage')}",
        f"  Source Diversity : {ret.get('avg_source_diversity')}",
        "",
        "── PERBANDINGAN GENERASI (A vs B) ───────────────────────────────────",
        f"  {'Metrik':<22} {'Kondisi A':>12} {'Kondisi B':>12} {'Delta (A-B)':>14}",
        f"  {'-'*22} {'-'*12} {'-'*12} {'-'*14}",
        f"  {'Faithfulness':<22} {str(ka.get('avg_faithfulness','-')):>12} "
        f"{str(kb.get('avg_faithfulness','-')):>12} "
        f"{str(kom.get('delta_faithfulness','-')):>14}",
        f"  {'Hallucination Risk':<22} {str(ka.get('avg_hallucination_risk','-')):>12} "
        f"{str(kb.get('avg_hallucination_risk','-')):>12} "
        f"{str(kom.get('delta_hallucination_risk','-')):>14}",
        f"  {'Answer Accuracy':<22} {str(aa.get('accuracy','-')):>12} "
        f"{str(ba.get('accuracy','-')):>12} "
        f"{str(kom.get('delta_answer_accuracy','-')):>14}",
        "",
        "── DISTRIBUSI RISIKO HALUSINASI ─────────────────────────────────────",
        f"  {'Kat':<10} {'A(n)':>8} {'A(%)':>8} {'B(n)':>8} {'B(%)':>8}",
        f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ]
    for cat in ["rendah", "sedang", "tinggi"]:
        da = hda.get(cat, {}); db = hdb.get(cat, {})
        lines.append(
            f"  {cat.upper():<10} {da.get('n','-'):>8} {da.get('pct','-'):>7}% "
            f"{db.get('n','-'):>8} {db.get('pct','-'):>7}%"
        )
    lines += [
        "",
        "── EFFECT SIZE ──────────────────────────────────────────────────────",
        f"  Cohen's d (Faithfulness)     : {kom.get('cohen_d_faithfulness','-')}",
        f"  Faithfulness improvement (%) : {kom.get('relative_faith_improvement_pct','-')}",
        f"  Hallucination reduction (%)  : {kom.get('relative_hall_reduction_pct','-')}",
        "",
        "── ANSWER ACCURACY ──────────────────────────────────────────────────",
        f"  Kondisi A : {aa.get('accuracy')}  ({aa.get('correct')}/{aa.get('total_evaluated')})",
        f"  Kondisi B : {ba.get('accuracy')}  ({ba.get('correct')}/{ba.get('total_evaluated')})",
        "", "── PER TEST CASE ────────────────────────────────────────────────────",
    ]
    for r in results:
        ka_r = r.get("kondisi_a") or {}; kb_r = r.get("kondisi_b") or {}
        fa   = (ka_r.get("faithfulness") or {}); fb = (kb_r.get("faithfulness") or {})
        ha   = (ka_r.get("hallucination") or {}); hb = (kb_r.get("hallucination") or {})
        ret_r = r.get("retrieval", {})
        df = round(fa["faithfulness_score"] - fb["faithfulness_score"], 3) \
             if fa.get("faithfulness_score") is not None \
             and fb.get("faithfulness_score") is not None else "-"
        ent_a = fa.get("entailment_score"); ent_b = fb.get("entailment_score")
        lines += [
            f"  [{r['test_id']:03d}] {r['query'][:60]}…",
            f"        P@K={ret_r.get('precision_at_k','-')}  "
            f"R@K={ret_r.get('recall_at_k','-')}  "
            f"MeanSim={ret_r.get('mean_similarity','-')}",
            f"        A: Faith={fa.get('faithfulness_score','-')}"
            + (f" (ent={ent_a:.2f})" if ent_a is not None else "")
            + f"  Hall={ha.get('hallucination_risk','-')} [{ha.get('risk_label','-')}]"
            + f"  {'✅' if ka_r.get('answer_correct') else '❌' if ka_r.get('answer_correct') is False else '?'}",
            f"        B: Faith={fb.get('faithfulness_score','-')}"
            + (f" (ent={ent_b:.2f})" if ent_b is not None else "")
            + f"  Hall={hb.get('hallucination_risk','-')} [{hb.get('risk_label','-')}]"
            + f"  {'✅' if kb_r.get('answer_correct') else '❌' if kb_r.get('answer_correct') is False else '?'}"
            + f"  ΔFaith={df}",
            "",
        ]

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return json_path, csv_path, txt_path, response_path, xlsx_path