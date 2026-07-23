"""
evaluation/runner_v2.py
──────────────────────────────────────────────────────────────────────────
Evaluasi RAG-LLM Chatbot CT IPB — Versi 2 (4 Metrik Final)
100% lokal via Ollama, tidak ada panggilan API eksternal.

PERUBAHAN DARI runner.py (v1):
  - Metrik yang DIHAPUS: Coverage, Source Diversity, Mean Similarity,
    Hallucination Risk. Tidak ada lagi komputasi atau penyimpanan
    untuk metrik-metrik tersebut.
  - Metrik yang DIPERTAHANKAN: Precision@K, Recall@K, Faithfulness,
    Answer Accuracy.
  - KONDISI GANDA (A dan B) dalam satu kali jalan:
      Kondisi A = RAG aktif + profil kognitif + instruksi tutor
      Kondisi B = prompt minimal tanpa RAG, tanpa profil, tanpa instruksi
  - Reply LLM PENUH (tidak dipotong) disimpan di JSON untuk keperluan
    dokumentasi dan sidang.
  - Struktur JSON output berubah: setiap kasus punya satu blok retrieval
    dan dua blok kondisi (kondisi_a, kondisi_b), masing-masing berisi
    reply penuh + faithfulness + answer_correct.

STRUKTUR OUTPUT JSON:
  {
    "metadata": {...},
    "agregat": {
      "precision_at_k": {...},
      "recall_at_k": {...},
      "faithfulness_a": {...},
      "faithfulness_b": {...},
      "answer_accuracy_a": {...},
      "answer_accuracy_b": {...}
    },
    "hasil": [
      {
        "id": 1,
        "query_type": "gap",
        "cognitive": "2PAR",
        "query": "...(teks penuh)...",
        "precision_at_k": 0.6667,
        "recall_at_k": 0.8000,
        "kondisi_a": {
          "reply": "...(teks penuh, tidak dipotong)...",
          "faithfulness": 0.803,
          "entailment_score": 1.0,
          "claims_evaluated": 6,
          "claims_supported": 6,
          "answer_correct": true,
          "answer_feedback": "..."
        },
        "kondisi_b": {
          "reply": "...",
          "faithfulness": 0.767,
          "entailment_score": 1.0,
          "claims_evaluated": 6,
          "claims_supported": 6,
          "answer_correct": false,
          "answer_feedback": "..."
        }
      },
      ...
    ]
  }

CARA MENJALANKAN:
  cd <project_root>
  python -m evaluation.runner_v2

  atau dengan variabel lingkungan:
  OLLAMA_BASE_URL=http://192.168.x.x:11434 python -m evaluation.runner_v2

VARIABEL LINGKUNGAN (semua opsional):
  OLLAMA_BASE_URL      http://localhost:11434
  OLLAMA_CHAT_MODEL    llama3
  OLLAMA_EMBED_MODEL   nomic-embed-text
  MATERIALS_DIR        <project_root>/materials
  OUTPUT_DIR           <project_root>/evaluation/results
  PACE_MIN             1.0  (detik antar panggilan Ollama)
  PACE_MAX             3.0
  CASE_PAUSE_MIN       2.0  (detik antar kasus uji)
  CASE_PAUSE_MAX       4.0
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

from evaluation.test_cases import TEST_CASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_HERE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALS_DIR = os.getenv("MATERIALS_DIR", os.path.join(_HERE, "materials"))
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",    os.path.join(_HERE, "evaluation", "results"))

TOP_K               = 6      # jumlah chunk yang diambil RAG
RELEVANCE_THRESHOLD = 0.25   # θ untuk menentukan chunk relevan (P@K, R@K)
RAG_CHUNK_SIZE      = 1200   # karakter per chunk

PACE_MIN       = float(os.getenv("PACE_MIN",       "1.0"))
PACE_MAX       = float(os.getenv("PACE_MAX",       "3.0"))
CASE_PAUSE_MIN = float(os.getenv("CASE_PAUSE_MIN", "2.0"))
CASE_PAUSE_MAX = float(os.getenv("CASE_PAUSE_MAX", "4.0"))

# Timeout dan retry — naikkan dari default 60s karena cold-start model bisa 60-120s
EMBED_TIMEOUT    = int(os.getenv("EMBED_TIMEOUT",    "180"))  # detik per percobaan embed
GENERATE_TIMEOUT = int(os.getenv("GENERATE_TIMEOUT", "300"))  # detik per generate
EMBED_MAX_RETRY  = int(os.getenv("EMBED_MAX_RETRY",  "3"))    # maks percobaan embed
RETRY_WAIT       = float(os.getenv("RETRY_WAIT",     "15.0")) # detik tunggu sebelum retry

MAX_CLAIMS          = 6    # maks kalimat yang dievaluasi entailment per reply
MIN_CLAIM_LENGTH    = 40   # char minimum sebuah kalimat dianggap klaim
ENTAILMENT_CONTEXT_CHARS = 1500  # maks chars konteks yang dikirim ke evaluator


# ══════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _pause(label: str = "") -> None:
    delay = round(random.uniform(PACE_MIN, PACE_MAX), 2)
    time.sleep(delay)


def _case_pause() -> None:
    delay = round(random.uniform(CASE_PAUSE_MIN, CASE_PAUSE_MAX), 2)
    time.sleep(delay)


def _embed(text: str, _attempt: int = 1) -> Optional[np.ndarray]:
    """
    Embed teks via Ollama. Kembalikan vektor float32 ternormalisasi L2.

    Timeout dinaikkan ke EMBED_TIMEOUT (default 180s) karena:
    - Cold-start model nomic-embed-text saat pertama kali dipanggil bisa 60-120s
    - GPU yang sibuk setelah pemanggilan llama3 juga bisa memperlambat embed

    Retry otomatis hingga EMBED_MAX_RETRY kali dengan jeda RETRY_WAIT detik.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    except requests.exceptions.Timeout:
        if _attempt < EMBED_MAX_RETRY:
            logger.warning(
                "Embed timeout (percobaan %d/%d) — tunggu %.0fs lalu coba lagi…",
                _attempt, EMBED_MAX_RETRY, RETRY_WAIT,
            )
            time.sleep(RETRY_WAIT)
            return _embed(text, _attempt + 1)
        logger.error(
            "Embed gagal setelah %d percobaan (timeout %ds). "
            "Pastikan Ollama berjalan dan model '%s' sudah ter-pull.",
            EMBED_MAX_RETRY, EMBED_TIMEOUT, OLLAMA_EMBED_MODEL,
        )
        return None
    except Exception as exc:
        if _attempt < EMBED_MAX_RETRY:
            logger.warning("Embed error (percobaan %d/%d): %s", _attempt, EMBED_MAX_RETRY, exc)
            time.sleep(RETRY_WAIT)
            return _embed(text, _attempt + 1)
        logger.error("Embed gagal setelah %d percobaan: %s", EMBED_MAX_RETRY, exc)
        return None


def _generate(prompt: str, timeout: int = None) -> Optional[str]:
    """Generate teks via Ollama. Coba /api/generate, fallback ke /api/chat."""
    if timeout is None:
        timeout = GENERATE_TIMEOUT
    _pause("generate")
    for endpoint, payload in [
        ("/api/generate", {"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False}),
        ("/api/chat",     {"model": OLLAMA_CHAT_MODEL,
                           "messages": [{"role": "user", "content": prompt}],
                           "stream": False}),
    ]:
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}{endpoint}", json=payload, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or
                    data.get("message", {}).get("content") or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.debug("Ollama %s failed: %s", endpoint, exc)
    logger.error("Ollama generate gagal pada kedua endpoint.")
    return None


# ══════════════════════════════════════════════════════════════════════════
# CHUNK CACHE DAN RETRIEVAL LOKAL
# ══════════════════════════════════════════════════════════════════════════

_chunk_cache: Dict[str, List[Dict]] = {}


def _load_file(path: str, fname: str) -> List[Dict]:
    """Load dan embed satu file materi. Hasil di-cache per fname."""
    if fname in _chunk_cache:
        return _chunk_cache[fname]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except Exception as exc:
        logger.warning("Tidak bisa baca %s: %s", path, exc)
        return []
    chunks = []
    for i in range(0, len(text), RAG_CHUNK_SIZE):
        chunk_text = text[i: i + RAG_CHUNK_SIZE]
        emb = _embed(chunk_text)
        if emb is not None:
            chunks.append({"text": chunk_text, "source": fname, "embedding": emb})
    _chunk_cache[fname] = chunks
    return chunks


def _load_all_materials() -> List[Dict]:
    """Load semua file materi dari MATERIALS_DIR."""
    if not os.path.isdir(MATERIALS_DIR):
        logger.error("MATERIALS_DIR tidak ditemukan: %s", MATERIALS_DIR)
        return []
    all_chunks: List[Dict] = []
    for fname in sorted(os.listdir(MATERIALS_DIR)):
        if fname.lower().endswith((".txt", ".md")):
            all_chunks.extend(_load_file(os.path.join(MATERIALS_DIR, fname), fname))
    return all_chunks


def _retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Ambil top-K chunk paling relevan dengan query menggunakan cosine similarity."""
    all_chunks = _load_all_materials()
    if not all_chunks:
        return []
    q_emb = _embed(query)
    if q_emb is None:
        return []
    scored = sorted(
        all_chunks,
        key=lambda c: float(np.dot(q_emb, c["embedding"])),
        reverse=True,
    )
    return scored[:k]


def _chunks_to_context(chunks: List[Dict], max_chars_per_chunk: int = 600) -> str:
    if not chunks:
        return "Tidak ada konteks materi relevan."
    return "\n\n".join(
        f"[{c['source']}]\n{c['text'][:max_chars_per_chunk]}" for c in chunks
    )


def _get_chunk_cosine_scores(q_emb: np.ndarray, chunks: List[Dict]) -> List[float]:
    """Hitung cosine similarity sebenarnya antara query dan tiap chunk yang diambil."""
    return [float(np.dot(q_emb, c["embedding"])) for c in chunks]


# ══════════════════════════════════════════════════════════════════════════
# LABEL KOGNITIF
# ══════════════════════════════════════════════════════════════════════════

_LEVEL = {"1": "Pemula", "2": "Dasar", "3": "Menengah",
          "4": "Mahir",  "5": "Lanjut", "6": "Pakar"}
_PT = {"P": "Pragmatis", "T": "Teoritis"}
_AG = {"A": "Analitis",  "G": "Global"}
_IR = {"I": "Intuitif",  "R": "Reflektif"}


def _cog_label(code: str) -> str:
    if len(code) < 4:
        return code
    return (f"Level {code[0]} {_LEVEL.get(code[0], '')} / "
            f"{_PT.get(code[1], code[1])} / "
            f"{_AG.get(code[2], code[2])} / "
            f"{_IR.get(code[3], code[3])}")


# ══════════════════════════════════════════════════════════════════════════
# METRIK 1 & 2: PRECISION@K DAN RECALL@K
# ══════════════════════════════════════════════════════════════════════════

def _compute_retrieval(query: str, keywords: List[str]) -> Dict:
    """
    Hitung Precision@K dan Recall@K.

    Metode: embed query, embed setiap keyword, hitung cosine similarity
    antara query embedding dan keyword embedding. Keyword dengan skor ≥ θ
    dianggap 'relevan' (terwakili dalam query).

    Precision@K = proporsi keyword relevan dari K keyword teratas
    Recall@K    = proporsi keyword relevan dari semua keyword yang ada
    """
    q_emb = _embed(query)
    if q_emb is None:
        return {"error": "embed query gagal"}

    kw_scores: List[Tuple[str, float]] = []
    for kw in keywords:
        kw_emb = _embed(kw)
        _pause("embed-kw")
        if kw_emb is not None:
            score = float(np.dot(q_emb, kw_emb))
            kw_scores.append((kw, score))

    if not kw_scores:
        return {"error": "embed keywords gagal"}

    # Sort descending, ambil top K
    kw_scores.sort(key=lambda x: x[1], reverse=True)
    top_k     = kw_scores[:TOP_K]
    top_scores = [s for _, s in top_k]

    relevant_in_top = sum(1 for s in top_scores if s >= RELEVANCE_THRESHOLD)
    total_relevant  = sum(1 for _, s in kw_scores if s >= RELEVANCE_THRESHOLD)

    precision = relevant_in_top / TOP_K if TOP_K > 0 else 0.0
    recall    = relevant_in_top / len(keywords) if keywords else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k":    round(recall, 4),
        "top_k_keywords": [kw for kw, _ in top_k],
        "top_k_scores":   [round(s, 4) for s in top_scores],
        "n_keywords":     len(keywords),
        "theta":          RELEVANCE_THRESHOLD,
    }


# ══════════════════════════════════════════════════════════════════════════
# METRIK 3: FAITHFULNESS (LLM-as-Judge Entailment)
# ══════════════════════════════════════════════════════════════════════════

def _stem_id(text: str) -> set:
    """Stemming ringan bahasa Indonesia untuk keyword overlap."""
    STOPWORDS = {
        "yang", "dan", "di", "ke", "dari", "dengan", "adalah", "pada",
        "untuk", "dalam", "ini", "itu", "atau", "juga", "oleh", "tidak",
        "ada", "bisa", "akan", "sudah", "lebih", "jika", "maka", "agar",
    }
    PREFIXES = ("me", "di", "ke", "se", "pe", "ber", "ter", "per")
    SUFFIXES = ("kan", "an", "nya", "lah", "pun", "kah")

    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    stems = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        for suf in SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf) + 3:
                t = t[: -len(suf)]
                break
        for pre in PREFIXES:
            if t.startswith(pre) and len(t) > len(pre) + 3:
                t = t[len(pre):]
                break
        stems.add(t)
    return stems


def _keyword_overlap(reply: str, context: str) -> float:
    """KW_overlap = (|Stems(reply) ∩ Stems(context)| / |Stems(context)|) ^ 0.65"""
    stems_r = _stem_id(reply)
    stems_c = _stem_id(context)
    if not stems_c:
        return 0.0
    ratio = len(stems_r & stems_c) / len(stems_c)
    return ratio ** 0.65


_ENTAILMENT_PROMPT = """\
Kamu adalah evaluator klaim yang ketat dan objektif.

Konteks referensi (dokumen materi CT):
{context}

Klaim yang dievaluasi:
"{claim}"

Apakah klaim ini dapat disimpulkan atau didukung oleh konteks referensi di atas?
Jawab HANYA dengan satu kata: YA atau TIDAK. Tidak ada penjelasan lain."""


def _entailment_verdict(claim: str, context: str) -> Optional[bool]:
    """
    Minta llama3 menilai apakah sebuah klaim dapat disimpulkan dari konteks.
    Kembalikan True (YA), False (TIDAK), atau None (gagal parse).
    """
    prompt = _ENTAILMENT_PROMPT.format(
        context=context[:ENTAILMENT_CONTEXT_CHARS],
        claim=claim,
    )
    raw = _generate(prompt)
    if raw is None:
        return None
    raw_upper = raw.strip().upper()
    if re.search(r"\bYA\b|\bYES\b|\bBENAR\b|\bYEP\b", raw_upper):
        return True
    if re.search(r"\bTIDAK\b|\bNO\b|\bSALAH\b|\bNOPE\b", raw_upper):
        return False
    # Fallback: cek kata pertama
    first_word = raw_upper.split()[0] if raw_upper.split() else ""
    if first_word in {"YA", "YES", "BENAR"}:
        return True
    if first_word in {"TIDAK", "NO", "SALAH"}:
        return False
    return None


def _compute_faithfulness(reply: str, context_chunks: List[str]) -> Dict:
    """
    Hitung Faithfulness menggunakan LLM-as-Judge Entailment.

    Langkah 1: Ekstrak maks MAX_CLAIMS kalimat terpanjang dari reply sebagai klaim.
    Langkah 2: Gabungkan chunk GT menjadi konteks referensi.
    Langkah 3: Evaluasi setiap klaim dengan llama3 → YA/TIDAK.
    Langkah 4: Entailment_score = Σverdict / m
    Langkah 5: KW_overlap = (|stems_reply ∩ stems_context| / |stems_context|)^0.65
    Langkah 6: Faithfulness = 0.70 × Entailment_score + 0.30 × KW_overlap
    """
    # Langkah 1
    sentences = [
        s.strip()
        for s in re.split(r"[.!?\n]", reply)
        if len(s.strip()) >= MIN_CLAIM_LENGTH
    ]
    claims = sorted(sentences, key=len, reverse=True)[:MAX_CLAIMS]

    # Langkah 2
    context = "\n\n".join(context_chunks)

    if not claims:
        kw = _keyword_overlap(reply, context)
        return {
            "faithfulness": round(0.30 * kw, 4),
            "entailment_score": None,
            "kw_overlap": round(kw, 4),
            "claims_evaluated": 0,
            "claims_supported": 0,
            "method": "kw_only (no extractable claims)",
        }

    # Langkah 3-4
    verdicts: List[bool] = []
    for claim in claims:
        v = _entailment_verdict(claim, context)
        if v is not None:
            verdicts.append(v)
        _pause("entailment")

    m = len(verdicts)
    entailment_score = sum(verdicts) / m if m > 0 else 0.0

    # Langkah 5
    kw_overlap = _keyword_overlap(reply, context)

    # Langkah 6
    faithfulness = 0.70 * entailment_score + 0.30 * kw_overlap

    return {
        "faithfulness": round(faithfulness, 4),
        "entailment_score": round(entailment_score, 4),
        "kw_overlap": round(kw_overlap, 4),
        "claims_evaluated": m,
        "claims_supported": sum(verdicts),
        "claims_total_extracted": len(claims),
        "method": "llm_entailment + kw_overlap",
    }


# ══════════════════════════════════════════════════════════════════════════
# METRIK 4: ANSWER ACCURACY
# ══════════════════════════════════════════════════════════════════════════

_ANSWER_EVAL_PROMPT = """\
Kamu adalah penilai jawaban yang ketat dan objektif.

Pertanyaan: {question}

Referensi jawaban:
{reference}

Jawaban sistem yang dinilai:
{reply}

Apakah jawaban sistem sudah menjawab pertanyaan dengan benar berdasarkan referensi?
Jawaban BENAR jika konsep utama tercakup, meskipun tidak harus sama persis kata-katanya.
Jawaban SALAH jika konsep utama hilang, keliru, atau tidak relevan.

Tulis penjelasan singkat (1-2 kalimat), lalu pada baris terakhir tulis TEPAT salah satu:
HASIL: BENAR
HASIL: SALAH"""


def _evaluate_answer(question: str, reference: str, reply: str) -> Dict:
    """Evaluasi ketepatan jawaban LLM terhadap referensi menggunakan llama3."""
    prompt = _ANSWER_EVAL_PROMPT.format(
        question=question,
        reference=reference[:600],
        reply=reply[:600],
    )
    raw = _generate(prompt)
    if raw is None:
        return {"answer_correct": None, "answer_feedback": None}

    raw_clean = raw.strip()

    # Tier 1: exact HASIL: BENAR/SALAH
    match = re.search(r"HASIL\s*:\s*(BENAR|SALAH)", raw_clean, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        feedback   = re.sub(r"HASIL\s*:\s*(BENAR|SALAH)", "", raw_clean,
                             flags=re.IGNORECASE).strip()
        return {"answer_correct": is_correct, "answer_feedback": feedback}

    # Tier 2: kata kunci positif/negatif
    has_pos = bool(re.search(
        r"\b(ya|benar|correct|true|sudah|tepat|sesuai|iya)\b", raw_clean, re.IGNORECASE))
    has_neg = bool(re.search(
        r"\b(tidak|salah|incorrect|false|belum|kurang|keliru)\b", raw_clean, re.IGNORECASE))
    if has_pos and not has_neg:
        return {"answer_correct": True,  "answer_feedback": raw_clean}
    if has_neg and not has_pos:
        return {"answer_correct": False, "answer_feedback": raw_clean}

    # Tier 3: hitung kata positif vs negatif
    pos = len(re.findall(
        r"\b(mencakup|sesuai|tepat|benar|akurat|relevan|menjawab|sudah|ada|jelas)\b",
        raw_clean, re.IGNORECASE))
    neg = len(re.findall(
        r"\b(tidak|kurang|salah|hilang|keliru|gagal|belum|buruk)\b",
        raw_clean, re.IGNORECASE))
    return {"answer_correct": pos >= neg, "answer_feedback": raw_clean}


# ══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════

def _build_prompt_a(query: str, context: str, cognitive_code: str) -> str:
    """Prompt Kondisi A: RAG aktif + profil kognitif + instruksi tutor."""
    label = _cog_label(cognitive_code)
    return (
        f"Kamu adalah tutor Computational Thinking untuk mahasiswa universitas di Indonesia.\n"
        f"Tipe kognitif mahasiswa: {label}\n\n"
        f"Materi referensi yang relevan:\n{context}\n\n"
        f"Pertanyaan mahasiswa:\n{query}\n\n"
        f"INSTRUKSI WAJIB:\n"
        f"- Jawaban hanya boleh berasal dari materi referensi.\n"
        f"- Gunakan istilah persis seperti materi.\n"
        f"- Jelaskan konsep sesuai gaya kognitif mahasiswa ({label}).\n"
        f"- WAJIB berikan satu contoh terapan konkret dengan nilai spesifik.\n"
        f"- Penjelasan MAKSIMAL 4 paragraf. Padat dan langsung ke inti.\n"
        f"- Jawab dalam Bahasa Indonesia yang jelas dan akademis.\n"
        f"- JANGAN tambahkan pertanyaan di akhir."
    )


def _build_prompt_b(query: str) -> str:
    """Prompt Kondisi B: minimal, tanpa RAG, tanpa profil, tanpa instruksi."""
    return (
        f"Kamu adalah asisten AI yang membantu menjawab pertanyaan.\n"
        f"Jawab dalam Bahasa Indonesia.\n\n"
        f"Pertanyaan:\n{query}\n\n"
        f"Jawablah pertanyaan di atas."
    )


# ══════════════════════════════════════════════════════════════════════════
# EVALUASI SATU KASUS
# ══════════════════════════════════════════════════════════════════════════

def _run_one(tc: Dict, idx: int, total: int) -> Dict:
    """Jalankan satu kasus uji: retrieval + Kondisi A + Kondisi B."""
    query    = tc["query"]
    keywords = tc["relevant_keywords"]
    cog_code = tc["cognitive"]
    ref_ans  = tc.get("reference_answer", "")

    print(f"\n[{idx:03d}/{total}] {query[:70]}…")
    print(f"         {cog_code} | {tc.get('query_type','?')} | {tc.get('context_note','')}")

    result: Dict[str, Any] = {
        "id":           idx,
        "query_type":   tc.get("query_type", ""),
        "cognitive":    cog_code,
        "query":        query,           # teks penuh, tidak dipotong
        "timestamp":    datetime.now().isoformat(),
    }

    # ── RETRIEVAL ──────────────────────────────────────────────────────
    print("  [R] Retrieval metrics…", end=" ", flush=True)
    ret = _compute_retrieval(query, keywords)
    if "error" in ret:
        print(f"⚠️  {ret['error']}")
        result["precision_at_k"] = None
        result["recall_at_k"]    = None
    else:
        result["precision_at_k"] = ret["precision_at_k"]
        result["recall_at_k"]    = ret["recall_at_k"]
        print(f"P@K={ret['precision_at_k']:.3f}  R@K={ret['recall_at_k']:.3f}")

    # ── RAG CONTEXT (untuk Kondisi A) ──────────────────────────────────
    chunks  = _retrieve(query)
    context = _chunks_to_context(chunks)
    ctx_list = [c["text"][:600] for c in chunks]  # untuk faithfulness

    # ── KONDISI A ──────────────────────────────────────────────────────
    print("  [A] Kondisi A (RAG + profil kognitif)…", end=" ", flush=True)
    prompt_a = _build_prompt_a(query, context, cog_code)
    reply_a  = _generate(prompt_a)

    if reply_a:
        print(f"{len(reply_a)} chars", end=" ", flush=True)
        faith_a = _compute_faithfulness(reply_a, ctx_list)
        print(f"| Faith={faith_a['faithfulness']:.3f}", end=" ", flush=True)
        _pause("pre-eval-a")
        acc_a   = _evaluate_answer(query, ref_ans, reply_a)
        verdict_a = acc_a.get("answer_correct")
        print(f"| Acc={'✅' if verdict_a else '❌' if verdict_a is False else '?'}")
    else:
        print("⚠️  Ollama tidak merespons")
        faith_a = {"faithfulness": None, "entailment_score": None,
                   "kw_overlap": None, "claims_evaluated": 0,
                   "claims_supported": 0, "method": "timeout"}
        acc_a   = {"answer_correct": None, "answer_feedback": None}

    result["kondisi_a"] = {
        "reply":             reply_a,          # PENUH, tidak dipotong
        "faithfulness":      faith_a["faithfulness"],
        "entailment_score":  faith_a.get("entailment_score"),
        "kw_overlap":        faith_a.get("kw_overlap"),
        "claims_evaluated":  faith_a.get("claims_evaluated", 0),
        "claims_supported":  faith_a.get("claims_supported", 0),
        "answer_correct":    acc_a.get("answer_correct"),
        "answer_feedback":   acc_a.get("answer_feedback"),
    }

    # ── KONDISI B ──────────────────────────────────────────────────────
    print("  [B] Kondisi B (tanpa RAG)…", end=" ", flush=True)
    prompt_b = _build_prompt_b(query)
    reply_b  = _generate(prompt_b)

    if reply_b:
        print(f"{len(reply_b)} chars", end=" ", flush=True)
        # Faithfulness Kondisi B dievaluasi terhadap chunk GT yang SAMA dengan A
        # sehingga perbandingan fair: standar evaluasi identik
        faith_b = _compute_faithfulness(reply_b, ctx_list)
        print(f"| Faith={faith_b['faithfulness']:.3f}", end=" ", flush=True)
        _pause("pre-eval-b")
        acc_b   = _evaluate_answer(query, ref_ans, reply_b)
        verdict_b = acc_b.get("answer_correct")
        print(f"| Acc={'✅' if verdict_b else '❌' if verdict_b is False else '?'}")
    else:
        print("⚠️  Ollama tidak merespons")
        faith_b = {"faithfulness": None, "entailment_score": None,
                   "kw_overlap": None, "claims_evaluated": 0,
                   "claims_supported": 0, "method": "timeout"}
        acc_b   = {"answer_correct": None, "answer_feedback": None}

    result["kondisi_b"] = {
        "reply":             reply_b,          # PENUH, tidak dipotong
        "faithfulness":      faith_b["faithfulness"],
        "entailment_score":  faith_b.get("entailment_score"),
        "kw_overlap":        faith_b.get("kw_overlap"),
        "claims_evaluated":  faith_b.get("claims_evaluated", 0),
        "claims_supported":  faith_b.get("claims_supported", 0),
        "answer_correct":    acc_b.get("answer_correct"),
        "answer_feedback":   acc_b.get("answer_feedback"),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════
# AGREGASI
# ══════════════════════════════════════════════════════════════════════════

def _aggregate(results: List[Dict]) -> Dict:
    def _stats(lst):
        if not lst:
            return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(statistics.mean(lst), 4),
            "sd":   round(statistics.stdev(lst), 4) if len(lst) > 1 else 0.0,
            "min":  round(min(lst), 4),
            "max":  round(max(lst), 4),
            "n":    len(lst),
        }

    pk_list, rk_list = [], []
    fa_list, fb_list = [], []
    acc_a_list, acc_b_list = [], []

    for r in results:
        if r.get("precision_at_k") is not None:
            pk_list.append(r["precision_at_k"])
        if r.get("recall_at_k") is not None:
            rk_list.append(r["recall_at_k"])

        ka = r.get("kondisi_a") or {}
        kb = r.get("kondisi_b") or {}

        if ka.get("faithfulness") is not None:
            fa_list.append(ka["faithfulness"])
        if kb.get("faithfulness") is not None:
            fb_list.append(kb["faithfulness"])
        if ka.get("answer_correct") is not None:
            acc_a_list.append(1 if ka["answer_correct"] else 0)
        if kb.get("answer_correct") is not None:
            acc_b_list.append(1 if kb["answer_correct"] else 0)

    n_correct_a = sum(acc_a_list)
    n_correct_b = sum(acc_b_list)
    n_total_a   = len(acc_a_list)
    n_total_b   = len(acc_b_list)

    faith_a_gt_b = sum(
        1 for r in results
        if (r.get("kondisi_a") or {}).get("faithfulness") is not None
        and (r.get("kondisi_b") or {}).get("faithfulness") is not None
        and r["kondisi_a"]["faithfulness"] > r["kondisi_b"]["faithfulness"]
    )
    faith_comparable = sum(
        1 for r in results
        if (r.get("kondisi_a") or {}).get("faithfulness") is not None
        and (r.get("kondisi_b") or {}).get("faithfulness") is not None
    )

    return {
        "precision_at_k":    _stats(pk_list),
        "recall_at_k":       _stats(rk_list),
        "faithfulness_a":    _stats(fa_list),
        "faithfulness_b":    _stats(fb_list),
        "faithfulness_delta_a_minus_b": round(
            statistics.mean(fa_list) - statistics.mean(fb_list), 4
        ) if fa_list and fb_list else None,
        "faithfulness_a_unggul_b": {
            "jumlah":    faith_a_gt_b,
            "dari":      faith_comparable,
            "proporsi":  round(faith_a_gt_b / faith_comparable, 4) if faith_comparable else None,
        },
        "answer_accuracy_a": {
            "benar": n_correct_a,
            "total": n_total_a,
            "nilai": round(n_correct_a / n_total_a, 4) if n_total_a else None,
        },
        "answer_accuracy_b": {
            "benar": n_correct_b,
            "total": n_total_b,
            "nilai": round(n_correct_b / n_total_b, 4) if n_total_b else None,
        },
        "answer_accuracy_delta_a_minus_b": round(
            n_correct_a / n_total_a - n_correct_b / n_total_b, 4
        ) if n_total_a and n_total_b else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# SIMPAN HASIL
# ══════════════════════════════════════════════════════════════════════════

def _save(results: List[Dict], agregat: Dict, output_dir: str) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"eval_v2_{ts}.json")
    csv_path  = os.path.join(output_dir, f"eval_v2_{ts}.csv")
    qry_path  = os.path.join(output_dir, f"kasus_uji_{ts}.json")

    # ── JSON utama: semua data ─────────────────────────────────────────
    output_json = {
        "metadata": {
            "versi":        "runner_v2",
            "tanggal":      ts,
            "model_llm":    f"ollama/{OLLAMA_CHAT_MODEL}",
            "model_embed":  f"ollama/{OLLAMA_EMBED_MODEL}",
            "total_kasus":  len(TEST_CASES),
            "n_dijalankan": len(results),
            "top_k":        TOP_K,
            "theta":        RELEVANCE_THRESHOLD,
            "chunk_size":   RAG_CHUNK_SIZE,
            "metrik": [
                "precision_at_k",
                "recall_at_k",
                "faithfulness (Kondisi A)",
                "faithfulness (Kondisi B)",
                "answer_accuracy (Kondisi A)",
                "answer_accuracy (Kondisi B)",
            ],
            "catatan": (
                "Reply LLM disimpan penuh tanpa pemotongan. "
                "Coverage, Source Diversity, Mean Similarity, dan Hallucination Risk "
                "dihapus dari evaluasi ini."
            ),
        },
        "agregat": agregat,
        "hasil":   results,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, ensure_ascii=False, indent=2, default=str)

    # ── CSV ringkasan per baris ────────────────────────────────────────
    rows = []
    for r in results:
        ka = r.get("kondisi_a") or {}
        kb = r.get("kondisi_b") or {}
        rows.append({
            "id":             r["id"],
            "query_type":     r.get("query_type", ""),
            "cognitive":      r.get("cognitive", ""),
            "query_80":       r.get("query", "")[:80],
            "precision_at_k": r.get("precision_at_k"),
            "recall_at_k":    r.get("recall_at_k"),
            "faith_a":        ka.get("faithfulness"),
            "faith_b":        kb.get("faithfulness"),
            "acc_a":          ka.get("answer_correct"),
            "acc_b":          kb.get("answer_correct"),
            "reply_a_chars":  len(ka.get("reply") or ""),
            "reply_b_chars":  len(kb.get("reply") or ""),
        })
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # ── JSON kasus uji: ID + query penuh + reply penuh (untuk sidang) ─
    kasus_uji = {
        "metadata": {
            "deskripsi": (
                "Daftar 210 kasus uji beserta pertanyaan penuh dan respons "
                "LLM penuh dari Kondisi A dan Kondisi B. "
                "Tanpa metrik angka — untuk dokumentasi dan keperluan sidang."
            ),
            "tanggal": ts,
            "total":   len(results),
        },
        "kasus": [
            {
                "id":           r["id"],
                "query_type":   r.get("query_type", ""),
                "cognitive":    r.get("cognitive", ""),
                "query":        r.get("query", ""),              # penuh
                "kondisi_a":    {
                    "reply": (r.get("kondisi_a") or {}).get("reply"),  # penuh
                },
                "kondisi_b":    {
                    "reply": (r.get("kondisi_b") or {}).get("reply"),  # penuh
                },
            }
            for r in results
        ],
    }
    with open(qry_path, "w", encoding="utf-8") as fh:
        json.dump(kasus_uji, fh, ensure_ascii=False, indent=2, default=str)

    return json_path, csv_path, qry_path


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def _warmup_check() -> bool:
    """
    Cek koneksi Ollama dan warm-up model sebelum evaluasi dimulai.
    Kirim satu embed pendek untuk memastikan model sudah di-load ke VRAM.
    Ini mencegah timeout pada kasus uji pertama akibat cold-start.
    """
    print("  Memeriksa koneksi Ollama...", end=" ", flush=True)
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"OK — {len(models)} model tersedia")
    except Exception as exc:
        print(f"\n  ⚠️  Tidak bisa terhubung ke Ollama: {exc}")
        print(f"  Pastikan Ollama berjalan di {OLLAMA_BASE_URL}")
        return False

    print(f"  Warm-up embed model '{OLLAMA_EMBED_MODEL}'...", end=" ", flush=True)
    test_vec = _embed("computational thinking")
    if test_vec is None:
        print(f"\n  ❌ Warm-up gagal — model belum ter-pull atau timeout.")
        print(f"     Jalankan: ollama pull {OLLAMA_EMBED_MODEL}")
        return False
    print(f"OK ({len(test_vec)}-dim)")

    print(f"  Warm-up chat model '{OLLAMA_CHAT_MODEL}'...", end=" ", flush=True)
    test_reply = _generate("Balas dengan kata 'siap'.", timeout=120)
    if test_reply is None:
        print(f"\n  ❌ Warm-up LLM gagal.")
        print(f"     Jalankan: ollama pull {OLLAMA_CHAT_MODEL}")
        return False
    print(f"OK")
    return True


def run() -> None:
    print("\n" + "=" * 70)
    print("  LogiCT — RAG EVALUATION SUITE v2  [100% LOCAL OLLAMA]")
    print(f"  {len(TEST_CASES)} kasus uji | K={TOP_K} | θ={RELEVANCE_THRESHOLD}")
    print(f"  LLM  : ollama/{OLLAMA_CHAT_MODEL}  @ {OLLAMA_BASE_URL}")
    print(f"  Embed: ollama/{OLLAMA_EMBED_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  RAG  : {MATERIALS_DIR}")
    print(f"  Metrik: Precision@K, Recall@K, Faithfulness A/B, Answer Accuracy A/B")
    print(f"  Mulai : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Warm-up: pastikan Ollama siap dan model sudah di-load ke VRAM
    if not _warmup_check():
        print("\n  Evaluasi dibatalkan. Perbaiki koneksi Ollama terlebih dahulu.")
        return

    results: List[Dict] = []

    for idx, tc in enumerate(TEST_CASES, 1):
        try:
            result = _run_one(tc, idx, len(TEST_CASES))
            results.append(result)
        except Exception as exc:
            logger.error("Kasus %d error: %s", idx, exc)
            results.append({
                "id": idx,
                "query_type": tc.get("query_type", ""),
                "cognitive": tc.get("cognitive", ""),
                "query": tc.get("query", ""),
                "error": str(exc),
            })

        if idx < len(TEST_CASES):
            _case_pause()

    # Agregasi
    agregat = _aggregate(results)

    # Simpan
    json_p, csv_p, qry_p = _save(results, agregat, OUTPUT_DIR)

    # Ringkasan terminal
    print("\n" + "=" * 70)
    print("  SELESAI")
    print(f"  JSON utama : {json_p}")
    print(f"  CSV        : {csv_p}")
    print(f"  Kasus uji  : {qry_p}")
    print("─" * 70)
    agg = agregat
    print(f"  Precision@K      : {agg['precision_at_k']['mean']}")
    print(f"  Recall@K         : {agg['recall_at_k']['mean']}")
    print(f"  Faithfulness A   : {agg['faithfulness_a']['mean']}")
    print(f"  Faithfulness B   : {agg['faithfulness_b']['mean']}")
    print(f"  Δ Faithfulness   : {agg['faithfulness_delta_a_minus_b']}")
    print(f"  Faith A>B        : {agg['faithfulness_a_unggul_b']['jumlah']}"
          f"/{agg['faithfulness_a_unggul_b']['dari']}"
          f" ({agg['faithfulness_a_unggul_b']['proporsi']})")
    acc_a = agg["answer_accuracy_a"]
    acc_b = agg["answer_accuracy_b"]
    print(f"  Accuracy A       : {acc_a['nilai']} ({acc_a['benar']}/{acc_a['total']})")
    print(f"  Accuracy B       : {acc_b['nilai']} ({acc_b['benar']}/{acc_b['total']})")
    print(f"  Δ Accuracy       : {agg['answer_accuracy_delta_a_minus_b']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()
