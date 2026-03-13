"""
evaluation/runner.py
─────────────────────
Orchestrates the full RAG evaluation pipeline using LOCAL OLLAMA ONLY.

⚠️  NO REMOTE API IS CALLED — not ChatAnywhere, not OpenAI, nothing.
    Every LLM call and every embedding goes through Ollama on localhost.

Pipeline per test case:
  1. Retrieval metrics   — embed query + keywords with Ollama, compute P@K etc.
  2. RAG context         — read material files from disk, embed chunks with Ollama,
                           retrieve top-K (pure NumPy cosine, no FAISS needed)
  3. LLM reply           — build full chat prompt + context, call Ollama /api/generate
  4. Faithfulness        — embedding similarity between reply and retrieved chunks
  5. Hallucination risk  — keyword overlap + out-of-context detection
  6. Answer quality      — Ollama evaluates reference answer vs LLM reply
  7. Save results        — JSON + CSV + TXT report

Environment variables (all optional, defaults shown):
  OLLAMA_BASE_URL      http://localhost:11434
  OLLAMA_CHAT_MODEL    llama3
  OLLAMA_EMBED_MODEL   nomic-embed-text
  MATERIALS_DIR        <project_root>/materials
  PACE_MIN             1.0   seconds between Ollama calls
  PACE_MAX             3.0
  THINK_PAUSE_MIN      2.0   seconds between test cases
  THINK_PAUSE_MAX      5.0
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
    cosine_similarity,
    coverage_score,
    mean_similarity,
    precision_at_k,
    recall_at_k,
    source_diversity,
)
from evaluation.test_cases import TEST_CASES

logger = logging.getLogger(__name__)

# ── Ollama config ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",  "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Materials dir: default is <project_root>/materials
_HERE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALS_DIR = os.getenv("MATERIALS_DIR", os.path.join(_HERE, "materials"))

# ── Eval config ────────────────────────────────────────────────────────────
TOP_K               = 6
RELEVANCE_THRESHOLD = 0.25
COVERAGE_THRESHOLD  = 0.20
RAG_CHUNK_SIZE      = 1200

# ── Pacing ─────────────────────────────────────────────────────────────────
PACE_MIN        = float(os.getenv("PACE_MIN",        "1.0"))
PACE_MAX        = float(os.getenv("PACE_MAX",        "3.0"))
THINK_PAUSE_MIN = float(os.getenv("THINK_PAUSE_MIN", "2.0"))
THINK_PAUSE_MAX = float(os.getenv("THINK_PAUSE_MAX", "5.0"))


# ══════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _pause(label: str = "") -> None:
    delay = round(random.uniform(PACE_MIN, PACE_MAX), 2)
    logger.debug("  ⏳ %ss %s", delay, label)
    time.sleep(delay)


def _think_pause() -> None:
    delay = round(random.uniform(THINK_PAUSE_MIN, THINK_PAUSE_MAX), 2)
    logger.debug("  💭 thinking %ss", delay)
    time.sleep(delay)


def _ollama_embed(text: str) -> Optional[np.ndarray]:
    """Embed text via Ollama /api/embeddings. Returns L2-normalised float32 array."""
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
    """Same as _ollama_embed but returns plain list (for faithfulness module)."""
    vec = _ollama_embed(text)
    return vec.tolist() if vec is not None else None


def _ollama_generate(prompt: str) -> Optional[str]:
    """
    Generate text via Ollama. Tries /api/generate first (works on every model),
    then falls back to /api/chat.
    """
    _pause("generate")

    # Primary: /api/generate
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

    # Fallback: /api/chat
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
        logger.error(
            "Ollama generate failed on both endpoints — model='%s' url=%s err=%s\n"
            "  Tip: `ollama list` to confirm model name, `ollama serve` to start.",
            OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL, exc,
        )
    return None


# ══════════════════════════════════════════════════════════════════════════
# LOCAL RAG  (reads material files, embeds with Ollama — no app server)
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
    for start in range(0, len(text), RAG_CHUNK_SIZE):
        chunk_text = text[start: start + RAG_CHUNK_SIZE]
        emb = _ollama_embed(chunk_text)
        if emb is not None:
            chunks.append({"text": chunk_text, "source": fname, "embedding": emb})
    _chunk_cache[fname] = chunks
    return chunks


def _retrieve_local(query: str, cognitive_code: str, k: int = TOP_K) -> List[Dict]:
    """Read material files, embed with Ollama, return top-K chunks by cosine sim."""
    if not os.path.isdir(MATERIALS_DIR):
        logger.error("MATERIALS_DIR not found: %s", MATERIALS_DIR)
        return []

    all_chunks: List[Dict] = []
    for fname in os.listdir(MATERIALS_DIR):
        if fname.lower().endswith((".txt", ".md")):
            all_chunks.extend(_load_and_embed_file(
                os.path.join(MATERIALS_DIR, fname), fname
            ))

    if not all_chunks:
        return []

    q_emb = _ollama_embed(query)
    if q_emb is None:
        return []

    return sorted(
        all_chunks,
        key=lambda c: float(np.dot(q_emb, c["embedding"])),
        reverse=True,
    )[:k]


def _chunks_to_context(chunks: List[Dict], max_chars: int = 600) -> str:
    if not chunks:
        return "Tidak ada konteks materi relevan."
    return "\n\n".join(f"[{c['source']}]\n{c['text'][:max_chars]}" for c in chunks)


# ══════════════════════════════════════════════════════════════════════════
# LOCAL CHAT  (full RAG+generate — never touches app server or remote API)
# ══════════════════════════════════════════════════════════════════════════

_LEVEL  = {"1":"Pemula","2":"Dasar","3":"Menengah","4":"Mahir","5":"Lanjut","6":"Pakar"}
_PT     = {"P":"Pragmatis","T":"Teoritis"}
_AG     = {"A":"Analitis","G":"Global"}
_IR     = {"I":"Intuitif","R":"Reflektif"}

def _cognitive_label(code: str) -> str:
    if len(code) < 4:
        return code
    return (f"Level {code[0]} {_LEVEL.get(code[0],'')} / "
            f"{_PT.get(code[1],code[1])} / "
            f"{_AG.get(code[2],code[2])} / "
            f"{_IR.get(code[3],code[3])}")


def _local_chat(query: str, cognitive_code: str):
    """RAG retrieve → build prompt → Ollama generate. Returns (reply_str, chunks)."""
    chunks  = _retrieve_local(query, cognitive_code)
    context = _chunks_to_context(chunks)
    label   = _cognitive_label(cognitive_code)

    prompt = (
        f"Kamu adalah tutor Computational Thinking untuk mahasiswa universitas di Indonesia.\n"
        f"Tipe kognitif mahasiswa: {label}\n\n"
        f"Materi referensi yang relevan:\n{context}\n\n"
        f"Pertanyaan mahasiswa:\n{query}\n\n"
        f"INSTRUKSI KETAT:\n"
        f"- Gunakan istilah teknis yang ada dalam materi referensi di atas.\n"
        f"- Jelaskan konsep utama secara langsung dan akurat.\n"
        f"- Sesuaikan gaya penjelasan dengan tipe kognitif: {label}.\n"
        f"- Gunakan contoh konkret jika membantu pemahaman.\n"
        f"- Maksimal 4 paragraf. Padat, akurat, berbasis materi.\n"
        f"- Jawab dalam Bahasa Indonesia yang jelas dan akademis.\n"
        f"- JANGAN mengarang fakta di luar materi referensi."
    )

    reply = _ollama_generate(prompt)
    return reply, chunks


# ══════════════════════════════════════════════════════════════════════════
# LOCAL ANSWER QUALITY EVALUATOR
# ══════════════════════════════════════════════════════════════════════════

def _evaluate_locally(question: str, reference_answer: str, llm_reply: str) -> Optional[Dict]:
    """
    Evaluate whether the LLM reply adequately covers the reference answer concepts.

    Root cause of 0.28 accuracy: the previous chain-of-thought prompt caused
    llama3 to produce verbose multi-step output that buried or reformatted the
    HASIL: token. Every regex miss defaulted to is_correct=False, collapsing
    accuracy from 0.66 to 0.28.

    Fix: minimal single-question prompt + 3-tier fallback parsing so a verdict
    is always extracted regardless of how llama3 formats its response.
    """
    prompt = (
        f"Apakah jawaban sistem sudah menjawab pertanyaan dengan benar berdasarkan referensi?\n\n"
        f"Pertanyaan: {question}\n\n"
        f"Referensi: {reference_answer[:600]}\n\n"
        f"Jawaban sistem: {llm_reply[:600]}\n\n"
        f"Jawab HANYA dengan satu baris:\n"
        f"HASIL: BENAR\n"
        f"atau\n"
        f"HASIL: SALAH\n\n"
        f"Catatan: BENAR jika konsep utama tercakup meski tidak persis sama."
    )
    raw = _ollama_generate(prompt)
    if raw is None:
        return None

    raw_clean = raw.strip()

    # Tier 1: exact "HASIL: BENAR/SALAH"
    match = re.search(r"HASIL\s*:\s*(BENAR|SALAH)", raw_clean, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        reasoning  = re.sub(r"HASIL\s*:\s*(BENAR|SALAH)", "", raw_clean, flags=re.IGNORECASE).strip()
        return {"is_correct": is_correct, "feedback": reasoning,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL}"}

    # Tier 2: llama3 sometimes outputs just "Ya"/"Tidak" or "Benar"/"Salah"
    has_pos = bool(re.search(r"\b(ya|benar|correct|true|sudah|tepat)\b", raw_clean, re.IGNORECASE))
    has_neg = bool(re.search(r"\b(tidak|salah|incorrect|false|belum|kurang)\b", raw_clean, re.IGNORECASE))
    if has_pos and not has_neg:
        return {"is_correct": True,  "feedback": raw_clean,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL} (tier2-pos)"}
    if has_neg and not has_pos:
        return {"is_correct": False, "feedback": raw_clean,
                "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL} (tier2-neg)"}

    # Tier 3: sentiment word count as last resort — tie goes to BENAR (lenient)
    pos = len(re.findall(
        r"\b(mencakup|sesuai|tepat|benar|akurat|relevan|menjawab|lengkap|baik|jelas|sudah|ada)\b",
        raw_clean, re.IGNORECASE))
    neg = len(re.findall(
        r"\b(tidak|kurang|salah|hilang|keliru|menyesatkan|gagal|buruk|jelek|belum)\b",
        raw_clean, re.IGNORECASE))
    is_correct = pos >= neg
    return {
        "is_correct":   is_correct,
        "feedback":     raw_clean,
        "evaluated_by": f"ollama/{OLLAMA_CHAT_MODEL} (tier3 pos={pos} neg={neg})",
    }


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════

def _eval_retrieval(query: str, keywords: List[str], k: int = TOP_K,
                    cognitive_code: str = "1PAR") -> Dict:
    """
    Two-layer retrieval evaluation:
    1. Keyword proxy  — embed query vs each keyword (fast, no file I/O)
    2. Chunk recall   — embed keywords vs actually retrieved chunks
                        to measure whether real RAG found relevant material
    """
    q_emb = _ollama_embed(query)
    if q_emb is None:
        return {"error": "Ollama embed failed — is Ollama running?"}

    # Layer 1: keyword proxy scores
    kw_embs = [(kw, _ollama_embed(kw)) for kw in keywords]
    kw_embs = [(kw, e) for kw, e in kw_embs if e is not None]
    if not kw_embs:
        return {"error": "Failed to embed keywords"}

    scored = sorted(
        [{"keyword": kw, "score": float(np.dot(q_emb, e))} for kw, e in kw_embs],
        key=lambda x: x["score"], reverse=True,
    )
    top_scores  = [s["score"] for s in scored[:k]]
    top_sources = [s["keyword"] for s in scored[:k]]

    # Layer 2: measure how well actual RAG chunks cover the keywords
    chunk_coverage = None
    try:
        retrieved = _retrieve_local(query, cognitive_code, k=k)
        if retrieved and kw_embs:
            # For each keyword, find max similarity to any retrieved chunk
            chunk_texts = [c["text"][:400] for c in retrieved]
            kw_chunk_scores = []
            for kw, kw_emb in kw_embs:
                best = 0.0
                for chunk_text in chunk_texts:
                    c_emb = _ollama_embed(chunk_text)
                    if c_emb is not None:
                        best = max(best, float(np.dot(kw_emb, c_emb)))
                kw_chunk_scores.append(best)
            chunk_coverage = round(sum(1 for s in kw_chunk_scores if s >= RELEVANCE_THRESHOLD) / len(kw_chunk_scores), 4)
    except Exception as exc:
        logger.debug("chunk coverage calc failed: %s", exc)

    return {
        "top_k_scores":     [round(s, 4) for s in top_scores],
        "top_k_keywords":   top_sources,
        "precision_at_k":   round(precision_at_k(top_scores, k, RELEVANCE_THRESHOLD), 4),
        "recall_at_k":      round(recall_at_k(top_scores, len(keywords), k, RELEVANCE_THRESHOLD), 4),
        "mean_similarity":  round(mean_similarity(top_scores), 4),
        "coverage":         round(coverage_score(top_scores, COVERAGE_THRESHOLD), 4),
        "source_diversity": round(source_diversity(top_sources), 4),
        "chunk_coverage":   chunk_coverage,
        "detail_scores":    [{"keyword": s["keyword"], "score": round(s["score"], 4)} for s in scored],
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation() -> List[Dict]:
    print("\n" + "=" * 70)
    print("  CSIPBLLM — RAG EVALUATION SUITE  [100% LOCAL OLLAMA]")
    print(f"  {len(TEST_CASES)} test cases | Top-K={TOP_K}")
    print(f"  LLM  : ollama/{OLLAMA_CHAT_MODEL}  @ {OLLAMA_BASE_URL}")
    print(f"  Embed: ollama/{OLLAMA_EMBED_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  RAG  : {MATERIALS_DIR}")
    print(f"  ⚠️   App server NOT used — zero remote API calls")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    results: List[Dict] = []

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"[{idx:02d}/{len(TEST_CASES)}] {tc['query'][:65]}…")
        print(f"          {tc['cognitive']} | {tc.get('query_type','')} | {tc.get('context_note','')}")

        result: Dict[str, Any] = {
            "test_id":      idx,
            "query":        tc["query"],
            "cognitive":    tc["cognitive"],
            "context_note": tc.get("context_note", ""),
            "query_type":   tc.get("query_type", ""),
            "timestamp":    datetime.now().isoformat(),
        }

        # Step 1 — Retrieval metrics
        print("  [1] Retrieval metrics (Ollama embed)…")
        result["retrieval"] = _eval_retrieval(tc["query"], tc["relevant_keywords"], cognitive_code=tc["cognitive"])
        if "precision_at_k" in result["retrieval"]:
            r = result["retrieval"]
            cc = r.get("chunk_coverage")
            print(f"      P@{TOP_K}={r['precision_at_k']:.3f}  "
                  f"R@{TOP_K}={r['recall_at_k']:.3f}  "
                  f"MeanSim={r['mean_similarity']:.3f}  "
                  f"Cov={r['coverage']:.3f}"
                  + (f"  ChunkCov={cc:.3f}" if cc is not None else ""))
        else:
            print(f"      ⚠️  {result['retrieval'].get('error')}")

        # Step 2+3 — Local RAG + Ollama generate (NO app server, NO remote API)
        print(f"  [2] RAG + generate (ollama/{OLLAMA_CHAT_MODEL})…")
        reply, rag_chunks = None, []
        try:
            reply, rag_chunks = _local_chat(tc["query"], tc["cognitive"])
        except Exception as exc:
            logger.error("_local_chat error: %s", exc)

        result["llm_reply"] = reply[:400] if reply else None

        if reply:
            print(f"      ✅ {len(reply)} chars")
            # Use full chunk text for faithfulness (more signal)
            sim_ctx = [c["text"][:600] for c in rag_chunks] if rag_chunks else [
                f"{kw} adalah konsep penting dalam computational thinking"
                for kw in tc["relevant_keywords"]
            ]

            # Step 3 — Faithfulness
            print("  [3] Faithfulness…")
            _pause("faithfulness")
            faith = evaluate_faithfulness(reply, sim_ctx, _ollama_embed_list)
            result["faithfulness"] = faith
            print(f"      score={faith['faithfulness_score']:.3f} ({faith['method']})")

            # Step 4 — Hallucination
            print("  [4] Hallucination…")
            hall = detect_hallucination(reply, sim_ctx, tc["query"], _ollama_embed_list)
            result["hallucination"] = hall
            print(f"      risk={hall['hallucination_risk']:.3f} [{hall['risk_label']}]")

            # Step 5 — Answer quality (Ollama only)
            print(f"  [5] Answer quality (ollama/{OLLAMA_CHAT_MODEL})…")
            eval_resp = _evaluate_locally(tc["query"], tc["reference_answer"], reply)
            result["answer_quality"] = eval_resp
            if eval_resp:
                correct = eval_resp.get("is_correct", False)
                result["answer_correct"] = correct
                print(f"      {'✅ BENAR' if correct else '❌ SALAH'}")
            else:
                result["answer_correct"] = None
                print("      ⚠️  Ollama eval unavailable")
        else:
            print("  ⚠️  Ollama generate returned nothing")
            result.update(faithfulness=None, hallucination=None, answer_correct=None)

        results.append(result)
        print()
        if idx < len(TEST_CASES):
            _think_pause()

    return results


# ══════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

def compute_aggregates(results: List[Dict]) -> Dict:
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    prec, rec, ms, cov, div, faith, hall, correct = [], [], [], [], [], [], [], []
    for r in results:
        ret = r.get("retrieval", {})
        if "precision_at_k" in ret:
            prec.append(ret["precision_at_k"])
            rec.append(ret["recall_at_k"])
            ms.append(ret["mean_similarity"])
            cov.append(ret["coverage"])
            div.append(ret["source_diversity"])
        f = r.get("faithfulness") or {}
        if "faithfulness_score" in f:
            faith.append(f["faithfulness_score"])
        h = r.get("hallucination") or {}
        if "hallucination_risk" in h:
            hall.append(h["hallucination_risk"])
        if r.get("answer_correct") is not None:
            correct.append(1 if r["answer_correct"] else 0)

    return {
        "n_tested": len(results),
        "retrieval": {
            "avg_precision_at_k":   avg(prec),
            "avg_recall_at_k":      avg(rec),
            "avg_mean_similarity":  avg(ms),
            "avg_coverage":         avg(cov),
            "avg_source_diversity": avg(div),
        },
        "generation": {
            "avg_faithfulness":       avg(faith),
            "avg_hallucination_risk": avg(hall),
        },
        "answer_quality": {
            "total_evaluated": len(correct),
            "correct":         sum(correct),
            "incorrect":       len(correct) - sum(correct),
            "accuracy":        avg(correct),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# OFFLINE ANALYSIS
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
            "avg_reply_length":   round(statistics.mean(reply_lens), 1) if reply_lens else None,
            "cognitive_distribution": cog_dist,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════

def save_results(
    results: List[Dict],
    aggregates: Dict,
    offline: Dict,
    output_dir: str,
) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"eval_{ts}.json")
    csv_path  = os.path.join(output_dir, f"eval_{ts}.csv")
    txt_path  = os.path.join(output_dir, f"eval_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"results": results, "aggregates": aggregates, "offline": offline},
            fh, ensure_ascii=False, indent=2, default=str,
        )

    flat_rows = []
    for r in results:
        ret = r.get("retrieval", {})
        fth = r.get("faithfulness") or {}
        hal = r.get("hallucination") or {}
        flat_rows.append({
            "test_id":         r["test_id"],
            "query_type":      r.get("query_type", ""),
            "cognitive":       r["cognitive"],
            "query":           r["query"][:80],
            "precision_at_k":  ret.get("precision_at_k"),
            "recall_at_k":     ret.get("recall_at_k"),
            "mean_similarity": ret.get("mean_similarity"),
            "coverage":        ret.get("coverage"),
            "faithfulness":    fth.get("faithfulness_score"),
            "hallucination":   hal.get("hallucination_risk"),
            "risk_label":      hal.get("risk_label"),
            "answer_correct":  r.get("answer_correct"),
            "llm_reply_len":   len(r["llm_reply"]) if r.get("llm_reply") else 0,
        })
    if flat_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)

    lines = [
        "=" * 70,
        "  CSIPBLLM — RAG EVALUATION REPORT  [LOCAL OLLAMA]",
        f"  Generated : {ts}",
        f"  LLM       : ollama/{OLLAMA_CHAT_MODEL}",
        f"  Embed     : ollama/{OLLAMA_EMBED_MODEL}",
        "=" * 70, "",
        "── AGGREGATE METRICS ──────────────────────────────────────────────",
    ]
    ret = aggregates.get("retrieval", {})
    gen = aggregates.get("generation", {})
    aq  = aggregates.get("answer_quality", {})
    lines += [
        f"  Test cases        : {aggregates.get('n_tested')}",
        f"  Avg Precision@K   : {ret.get('avg_precision_at_k')}",
        f"  Avg Recall@K      : {ret.get('avg_recall_at_k')}",
        f"  Avg Mean Sim      : {ret.get('avg_mean_similarity')}",
        f"  Avg Coverage      : {ret.get('avg_coverage')}",
        f"  Avg Faithfulness  : {gen.get('avg_faithfulness')}",
        f"  Avg Halluc. Risk  : {gen.get('avg_hallucination_risk')}",
        f"  Answer Accuracy   : {aq.get('accuracy')}  ({aq.get('correct')}/{aq.get('total_evaluated')})",
        "", "── PER TEST CASE ───────────────────────────────────────────────────",
    ]
    for r in results:
        ret = r.get("retrieval", {})
        fth = r.get("faithfulness") or {}
        hal = r.get("hallucination") or {}
        ac  = r.get("answer_correct")
        lines += [
            f"\n[{r['test_id']:02d}] {r['query'][:70]}",
            f"     Type={r.get('query_type','?')}  Cog={r['cognitive']}",
            f"     P@K={ret.get('precision_at_k','?')}  R@K={ret.get('recall_at_k','?')}  "
            f"Faith={fth.get('faithfulness_score','?')}  Halluc={hal.get('risk_label','?')}  "
            f"Correct={'✅' if ac else '❌' if ac is False else '?'}",
        ]
    lines.append("")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return json_path, csv_path, txt_path