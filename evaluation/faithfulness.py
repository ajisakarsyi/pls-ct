"""
evaluation/faithfulness.py
───────────────────────────
Faithfulness and hallucination-detection metrics.

These rely on keyword heuristics and (optionally) embedding similarity
via the OpenAI API.
"""

import re
from typing import Dict, List, Optional

import numpy as np

from evaluation.metrics import cosine_similarity

# Indonesian stop-words used when computing keyword overlap
_STOPWORDS_ID = {
    "yang", "dan", "atau", "dari", "dalam", "untuk", "adalah", "dengan",
    "pada", "ke", "ini", "itu", "juga", "tidak", "akan", "ada", "oleh",
    "satu", "dapat", "lebih", "sudah", "telah", "bisa", "karena", "maka",
    "sebuah", "tersebut", "namun", "serta", "antara", "sebagai", "seperti",
}

_UNCERTAINTY_PHRASES = [
    "saya tidak yakin", "saya tidak tahu", "mungkin", "kemungkinan besar",
    "saya pikir", "tampaknya", "sepertinya", "belum pasti", "tidak pasti",
]

_NEGATION_PATTERNS = [
    r"\btidak\s+\w+", r"\bbukan\s+\w+", r"\bsalah\s+\w+",
    r"\bbertentangan\b", r"\bsebaliknya\b", r"\bpadahal\b",
]


def _keyword_overlap(answer: str, context: str) -> float:
    """Proportion of significant context words that appear in the answer."""
    context_words = {
        w for w in re.findall(r"\b\w{4,}\b", context.lower())
        if w not in _STOPWORDS_ID
    }
    if not context_words:
        return 0.0
    overlap = sum(1 for w in context_words if w in answer.lower())
    return min(overlap / len(context_words), 1.0)


def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: List[str],
    get_emb_fn=None,
) -> Dict:
    """
    Two-method faithfulness score:
      1. Keyword overlap    (fast heuristic)
      2. Embedding cosine   (semantic, requires *get_emb_fn*)

    Final score = 0.6 × embedding_sim + 0.4 × keyword_overlap
                  (keyword_only if embeddings unavailable)
    """
    if not retrieved_chunks or not answer:
        return {
            "faithfulness_score": 0.0,
            "keyword_overlap": 0.0,
            "embedding_sim": None,
            "method": "none",
        }

    context = " ".join(retrieved_chunks)
    kw_overlap = _keyword_overlap(answer, context)

    emb_sim: Optional[float] = None
    if get_emb_fn is not None:
        try:
            emb_answer  = np.array(get_emb_fn(answer[:1000]), dtype="float32")
            emb_context = np.array(get_emb_fn(context[:1000]), dtype="float32")
            emb_sim = cosine_similarity(emb_answer, emb_context)
        except Exception:
            pass

    if emb_sim is not None:
        score = round(0.6 * emb_sim + 0.4 * kw_overlap, 4)
        method = "embedding + keyword"
    else:
        score = round(kw_overlap, 4)
        method = "keyword_only"

    return {
        "faithfulness_score": score,
        "keyword_overlap": round(kw_overlap, 4),
        "embedding_sim": round(emb_sim, 4) if emb_sim is not None else None,
        "method": method,
    }


def detect_hallucination(
    answer: str,
    retrieved_chunks: List[str],
    query: str,
    get_emb_fn=None,
) -> Dict:
    """
    Hallucination risk ∈ [0, 1]:
      0.50 × out_of_context  +  0.30 × contradiction  +  0.20 × uncertainty
    """
    faith = evaluate_faithfulness(answer, retrieved_chunks, get_emb_fn)
    out_of_context = 1.0 - faith["faithfulness_score"]

    # Contradiction heuristic
    context_terms = set(re.findall(r"\b\w{5,}\b", " ".join(retrieved_chunks).lower()))
    contradiction = 0.0
    for pat in _NEGATION_PATTERNS:
        for match in re.findall(pat, answer.lower()):
            match_words = set(re.findall(r"\b\w{4,}\b", match))
            if match_words & context_terms:
                contradiction = min(contradiction + 0.2, 1.0)

    # Uncertainty flag
    uncertainty_score = 0.3 if any(p in answer.lower() for p in _UNCERTAINTY_PHRASES) else 0.0

    risk = round(0.50 * out_of_context + 0.30 * contradiction + 0.20 * uncertainty_score, 4)
    risk_label = "TINGGI" if risk >= 0.60 else "SEDANG" if risk >= 0.35 else "RENDAH"

    return {
        "hallucination_risk": risk,
        "risk_label": risk_label,
        "out_of_context": round(out_of_context, 4),
        "contradiction_score": round(contradiction, 4),
        "uncertainty_flag": bool(uncertainty_score),
        "faithfulness": faith["faithfulness_score"],
    }
