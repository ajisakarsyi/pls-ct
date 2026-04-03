"""
evaluation/faithfulness.py
───────────────────────────
Faithfulness and hallucination-detection metrics.

Key improvements over v1:
- Faithfulness uses per-chunk max similarity (not avg of one concatenated blob)
  so a reply that matches ANY chunk scores well, not just the centroid
- Keyword overlap uses stemming-lite (strip common suffixes) for Indonesian
- Hallucination contradiction score is softened — negation in tutor replies is
  often pedagogical ("bukan X, melainkan Y") not contradictory
- Uncertainty phrases list expanded with common llama3 hedging patterns
"""

import re
from typing import Dict, List, Optional

import numpy as np

from evaluation.metrics import cosine_similarity

# ── Indonesian stop-words ──────────────────────────────────────────────────
_STOPWORDS_ID = {
    "yang", "dan", "atau", "dari", "dalam", "untuk", "adalah", "dengan",
    "pada", "ke", "ini", "itu", "juga", "tidak", "akan", "ada", "oleh",
    "satu", "dapat", "lebih", "sudah", "telah", "bisa", "karena", "maka",
    "sebuah", "tersebut", "namun", "serta", "antara", "sebagai", "seperti",
    "kita", "anda", "saya", "dia", "mereka", "kami", "kalian", "nya",
    "bahwa", "ketika", "saat", "ketika", "selalu", "setiap", "semua",
    "jika", "maka", "meski", "walaupun", "setelah", "sebelum", "sehingga",
}

# ── Uncertainty phrases (expanded for llama3 Indonesian hedging) ───────────
_UNCERTAINTY_PHRASES = [
    "saya tidak yakin", "saya tidak tahu", "mungkin", "kemungkinan",
    "saya pikir", "tampaknya", "sepertinya", "belum pasti", "tidak pasti",
    "perlu dicatat", "perlu diperhatikan", "dalam beberapa kasus",
    "tergantung pada", "bisa jadi", "ada kemungkinan",
    # llama3 common hedges in Indonesian output
    "namun perlu", "perlu diingat", "sebaiknya dikonfirmasi",
]

# ── Negation patterns (only strong contradictions, not pedagogical negation) ─
_NEGATION_PATTERNS = [
    r"\bbertentangan\b", r"\bsebaliknya\b", r"\bkeliru\b",
    r"\bsalah besar\b", r"\btidak benar\b", r"\btidak tepat\b",
]


def _normalize_word(w: str) -> str:
    """Lite Indonesian stemming: strip common prefixes/suffixes for overlap."""
    w = re.sub(r"^(me|di|ke|se|pe|ber|ter|men|mem|pen|pem)", "", w)
    w = re.sub(r"(kan|an|nya|lah|kah|pun)$", "", w)
    return w if len(w) >= 3 else w + "_orig"


def _keyword_overlap(answer: str, context: str) -> float:
    """
    Proportion of significant context words (stemmed) that appear in the answer.
    Uses stemming-lite so 'menggunakan' matches 'gunakan', etc.
    """
    raw_context_words = re.findall(r"\b\w{4,}\b", context.lower())
    context_stems = {
        _normalize_word(w) for w in raw_context_words
        if w not in _STOPWORDS_ID
    }
    if not context_stems:
        return 0.0

    answer_words = re.findall(r"\b\w{4,}\b", answer.lower())
    answer_stems = {_normalize_word(w) for w in answer_words}

    overlap = len(context_stems & answer_stems)
    # Cap at 1.0, use sqrt to be less punishing for partial coverage
    raw = overlap / len(context_stems)
    return min(raw ** 0.7, 1.0)   # soften the penalty curve


def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: List[str],
    get_emb_fn=None,
) -> Dict:
    """
    Faithfulness score ∈ [0, 1].

    Improvements over v1:
    - Embedding similarity: max over per-chunk similarities (not one big blob)
      A reply that closely matches ANY retrieved chunk is faithful.
    - Keyword overlap: stemming-lite + softened penalty curve
    - Weights: 0.65 embedding + 0.35 keyword (embedding more reliable)
    """
    if not retrieved_chunks or not answer:
        return {
            "faithfulness_score": 0.0,
            "keyword_overlap": 0.0,
            "embedding_sim": None,
            "method": "none",
        }

    # Keyword overlap against full concatenated context
    context_concat = " ".join(retrieved_chunks)
    kw_overlap = _keyword_overlap(answer, context_concat)

    # Embedding: max similarity over individual chunks (not one blob)
    emb_sim: Optional[float] = None
    if get_emb_fn is not None:
        try:
            emb_answer = np.array(get_emb_fn(answer[:1500]), dtype="float32")
            chunk_sims = []
            for chunk in retrieved_chunks:
                emb_chunk = get_emb_fn(chunk[:800])
                if emb_chunk is not None:
                    sim = cosine_similarity(emb_answer, np.array(emb_chunk, dtype="float32"))
                    chunk_sims.append(sim)
            if chunk_sims:
                # Weighted: 70% max (best match) + 30% mean (coverage)
                emb_sim = 0.70 * max(chunk_sims) + 0.30 * (sum(chunk_sims) / len(chunk_sims))
        except Exception:
            pass

    if emb_sim is not None:
        score = round(0.65 * emb_sim + 0.35 * kw_overlap, 4)
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
    Hallucination risk ∈ [0, 1].

    Improvements over v1:
    - Uses improved faithfulness (per-chunk max similarity)
    - Contradiction detection only fires on STRONG contradiction patterns,
      not on all negation (pedagogical "bukan X" is not a hallucination)
    - Weights: 0.60 out_of_context + 0.25 contradiction + 0.15 uncertainty
      (out_of_context dominates — if reply matches context, risk is low)
    """
    faith = evaluate_faithfulness(answer, retrieved_chunks, get_emb_fn)
    out_of_context = 1.0 - faith["faithfulness_score"]

    # Contradiction: only strong contradiction phrases, not all negation
    contradiction = 0.0
    answer_lower = answer.lower()
    for pat in _NEGATION_PATTERNS:
        if re.search(pat, answer_lower):
            contradiction = min(contradiction + 0.25, 1.0)

    # Uncertainty flag
    uncertainty_score = 0.25 if any(p in answer_lower for p in _UNCERTAINTY_PHRASES) else 0.0

    risk = round(
        0.60 * out_of_context +
        0.25 * contradiction +
        0.15 * uncertainty_score,
        4,
    )
    risk_label = "TINGGI" if risk >= 0.60 else "SEDANG" if risk >= 0.35 else "RENDAH"

    return {
        "hallucination_risk": risk,
        "risk_label": risk_label,
        "out_of_context": round(out_of_context, 4),
        "contradiction_score": round(contradiction, 4),
        "uncertainty_flag": bool(uncertainty_score),
        "faithfulness": faith["faithfulness_score"],
    }