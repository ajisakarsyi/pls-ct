"""
evaluation/faithfulness.py
───────────────────────────
Faithfulness dan hallucination detection — v4 (LLM-as-Judge Entailment).

PERUBAHAN DARI v3:
─────────────────
v3 menggunakan embedding cosine similarity antara kalimat respons dan kalimat
chunk GT. Masalahnya: llama3 yang sudah tahu materi CT dari training data-nya
akan menghasilkan respons yang semantically mirip dengan chunk GT bahkan TANPA
RAG, karena vocabulary dan konsep CT sudah ada dalam pengetahuannya.

v4 menggunakan LLM-as-Judge entailment — pendekatan yang digunakan oleh
Es et al. (2024) dalam RAGAS dan Saad-Falcon et al. (2024) dalam ARES:
  1. Pecah respons LLM menjadi kalimat-kalimat klaim (claims).
  2. Untuk setiap klaim, tanya llama3: "Apakah klaim ini dapat disimpulkan
     dari konteks berikut? Jawab YA atau TIDAK."
  3. Faithfulness = proporsi klaim yang mendapat YA.

Mengapa ini membedakan Kondisi A dari Kondisi B:
  - Kondisi A: respons dibangun di atas konteks RAG → klaim-klaimnya
    konsisten dengan chunk GT → proporsi YA tinggi.
  - Kondisi B: respons dibangun dari pengetahuan parametrik saja → klaim-
    klaimnya mungkin benar secara umum tapi tidak ada di chunk GT spesifik
    (trace numerik, soal coklat bebek, pseudocode FPB versi tertentu, dsb.)
    → proporsi YA lebih rendah.

Faithfulness final = 0.70 * entailment_score + 0.30 * keyword_overlap
Keyword overlap dipertahankan sebagai sinyal komplementer yang tidak bergantung
pada kualitas evaluasi llama3.

Fallback: jika Ollama tidak tersedia untuk entailment, mundur ke
sentence-level embedding (v3). Jika embedding juga gagal, pakai keyword only.
"""

import logging
import re
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from evaluation.metrics import cosine_similarity

logger = logging.getLogger(__name__)

# ── Indonesian stop-words ──────────────────────────────────────────────────
_STOPWORDS_ID = {
    "yang", "dan", "atau", "dari", "dalam", "untuk", "adalah", "dengan",
    "pada", "ke", "ini", "itu", "juga", "tidak", "akan", "ada", "oleh",
    "satu", "dapat", "lebih", "sudah", "telah", "bisa", "karena", "maka",
    "sebuah", "tersebut", "namun", "serta", "antara", "sebagai", "seperti",
    "kita", "anda", "saya", "dia", "mereka", "kami", "kalian",
    "bahwa", "ketika", "saat", "selalu", "setiap", "semua",
    "jika", "meski", "walaupun", "setelah", "sebelum", "sehingga",
}

_UNCERTAINTY_PHRASES = [
    "saya tidak yakin", "saya tidak tahu", "mungkin", "kemungkinan",
    "tampaknya", "sepertinya", "belum pasti", "tidak pasti",
    "perlu dicatat", "tergantung pada", "bisa jadi",
    "namun perlu", "perlu diingat",
]

_CONTRADICTION_PATTERNS = [
    r"\bbertentangan\b", r"\bsebaliknya\b", r"\bkeliru\b",
    r"\bsalah besar\b", r"\btidak benar\b", r"\btidak tepat\b",
    r"\bmenyesatkan\b",
]

# ── Ollama helper (tanpa import dari runner untuk menghindari circular) ─────
def _ollama_generate_local(prompt: str, base_url: str = "http://localhost:11434",
                            model: str = "llama3") -> Optional[str]:
    """Minimal Ollama generate — dipakai hanya untuk entailment judgment."""
    import os
    try:
        import requests
        _base = os.getenv("OLLAMA_BASE_URL", base_url)
        _model = os.getenv("OLLAMA_CHAT_MODEL", model)
        resp = requests.post(
            f"{_base}/api/generate",
            json={"model": _model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.debug("Ollama entailment call failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str, min_len: int = 40) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for s in raw:
        for sub in s.split('\n'):
            sub = sub.strip()
            if len(sub) >= min_len:
                sentences.append(sub)
    return sentences if sentences else [text[:500]]


def _normalize_word(w: str) -> str:
    w = re.sub(r'^(me|di|ke|se|pe|ber|ter|men|mem|pen|pem)', '', w)
    w = re.sub(r'(kan|an|nya|lah|kah|pun)$', '', w)
    return w if len(w) >= 3 else w + '_orig'


def _keyword_overlap(answer: str, context: str) -> float:
    """
    ═══════════════════════════════════════════════════════════════
    KEYWORD OVERLAP — Persamaan 3.13 dalam skripsi
    File: evaluation/faithfulness.py → _keyword_overlap()
    ═══════════════════════════════════════════════════════════════

    Q: "Mengapa ada eksponen 0.65 di keyword overlap?"
    A: Prinsip sublinear scaling dari Manning et al. (2008) Section 6.4.1.
       Relevansi tidak meningkat proporsional dengan jumlah term yang overlap.
       Eksponen 0.65 memoderasi nilai tinggi agar tidak mendominasi skor akhir
       ketika respons mengulang terminologi teknis yang memang harus digunakan.
       Contoh: raw=0.8 → 0.8^0.65 = 0.856 (bukan 0.8, bukan 1.0)

    Q: "Mengapa minimum panjang kata 4 karakter (\\w{4,})?"
    A: Untuk menyaring kata pendek yang tidak informatif (misal: "dan",
       "itu", "ini") bahkan sebelum stopword filtering.

    Q: "Apa yang dimaksud Stems(answer) ∩ Stems(context)?"
    A: Setelah stemming ringan (hapus prefiks me-/di-/ke- dan sufiks -kan/-an/-nya),
       hitung intersection antara himpunan stem kata dari respons LLM dan
       himpunan stem kata dari konteks GT. Ini mengukur tumpang tindih kosakata.

    Formula (Pers. 3.13): KWoverlap = (|Stems(ans) ∩ Stems(ctx)| / |Stems(ctx)|)^0.65
    ═══════════════════════════════════════════════════════════════
    """
    raw_ctx = re.findall(r'\b\w{4,}\b', context.lower())
    ctx_stems = {_normalize_word(w) for w in raw_ctx if w not in _STOPWORDS_ID}
    if not ctx_stems:
        return 0.0
    ans_words = re.findall(r'\b\w{4,}\b', answer.lower())
    ans_stems = {_normalize_word(w) for w in ans_words}
    overlap = len(ctx_stems & ans_stems)
    raw = overlap / len(ctx_stems)
    # Eksponen 0.65: sublinear scaling (Manning et al. 2008, Section 6.4.1)
    return min(raw ** 0.65, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE ENTAILMENT  (Jalur 3)
# ══════════════════════════════════════════════════════════════════════════

_ENTAILMENT_PROMPT = """\
Kamu adalah evaluator yang menilai apakah sebuah klaim didukung oleh konteks yang diberikan.

Konteks dari dokumen materi:
\"\"\"
{context}
\"\"\"

Klaim yang dievaluasi:
\"{claim}\"

INSTRUKSI:
- Jawab YA jika klaim tersebut dapat disimpulkan atau didukung oleh informasi dalam konteks di atas.
- Jawab TIDAK jika klaim tersebut tidak ada dalam konteks, berlawanan dengan konteks, atau memerlukan pengetahuan di luar konteks.
- Jawab HANYA dengan satu kata: YA atau TIDAK. Tidak perlu penjelasan.

Jawaban:"""


def _judge_single_claim(claim: str, context_str: str) -> Optional[bool]:
    """
    Tanya llama3: apakah claim ini didukung oleh context_str?
    Returns True (didukung), False (tidak didukung), None (gagal parse).
    """
    prompt = _ENTAILMENT_PROMPT.format(
        context=context_str[:1200],
        claim=claim[:300],
    )
    raw = _ollama_generate_local(prompt)
    if raw is None:
        return None
    raw_clean = raw.strip().upper()
    # Parse verdict — toleran terhadap variasi format llama3
    if re.search(r'\bYA\b', raw_clean):
        return True
    if re.search(r'\bTIDAK\b', raw_clean):
        return False
    # Fallback: cek kata positif/negatif umum
    if re.search(r'\b(YES|SUPPORTED|BENAR|TEPAT|SESUAI)\b', raw_clean):
        return True
    if re.search(r'\b(NO|NOT|UNSUPPORTED|SALAH|TIDAK)\b', raw_clean):
        return False
    return None


def _entailment_score(
    answer: str,
    retrieved_chunks: List[str],
    max_claims: int = 6,
    pause_sec: float = 0.5,
) -> Dict:
    """
    Hitung entailment-based faithfulness:
      - Pecah respons menjadi klaim (kalimat)
      - Untuk setiap klaim, tanya llama3 apakah didukung oleh chunk GT
      - Faithfulness_entailment = proporsi klaim yang didukung

    max_claims: batasi jumlah klaim untuk efisiensi waktu evaluasi.
    pause_sec : jeda antar panggilan Ollama untuk menghindari overload.
    """
    claims = _split_sentences(answer)
    # Ambil max_claims kalimat yang paling informatif (paling panjang)
    claims = sorted(claims, key=len, reverse=True)[:max_claims]

    if not claims:
        return {"entailment_score": 0.0, "supported": 0, "total": 0,
                "detail": [], "method": "entailment_empty"}

    # Gabung semua chunk sebagai satu konteks (dibatasi 1500 karakter)
    context_str = "\n\n".join(retrieved_chunks)[:1500]

    results = []
    for claim in claims:
        verdict = _judge_single_claim(claim, context_str)
        results.append({
            "claim":     claim[:120],
            "supported": verdict,
        })
        if pause_sec > 0:
            time.sleep(pause_sec)

    evaluated = [r for r in results if r["supported"] is not None]
    supported  = sum(1 for r in evaluated if r["supported"] is True)
    total      = len(evaluated)

    score = round(supported / total, 4) if total > 0 else 0.0

    return {
        "entailment_score": score,
        "supported":        supported,
        "total_evaluated":  total,
        "total_claims":     len(claims),
        "detail":           results,
        "method":           "llm_entailment",
    }


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDING FALLBACK (v3 — dipakai jika Ollama entailment gagal)
# ══════════════════════════════════════════════════════════════════════════

def _embedding_faithfulness(
    answer: str,
    retrieved_chunks: List[str],
    get_emb_fn: Callable,
) -> Optional[float]:
    """Sentence-level embedding similarity — fallback dari entailment."""
    try:
        reply_sentences = _split_sentences(answer)[:8]
        chunk_sentences = []
        for chunk in retrieved_chunks:
            chunk_sentences.extend(_split_sentences(chunk))
        chunk_sentences = chunk_sentences[:20]

        if not reply_sentences or not chunk_sentences:
            return None

        chunk_embs = []
        for cs in chunk_sentences:
            e = get_emb_fn(cs[:400])
            if e is not None:
                chunk_embs.append(np.array(e, dtype='float32'))

        if not chunk_embs:
            return None

        per_sentence_sims = []
        for rs in reply_sentences:
            rs_emb = get_emb_fn(rs[:400])
            if rs_emb is None:
                continue
            rs_vec = np.array(rs_emb, dtype='float32')
            best = max(
                (cosine_similarity(rs_vec, ce) for ce in chunk_embs),
                default=0.0,
            )
            per_sentence_sims.append(best)

        if not per_sentence_sims:
            return None

        top_sim  = max(per_sentence_sims)
        mean_sim = sum(per_sentence_sims) / len(per_sentence_sims)
        return 0.30 * top_sim + 0.70 * mean_sim

    except Exception as exc:
        logger.warning("Embedding faithfulness error: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: List[str],
    get_emb_fn: Optional[Callable] = None,
    use_entailment: bool = True,
    max_claims: int = 6,
) -> Dict:
    """
    Faithfulness score ∈ [0, 1].

    Strategi v4 (LLM-as-Judge Entailment):
      1. Pecah respons menjadi klaim-klaim (kalimat).
      2. Untuk setiap klaim, llama3 menilai apakah klaim didukung konteks GT.
      3. Faithfulness_entailment = proporsi klaim yang didukung.
      4. Keyword overlap dihitung sebagai sinyal komplementer.
      5. Final: 0.70 * entailment + 0.30 * keyword_overlap

    Fallback ke sentence-level embedding (v3) jika Ollama tidak tersedia.
    Fallback ke keyword-only jika embedding juga gagal.

    Parameters
    ----------
    answer          : respons LLM yang akan dievaluasi
    retrieved_chunks: list chunk teks dari dokumen GT
    get_emb_fn      : fungsi embedding (untuk fallback)
    use_entailment  : set False untuk paksa pakai embedding (debug)
    max_claims      : maks jumlah kalimat yang dievaluasi per respons
    """
    if not retrieved_chunks or not answer:
        return {
            "faithfulness_score": 0.0,
            "keyword_overlap":    0.0,
            "entailment_score":   None,
            "embedding_sim":      None,
            "method":             "none",
            "entailment_detail":  [],
        }

    context_concat = " ".join(retrieved_chunks)
    kw_overlap = _keyword_overlap(answer, context_concat)

    # ── Coba LLM-as-Judge entailment ────────────────────────────────────
    entailment_result = None
    if use_entailment:
        entailment_result = _entailment_score(
            answer, retrieved_chunks, max_claims=max_claims
        )

    if entailment_result and entailment_result["total_evaluated"] > 0:
        ent_score = entailment_result["entailment_score"]

        # ── Faithfulness Final (Persamaan 3.14 dalam skripsi) ─────────────
        # Faithfulness = 0.70 × EntailmentScore + 0.30 × KWOverlap
        #
        # Bobot 0.70 untuk entailment (sinyal semantik) — dominan karena
        # sesuai dengan definisi RAGAS (Es et al. 2024): F = |V|/|S|
        # Bobot 0.30 untuk keyword overlap (sinyal leksikal) — sebagai
        # sinyal deterministik pelengkap yang tidak bergantung pada llama3
        score  = round(0.70 * ent_score + 0.30 * kw_overlap, 4)
        method = "llm_entailment + keyword"
        return {
            "faithfulness_score": score,
            "keyword_overlap":    round(kw_overlap, 4),
            "entailment_score":   round(ent_score, 4),
            "embedding_sim":      None,
            "method":             method,
            "entailment_detail":  entailment_result.get("detail", []),
            "claims_supported":   entailment_result["supported"],
            "claims_evaluated":   entailment_result["total_evaluated"],
        }

    # ── Fallback ke sentence-level embedding ────────────────────────────
    logger.info("Entailment unavailable — falling back to embedding similarity")
    emb_sim = None
    if get_emb_fn is not None:
        emb_sim = _embedding_faithfulness(answer, retrieved_chunks, get_emb_fn)

    if emb_sim is not None:
        score  = round(0.65 * emb_sim + 0.35 * kw_overlap, 4)
        method = "embedding + keyword (entailment fallback)"
        return {
            "faithfulness_score": score,
            "keyword_overlap":    round(kw_overlap, 4),
            "entailment_score":   None,
            "embedding_sim":      round(emb_sim, 4),
            "method":             method,
            "entailment_detail":  [],
        }

    # ── Fallback ke keyword only ─────────────────────────────────────────
    return {
        "faithfulness_score": round(kw_overlap, 4),
        "keyword_overlap":    round(kw_overlap, 4),
        "entailment_score":   None,
        "embedding_sim":      None,
        "method":             "keyword_only",
        "entailment_detail":  [],
    }


def detect_hallucination(
    answer: str,
    retrieved_chunks: List[str],
    query: str,
    get_emb_fn: Optional[Callable] = None,
) -> Dict:
    """
    ═══════════════════════════════════════════════════════════════
    HALLUCINATION RISK — Persamaan 3.17 dalam skripsi
    File: evaluation/faithfulness.py → detect_hallucination()
    ═══════════════════════════════════════════════════════════════

    Q: "Dari mana komponen OutOfContext?"
    A: OutOfContext = 1 - Faithfulness
       Derivasi matematis langsung dari definisi Faithfulness (Es et al. 2024)
       F = |V|/|S| → komplemen = (|S|-|V|)/|S| = proporsi klaim tidak didukung

    Q: "Mengapa bobot 0.65 untuk OutOfContext?"
    A: OutOfContext adalah sinyal paling langsung dari halusinasi.
       Contradiction (0.20) dan Uncertainty (0.15) adalah sinyal sekunder.

    Q: "Bagaimana threshold RENDAH/SEDANG/TINGGI ditetapkan?"
    A: Diturunkan dari kategori Faithfulness:
       Jika Faithfulness ≥ 0.65 (Sangat Baik), maka:
       OutOfContext ≤ 0.35 → Risk ≤ 0.65 × 0.35 = 0.228
       Threshold < 0.32 mencakup seluruh kasus Faithfulness Sangat Baik.
       Threshold 0.32-0.55 = Faithfulness Baik/Cukup. > 0.55 = Buruk.

    Formula (Pers. 3.17):
       HallucinationRisk = 0.65 × OutOfContext
                         + 0.20 × Contradiction
                         + 0.15 × Uncertainty
    ═══════════════════════════════════════════════════════════════
    Hallucination risk ∈ [0, 1].
    Semakin rendah proporsi klaim yang didukung chunk GT,
    semakin tinggi risiko halusinasi.
    """
    faith = evaluate_faithfulness(answer, retrieved_chunks, get_emb_fn)

    # OutOfContext = 1 - Faithfulness (komplemen matematis dari Persamaan 3.14)
    out_of_context = 1.0 - faith["faithfulness_score"]

    # Komponen Contradiction: terinspirasi dari deteksi kontradiksi ARES
    # (Saad-Falcon et al. 2024) — disederhanakan ke pencocokan pola regex
    contradiction = 0.0
    answer_lower = answer.lower()
    for pat in _CONTRADICTION_PATTERNS:
        if re.search(pat, answer_lower):
            contradiction = min(contradiction + 0.2, 1.0)

    # Komponen Uncertainty: sinyal keraguan leksikal dari respons LLM
    uncertainty = 0.2 if any(p in answer_lower for p in _UNCERTAINTY_PHRASES) else 0.0

    # Formula Hallucination Risk (Persamaan 3.17)
    risk = round(
        0.65 * out_of_context +
        0.20 * contradiction  +
        0.15 * uncertainty,
        4,
    )
    # Threshold diturunkan dari kategori Faithfulness (lihat skripsi Bab 3.4.7)
    risk_label = "TINGGI" if risk >= 0.55 else "SEDANG" if risk >= 0.32 else "RENDAH"

    return {
        "hallucination_risk":  risk,
        "risk_label":          risk_label,
        "out_of_context":      round(out_of_context, 4),
        "contradiction_score": round(contradiction, 4),
        "uncertainty_flag":    bool(uncertainty),
        "faithfulness":        faith["faithfulness_score"],
        "faithfulness_method": faith.get("method", ""),
    }