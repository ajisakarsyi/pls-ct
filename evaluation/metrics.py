"""
evaluation/metrics.py
──────────────────────
Fungsi metrik murni (stateless) untuk evaluasi RAG — selaras dengan subbab
"Metrik Evaluasi dan Formula Matematis" (Bab 3 skripsi).

AMBANG BATAS RESMI PENELITIAN (revisi pasca-sidang):
    THETA_RETRIEVAL = 0,25   → θ  pada Precision@K (Pers. 2) & Recall@K (Pers. 4)
    THETA_COVERAGE  = 0,35   → θc pada Coverage (Pers. 8)

CATATAN PERBAIKAN BUG (revisi item 6):
    Pada versi sidang, ambang kedua metrik TERTUKAR/terbalik:
        precision_at_k memakai default 0,30 (seharusnya θ  = 0,25) dan
        coverage_score memakai default 0,25 — bahkan runner memanggilnya
        dengan 0,20 — (seharusnya θc = 0,35, LEBIH TINGGI dari θ).
    Akibatnya Coverage selalu 1,00 (ambangnya lebih rendah dari ambang
    retrieval sehingga semua chunk otomatis lolos) — bertentangan dengan
    desain Pers. 8 yang menempatkan θc 0,10 poin DI ATAS θ. Konstanta di
    bawah kini menjadi satu-satunya sumber ambang bagi runner, demo, dan
    laporan Excel.

Setiap fungsi memiliki varian *_detail() yang mengembalikan langkah
perhitungan (substitusi angka + string rumus) untuk dicetak ke terminal dan
direkam ke Excel — menjawab kebutuhan transparansi penuh (revisi item 3 & 6).
"""

from typing import Dict, List

import numpy as np

# ── Ambang resmi (satu-satunya sumber; jangan hardcode di tempat lain) ─────
THETA_RETRIEVAL: float = 0.25   # θ  — Persamaan 2 & 4 (Precision@K, Recall@K)
THETA_COVERAGE:  float = 0.35   # θc — Persamaan 8 (Coverage) = θ + 0,10


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Persamaan 1:  sim(q, d) = (q · d) / (||q|| · ||d||)

    Mengukur kosinus sudut antara dua vektor embedding; robust terhadap
    perbedaan panjang teks (Manning et al. 2008).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ══════════════════════════════════════════════════════════════════════════
# PRECISION@K — Persamaan 2
# ══════════════════════════════════════════════════════════════════════════

def precision_at_k(scores: List[float], k: int,
                   threshold: float = THETA_RETRIEVAL) -> float:
    """
    Persamaan 2:  Precision@K = |{d ∈ TopK : sim(q,d) ≥ θ}| / K,  θ = 0,25.

    `scores` adalah skor kemiripan kosinus CHUNK NYATA hasil retrieval
    terhadap query (sesuai contoh Pers. 3 skripsi yang menghitung dari
    enam skor chunk).
    """
    if k == 0:
        return 0.0
    return sum(1 for s in scores[:k] if s >= threshold) / k


def precision_at_k_detail(scores: List[float], k: int,
                          threshold: float = THETA_RETRIEVAL) -> Dict:
    """Versi transparan: kembalikan langkah perhitungan Precision@K."""
    used = [round(float(s), 4) for s in scores[:k]]
    passed = sum(1 for s in used if s >= threshold)
    value = round(passed / k, 4) if k else 0.0
    return {
        "value": value,
        "formula": "Precision@K = |{d ∈ TopK : sim(q,d) ≥ θ}| / K",
        "threshold": threshold,
        "scores_used": used,
        "n_passed": passed,
        "k": k,
        "substitution": f"{passed} / {k}",
        "calc_str": (f"P@{k} = |{{skor ≥ θ={threshold:.2f}}}| / K "
                     f"= {passed}/{k} = {value:.4f}"),
    }


# ══════════════════════════════════════════════════════════════════════════
# RECALL@K — Persamaan 4
# ══════════════════════════════════════════════════════════════════════════

def recall_at_k(scores: List[float], total_relevant: int, k: int,
                threshold: float = THETA_RETRIEVAL) -> float:
    """
    Persamaan 4:  Recall@K = |{ditemukan : sim ≥ θ}| / |R|,  θ = 0,25.

    Sesuai Bab 3, relevansi Recall dinilai dari kemiripan semantik antara
    embedding query dan embedding KATA KUNCI acuan per kasus uji (`scores`
    = skor kata kunci, |R| = jumlah kata kunci relevan).
    """
    if total_relevant == 0:
        return 1.0
    return sum(1 for s in scores[:k] if s >= threshold) / total_relevant


def recall_at_k_detail(scores: List[float], total_relevant: int, k: int,
                       threshold: float = THETA_RETRIEVAL) -> Dict:
    """Versi transparan: kembalikan langkah perhitungan Recall@K."""
    used = [round(float(s), 4) for s in scores[:k]]
    found = sum(1 for s in used if s >= threshold)
    value = round(found / total_relevant, 4) if total_relevant else 1.0
    return {
        "value": value,
        "formula": "Recall@K = |{kata kunci ditemukan : sim ≥ θ}| / |R|",
        "threshold": threshold,
        "scores_used": used,
        "n_found": found,
        "total_relevant": total_relevant,
        "substitution": f"{found} / {total_relevant}",
        "calc_str": (f"R@{k} = |{{kw ≥ θ={threshold:.2f}}}| / |R| "
                     f"= {found}/{total_relevant} = {value:.4f}"),
    }


# ══════════════════════════════════════════════════════════════════════════
# MEAN SIMILARITY — Persamaan 6
# ══════════════════════════════════════════════════════════════════════════

def mean_similarity(scores: List[float]) -> float:
    """Persamaan 6:  MeanSim = (1/K) × Σ sim(q, dᵢ) — atas chunk nyata."""
    return sum(scores) / len(scores) if scores else 0.0


def mean_similarity_detail(scores: List[float]) -> Dict:
    """Versi transparan: kembalikan langkah perhitungan Mean Similarity."""
    used = [round(float(s), 4) for s in scores]
    value = round(sum(used) / len(used), 4) if used else 0.0
    return {
        "value": value,
        "formula": "MeanSim = (1/K) × Σ sim(q, dᵢ)",
        "scores_used": used,
        "substitution": f"({' + '.join(f'{s:.4f}' for s in used)}) / {len(used)}"
                        if used else "0 (tidak ada chunk)",
        "calc_str": (f"MeanSim = Σskor/K = {sum(used):.4f}/{len(used)} "
                     f"= {value:.4f}") if used else "MeanSim = 0.0",
    }


# ══════════════════════════════════════════════════════════════════════════
# COVERAGE — Persamaan 8
# ══════════════════════════════════════════════════════════════════════════

def coverage_score(scores: List[float],
                   threshold: float = THETA_COVERAGE) -> float:
    """
    Persamaan 8:  Coverage = |{dᵢ ∈ TopK : sim(q,dᵢ) ≥ θc}| / K,  θc = 0,35.

    θc sengaja 0,10 poin DI ATAS θ retrieval (0,25): Coverage menguji
    apakah chunk yang sudah masuk prompt sungguh substantif, bukan sekadar
    lolos ambang minimal (lihat pembahasan Pers. 8 di skripsi).
    """
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0


def coverage_detail(scores: List[float],
                    threshold: float = THETA_COVERAGE) -> Dict:
    """Versi transparan: kembalikan langkah perhitungan Coverage."""
    used = [round(float(s), 4) for s in scores]
    passed = sum(1 for s in used if s >= threshold)
    value = round(passed / len(used), 4) if used else 0.0
    return {
        "value": value,
        "formula": "Coverage = |{dᵢ ∈ TopK : sim(q,dᵢ) ≥ θc}| / K",
        "threshold": threshold,
        "scores_used": used,
        "n_passed": passed,
        "k": len(used),
        "substitution": f"{passed} / {len(used)}" if used else "0 / 0",
        "calc_str": (f"Coverage = |{{skor ≥ θc={threshold:.2f}}}| / K "
                     f"= {passed}/{len(used)} = {value:.4f}") if used
                    else "Coverage = 0.0 (tidak ada chunk)",
    }


# ══════════════════════════════════════════════════════════════════════════
# SOURCE DIVERSITY — Persamaan 10
# ══════════════════════════════════════════════════════════════════════════

def source_diversity(sources: List[str]) -> float:
    """Persamaan 10:  Diversity = |sumber unik dalam TopK| / K."""
    return len(set(sources)) / len(sources) if sources else 0.0


def source_diversity_detail(sources: List[str]) -> Dict:
    """
    Versi transparan: selain nilai, laporkan NAMA-NAMA FILE sumber unik
    (revisi item 6: "source diversity dapat memperlihatkan nama file").
    """
    uniq = sorted(set(sources))
    value = round(len(uniq) / len(sources), 4) if sources else 0.0
    return {
        "value": value,
        "formula": "Diversity = |sumber unik dalam TopK| / K",
        "unique_sources": uniq,
        "n_unique": len(uniq),
        "k": len(sources),
        "substitution": f"{len(uniq)} / {len(sources)}" if sources else "0 / 0",
        "calc_str": (f"Diversity = unik/K = {len(uniq)}/{len(sources)} "
                     f"= {value:.4f}  (file unik: {', '.join(uniq)})")
                    if sources else "Diversity = 0.0 (tidak ada sumber)",
    }


# ══════════════════════════════════════════════════════════════════════════
# CHUNK RELEVANCE (sinyal kontinu pelengkap — bukan metrik skripsi utama)
# ══════════════════════════════════════════════════════════════════════════

def chunk_relevance_score(chunk_scores: List[float]) -> float:
    """
    Rata-rata berbobot skor chunk (chunk teratas berbobot 2×) — sinyal
    kontinu pelengkap di log; tidak termasuk delapan metrik utama skripsi.
    """
    if not chunk_scores:
        return 0.0
    if len(chunk_scores) == 1:
        return float(chunk_scores[0])
    weighted = chunk_scores[0] * 2 + sum(chunk_scores[1:])
    return float(weighted / (len(chunk_scores) + 1))
