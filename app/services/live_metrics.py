"""
app/services/live_metrics.py
─────────────────────────────
Metrik evaluasi LIVE untuk demo penggunaan biasa — revisi pasca-sidang item 3
("terminal juga menampilkan metrik saat sistem dipakai normal, bukan hanya
saat evaluasi batch").

Untuk SETIAP jawaban chatbot:

  KONDISI A (RAG):
    • Precision@K, Coverage, MeanSim, Source Diversity — dihitung dari skor
      cosine chunk yang BENAR-BENAR dipakai menjawab (Pers. 2, 6, 8, 10),
      lengkap dengan rumus + substitusi angka.
    • Scan leksikal Uncertainty & Contradiction pada jawaban (komponen
      Pers. 18) beserta lokasi frasa.
    • Recall@K dan Faithfulness TIDAK dihitung live — keduanya membutuhkan
      acuan kasus uji (himpunan kata kunci relevan R untuk Recall; referensi
      GT + LLM-as-Judge untuk Faithfulness). Alasannya dilaporkan eksplisit
      alih-alih menampilkan angka yang tidak valid. Bila DEMO_FULL_METRICS=1,
      entailment LLM-as-Judge ikut dihitung live (lambat — m panggilan Ollama
      ekstra per jawaban).

  KONDISI B (LLM murni):
    • SEMUA metrik retrieval dilaporkan N/A dengan alasan — tidak ada
      retrieval sama sekali. Ini justru bagian dari BUKTI Kondisi B murni
      (item 2), bersama no_rag_proof dari NoRAGGuard.
    • Scan leksikal Uncertainty & Contradiction tetap dihitung (murni
      analisis teks jawaban, tanpa referensi).

Seluruh hasil dicetak ke terminal (LOGICT_VERBOSE) dan dikirim ke UI
(field live_metrics pada ChatResponse) untuk panel "Detail Transparansi".
"""

import logging
from typing import Dict, List, Optional

from app.core import verbose as V
from app.core.config import get_settings
from evaluation.lexicon import contradiction_scan, uncertainty_scan
from evaluation.metrics import (
    THETA_COVERAGE,
    THETA_RETRIEVAL,
    coverage_detail,
    mean_similarity_detail,
    precision_at_k_detail,
    source_diversity_detail,
)

logger = logging.getLogger(__name__)
_settings = get_settings()


def _lexical_block(reply: str) -> Dict:
    """Scan Uncertainty & Contradiction (Pers. 18) pada teks jawaban."""
    unc = uncertainty_scan(reply)
    con = contradiction_scan(reply)
    return {
        "uncertainty": {
            "value":   unc["value"],
            "flag":    unc["flag"],
            "matches": unc["matches"],
            "keterangan": ("0,20 karena ada frasa ketidakpastian (biner, "
                           "Pers. 18)" if unc["flag"] else
                           "0,00 — tidak ada frasa ketidakpastian terdeteksi"),
        },
        "contradiction": {
            "value":              con["value"],
            "n_sentences":        con["n_sentences"],
            "n_flagged_sentences": con["n_flagged_sentences"],
            "matches":            con["matches"],
            "keterangan": (f"{con['n_flagged_sentences']}/{con['n_sentences']} "
                           f"kalimat mengandung frasa kontradiksi kuat "
                           f"(proporsi, Pers. 18)"),
        },
    }


def _entailment_live(reply: str, chunks: List[Dict]) -> Optional[Dict]:
    """Faithfulness live opsional (DEMO_FULL_METRICS=1) — LAMBAT."""
    if not _settings.demo_full_metrics:
        return None
    try:
        from app.services.llm import get_embedding
        from evaluation.faithfulness import evaluate_faithfulness
        faith = evaluate_faithfulness(
            reply, [c.get("text", "") for c in chunks],
            lambda t: get_embedding(t), use_entailment=True,
        )
        return {
            "faithfulness_score": faith.get("faithfulness_score"),
            "entailment_score":   faith.get("entailment_score"),
            "keyword_overlap":    faith.get("keyword_overlap"),
            "calc_str":           faith.get("calc_str"),
            "method":             faith.get("method"),
            "catatan": "Dihitung live karena DEMO_FULL_METRICS=1 — "
                       "menambah beberapa panggilan Ollama per jawaban.",
        }
    except Exception as exc:                              # pragma: no cover
        logger.warning("Faithfulness live gagal: %s", exc)
        return {"error": str(exc)}


def compute_live_metrics_a(query: str, chunks: List[Dict],
                           reply: str) -> Dict:
    """
    Metrik live Kondisi A dari skor chunk yang dipakai menjawab.

    Parameters
    ----------
    query  : pertanyaan pengguna (untuk konteks log).
    chunks : hasil rag.retrieve() — tiap dict membawa score/source/topic.
    reply  : jawaban LLM final.

    Returns
    -------
    dict — precision/coverage/meansim/diversity (dengan *_detail berisi
    rumus + substitusi), blok not_computed beralasan, scan leksikal, dan
    faithfulness_live opsional.
    """
    scores  = [float(c.get("score", 0.0)) for c in chunks]
    sources = [c.get("source", "?") for c in chunks]
    k       = _settings.rag_top_k

    p_det   = precision_at_k_detail(scores, k, THETA_RETRIEVAL)
    cov_det = coverage_detail(scores, THETA_COVERAGE)
    ms_det  = mean_similarity_detail(scores)
    div_det = source_diversity_detail(sources)

    out = {
        "mode": "A",
        "k": k,
        "theta": THETA_RETRIEVAL,
        "theta_c": THETA_COVERAGE,
        "precision_at_k":   p_det["value"],
        "coverage":         cov_det["value"],
        "mean_similarity":  ms_det["value"],
        "source_diversity": div_det["value"],
        "precision_detail":        p_det,
        "coverage_detail":         cov_det,
        "mean_sim_detail":         ms_det,
        "source_diversity_detail": div_det,
        "not_computed": {
            "recall_at_k": ("Butuh himpunan kata kunci relevan R dari kasus "
                            "uji (Pers. 4) — tidak tersedia pada pertanyaan "
                            "bebas. Lihat evaluasi batch (run_evaluation)."),
            "faithfulness": None if _settings.demo_full_metrics else (
                "Butuh LLM-as-Judge terhadap referensi GT (Pers. 12-15) — "
                "berat untuk tiap jawaban live. Set DEMO_FULL_METRICS=1 "
                "untuk mengaktifkan."),
            "answer_accuracy": ("Butuh penilaian benar/salah terhadap acuan "
                                "kasus uji (Pers. 20) — hanya pada evaluasi "
                                "batch."),
        },
        **_lexical_block(reply),
    }
    faith_live = _entailment_live(reply, chunks)
    if faith_live is not None:
        out["faithfulness_live"] = faith_live
        out["not_computed"].pop("faithfulness", None)
    out["not_computed"] = {kk: vv for kk, vv in out["not_computed"].items()
                           if vv is not None}

    if V.enabled():
        V.section("METRIK LIVE — KONDISI A (dari skor chunk jawaban ini)")
        for d in (p_det, cov_det, ms_det, div_det):
            V.calc(d["calc_str"])
        unc, con = out["uncertainty"], out["contradiction"]
        V.kv("Uncertainty",   f"{unc['value']} — {unc['keterangan']}")
        V.kv("Contradiction", f"{con['value']} — {con['keterangan']}")
        for alasan_k, alasan_v in out["not_computed"].items():
            V.note(f"{alasan_k}: tidak dihitung live — {alasan_v}")
        if "faithfulness_live" in out and out["faithfulness_live"].get("calc_str"):
            V.calc(out["faithfulness_live"]["calc_str"])
    return out


def compute_live_metrics_b(reply: str) -> Dict:
    """
    Metrik live Kondisi B: seluruh metrik retrieval N/A karena TIDAK ADA
    retrieval — pelaporan eksplisit ini bagian dari bukti kemurnian B.
    Scan leksikal tetap dihitung (analisis teks jawaban semata).
    """
    na = ("N/A — Kondisi B tidak melakukan retrieval sama sekali "
          "(tidak ada chunk, tidak ada skor). Lihat no_rag_proof.")
    out = {
        "mode": "B",
        "precision_at_k":   None,
        "coverage":         None,
        "mean_similarity":  None,
        "source_diversity": None,
        "not_computed": {
            "precision_at_k":   na,
            "coverage":         na,
            "mean_similarity":  na,
            "source_diversity": na,
            "recall_at_k":      na,
            "faithfulness": ("Butuh referensi GT hasil retrieval — pada "
                             "evaluasi batch, jawaban B dinilai terhadap "
                             "chunk GT Kondisi A (Bab 3.4.6)."),
            "answer_accuracy": ("Hanya pada evaluasi batch (Pers. 20)."),
        },
        **_lexical_block(reply),
    }
    if V.enabled():
        V.section("METRIK LIVE — KONDISI B (LLM murni)")
        V.note("Semua metrik retrieval N/A — tidak ada retrieval "
               "(bukti Kondisi B; lihat no_rag_proof).")
        unc, con = out["uncertainty"], out["contradiction"]
        V.kv("Uncertainty",   f"{unc['value']} — {unc['keterangan']}")
        V.kv("Contradiction", f"{con['value']} — {con['keterangan']}")
    return out
