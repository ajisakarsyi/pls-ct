"""
app/services/rag.py
────────────────────
RAG (Retrieval-Augmented Generation) service.

Responsibilities:
  - Read .txt / .md material files from disk.
  - Chunk, embed, and index them with FAISS (or NumPy fallback).
  - Retrieve the top-K most relevant chunks for a given query.

Two index tiers:
  1. Per-cognitive-type index  — files whose stem matches a valid code
                                 (e.g. "3TGI.txt").
  2. Global fallback index     — all other topic files
                                 (e.g. "algoritma_sorting.txt").
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from app.core import verbose as V
from app.core.cognitive import VALID_COGNITIVE_TYPES
from app.core.config import get_settings
from app.core.rag_guard import assert_rag_allowed
from app.services.llm import get_embedding
from topics import topic_of

logger = logging.getLogger(__name__)
_settings = get_settings()

# Optional FAISS
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore
    logger.warning("faiss not installed — falling back to NumPy cosine search.")

# ── In-memory stores ───────────────────────────────────────────────────────
_cognitive_indices: Dict[str, Dict[str, Any]] = {}
_cognitive_loaded: Dict[str, bool] = {}

_global_chunks: List[Dict] = []
_global_faiss: Optional[Any] = None
_global_loaded = False


# ── Low-level helpers ──────────────────────────────────────────────────────

def _build_faiss_index(chunks: List[Dict]) -> Optional[Any]:
    if faiss is None or not chunks:
        return None
    # Filter chunk yang embedding-nya None atau dimensinya tidak konsisten.
    # Ini mencegah ValueError: all input arrays must have the same shape
    # yang terjadi ketika sebagian chunk gagal di-embed (misal 429 fallback
    # menghasilkan dimensi berbeda dari embedding Ollama).
    valid = [c for c in chunks
             if c.get("embedding") is not None
             and hasattr(c["embedding"], "shape")
             and c["embedding"].ndim == 1]
    if not valid:
        return None
    # Tentukan dimensi yang paling umum (modus) dan buang yang berbeda
    from collections import Counter
    dim_counts = Counter(c["embedding"].shape[0] for c in valid)
    expected_dim = dim_counts.most_common(1)[0][0]
    valid = [c for c in valid if c["embedding"].shape[0] == expected_dim]
    if len(valid) < len(chunks):
        logger.warning(
            "_build_faiss_index: membuang %d/%d chunk "
            "(dimensi tidak konsisten, expected=%d).",
            len(chunks) - len(valid), len(chunks), expected_dim,
        )
    if not valid:
        return None
    mat = np.stack([c["embedding"] for c in valid]).astype("float32")
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return idx


def _embed_file(path: str, fname: str) -> List[Dict]:
    """
    ═══════════════════════════════════════════════════════════════
    PROSES CHUNKING DAN EMBEDDING DOKUMEN MATERI
    File: app/services/rag.py → _embed_file()
    ═══════════════════════════════════════════════════════════════

    Q: "Bagaimana dokumen materi diproses sebelum bisa dicari?"
    A: Tiga tahap:
       1. Baca file .txt dari folder materials/
       2. Potong (chunk) menjadi potongan ~1200 karakter
          → chunk_size diatur di config (rag_embed_chunk_size)
       3. Setiap chunk di-embed menjadi vektor 768 dimensi
          menggunakan nomic-embed-text via Ollama, lalu dinormalisasi

    Q: "Mengapa chunk size 1200 karakter?"
    A: Keseimbangan antara:
       - Terlalu kecil (<600 kar) → kehilangan konteks antar kalimat
       - Terlalu besar (>1500 kar) → melampaui kapasitas evaluator
       Konsisten dengan trade-off chunk size dalam Brown et al. (2025)

    Q: "Mengapa setiap embedding dinormalisasi (emb /= norm)?"
    A: Agar dot product antar vektor = cosine similarity
       (bukan inner product yang dipengaruhi panjang vektor).
       FAISS IndexFlatIP menghitung inner product, jadi normalisasi
       membuatnya setara dengan cosine similarity.
    ═══════════════════════════════════════════════════════════════
    Read a file, chunk it, embed each chunk, and return a list of dicts.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return []

    if not text:
        return []

    out: List[Dict] = []
    # chunk_size dari config — default 1200 karakter (rag_embed_chunk_size)
    chunk_size = _settings.rag_embed_chunk_size
    n_expected = (len(text) + chunk_size - 1) // chunk_size
    V.step(f"Chunking {fname}: {len(text)} char → {n_expected} chunk "
           f"@ {chunk_size} char | topik: {topic_of(fname, _settings.materials_dir)}")
    for i, chunk in enumerate(
        text[j: j + chunk_size] for j in range(0, len(text), chunk_size)
    ):
        try:
            emb = np.array(get_embedding(chunk), dtype="float32")
            norm = np.linalg.norm(emb)
            if norm:
                emb /= norm  # L2 normalization → dot product = cosine similarity
            out.append({"embedding": emb, "text": chunk, "source": fname, "chunk_id": i})
        except Exception as exc:
            logger.error("Embedding error %s#%d: %s", fname, i, exc)
            # Skip chunk ini, lanjut ke chunk berikutnya.
            # Satu chunk gagal tidak boleh membatalkan seluruh file.
            continue
    V.step(f"Embedding {fname}: {len(out)}/{n_expected} chunk berhasil "
           f"({_settings.ollama_embed_model}, dinormalisasi L2)")
    return out


def _numpy_search(
    q_emb: np.ndarray, chunks: List[Dict], top_k: int
) -> List[Dict]:
    if not chunks:
        return []
    scores = [float(np.dot(q_emb, c["embedding"])) for c in chunks]
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"],
         "chunk_id": chunks[i].get("chunk_id"), "score": scores[i]}
        for i in indices
        if scores[i] > 0
    ]


def _faiss_search(
    q_emb: np.ndarray, chunks: List[Dict], index: Any, top_k: int
) -> List[Dict]:
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    return [
        {"text": chunks[int(i)]["text"], "source": chunks[int(i)]["source"],
         "chunk_id": chunks[int(i)].get("chunk_id"), "score": float(s)}
        for i, s in zip(I[0], D[0])
        if i >= 0 and s > 0
    ]


def _search(
    q_emb: np.ndarray, chunks: List[Dict], index: Optional[Any], top_k: int
) -> List[Dict]:
    if not chunks:
        return []
    if index is not None:
        return _faiss_search(q_emb, chunks, index, top_k)
    return _numpy_search(q_emb, chunks, top_k)


# ── Public loaders ─────────────────────────────────────────────────────────

def load_cognitive_materials(code: str) -> None:
    """Index all material files that belong to cognitive type *code*."""
    code = code.upper()
    if _cognitive_loaded.get(code):
        return

    mat_dir = _settings.materials_dir
    if not os.path.isdir(mat_dir):
        _cognitive_loaded[code] = True
        return

    logger.info("Indexing materials for cognitive type: %s", code)
    chunks: List[Dict] = []
    cog_set = set(VALID_COGNITIVE_TYPES)

    for root, _, files in os.walk(mat_dir):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md")):
                continue
            stem = os.path.splitext(fname)[0].upper()
            if stem == code or stem.startswith(code + "_"):
                chunks.extend(_embed_file(os.path.join(root, fname), fname))

    _cognitive_indices[code] = {"chunks": chunks, "faiss": _build_faiss_index(chunks)}
    _cognitive_loaded[code] = True
    logger.info("Indexed %d chunks for %s.", len(chunks), code)


def load_global_materials() -> None:
    """Index all shared (non-cognitive-type-specific) material files."""
    global _global_chunks, _global_faiss, _global_loaded
    if _global_loaded:
        return

    mat_dir = _settings.materials_dir
    if not os.path.isdir(mat_dir):
        _global_loaded = True
        return

    logger.info("Building global RAG fallback index…")
    chunks: List[Dict] = []
    cog_set = set(VALID_COGNITIVE_TYPES)

    for root, _, files in os.walk(mat_dir):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md")):
                continue
            stem = os.path.splitext(fname)[0].upper()
            if any(stem == ct or stem.startswith(ct + "_") for ct in cog_set):
                continue
            chunks.extend(_embed_file(os.path.join(root, fname), fname))

    _global_chunks = chunks
    _global_faiss = _build_faiss_index(chunks)
    _global_loaded = True
    logger.info("Global index ready: %d chunks.", len(chunks))


# ── Public retriever ───────────────────────────────────────────────────────

def retrieve(query: str, cognitive_code: str, k: int = None) -> List[Dict]:
    """
    ═══════════════════════════════════════════════════════════════
    FUNGSI INTI RAG — retrieve()
    File: app/services/rag.py
    ═══════════════════════════════════════════════════════════════

    PERTANYAAN SIDANG yang mungkin ditanyakan tentang fungsi ini:

    Q: "Bagaimana RAG mengambil dokumen yang relevan?"
    A: Fungsi ini menghitung cosine similarity antara embedding query
       mahasiswa dan embedding semua chunk dokumen materi GT, lalu
       mengambil k=6 chunk dengan skor tertinggi (TopK retrieval).

    Q: "Apa itu dua tingkat indeks (two-tier index)?"
    A: Ada dua indeks FAISS:
       1. Per-cognitive index  → file yang namanya cocok dengan kode
          kognitif mahasiswa (misal: 3TAR.txt) → materi terpersonalisasi
       2. Global fallback index → semua file materi umum (GT_CT*.txt)
          → digunakan jika slot TopK belum penuh dari indeks per-kognitif

    Q: "Mengapa menggunakan FAISS?"
    A: FAISS (Johnson et al. 2021) menyediakan approximate nearest
       neighbor search yang jauh lebih cepat dari brute-force NumPy.
       Ada fallback ke NumPy jika FAISS tidak tersedia (lihat _search).

    Q: "Bagaimana embedding query dinormalisasi?"
    A: q_emb dibagi normanya (L2 normalization) sehingga dot product
       antar vektor ternormalisasi = cosine similarity.
       Ini konsisten dengan cara chunk di-embed saat indexing (_embed_file).

    Q: "Mengapa ada deduplication di akhir?"
    A: Satu chunk bisa muncul dari dua indeks (per-kognitif dan global).
       Dedup via set of (source, text[:80]) mencegah chunk yang sama
       muncul dua kali di konteks prompt LLM.
    ═══════════════════════════════════════════════════════════════
    Return up to *k* most relevant chunks for *query* given the student's
    cognitive type.  Falls back to the global index if the per-type index
    doesn't fill the quota.

    REVISI PASCA-SIDANG:
      • assert_rag_allowed() — bila permintaan sedang berjalan dalam
        Kondisi B (NoRAGGuard aktif), panggilan ini langsung melempar
        RAGBlockedError dan tercatat di no_rag_proof (bukti item 2).
      • Setiap chunk hasil kini membawa "score" (cosine, 4 desimal) dan
        "topic" (resolusi header file materi) — dipakai metrik live,
        panel transparansi UI, dan tabel verbose terminal (item 3).
    """
    assert_rag_allowed("retrieval")
    if k is None:
        # k = rag_top_k dari config (default 6)
        # K=6 dipilih berdasarkan keseimbangan cakupan konteks vs panjang prompt
        k = _settings.rag_top_k

    code = cognitive_code.upper()

    # Muat indeks per-kognitif (misal: 3TAR.txt → indeks khusus 3TAR)
    load_cognitive_materials(code)
    # Muat indeks global (semua file GT_CT*.txt)
    load_global_materials()

    # ── LANGKAH 1: Embed query mahasiswa ──────────────────────────────
    # Menggunakan model nomic-embed-text via Ollama (sama dengan saat indexing)
    q_emb = np.array(get_embedding(query), dtype="float32")
    norm = np.linalg.norm(q_emb)
    if norm:
        # L2 normalization → dot product = cosine similarity
        q_emb /= norm

    # ── LANGKAH 2: Cari di indeks per-kognitif dulu ──────────────────
    cog = _cognitive_indices.get(code, {})
    hits = _search(q_emb, cog.get("chunks", []), cog.get("faiss"), k)

    # ── LANGKAH 3: Fallback ke indeks global jika slot belum penuh ───
    if len(hits) < k:
        hits += _search(q_emb, _global_chunks, _global_faiss, k - len(hits))

    # ── LANGKAH 4: Deduplication ──────────────────────────────────────
    seen: set = set()
    out: List[Dict] = []
    for r in hits:
        key = (r["source"], r["text"][:80])
        if key not in seen:
            seen.add(key)
            out.append(r)
    out = out[:k]

    # ── LANGKAH 5: Perkaya hasil (revisi pasca-sidang) ────────────────
    for r in out:
        r["score"] = round(float(r.get("score", 0.0)), 4)
        r["topic"] = topic_of(r["source"], _settings.materials_dir)

    if V.enabled():
        V.section(f"RETRIEVAL Top-{k} untuk query ({len(query)} char) "
                  f"| kognitif {code}")
        V.chunk_table(out, _settings.theta_retrieval, _settings.theta_coverage)

    return out


def chunks_to_context(chunks: List[Dict], max_chars: int = None) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    if max_chars is None:
        max_chars = _settings.rag_chunk_max_chars
    if not chunks:
        return "Tidak ada konteks materi relevan."
    return "\n\n".join(
        f"[{c['source']}]\n{c['text'][:max_chars]}" for c in chunks
    )