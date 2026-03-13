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

from app.core.cognitive import VALID_COGNITIVE_TYPES
from app.core.config import get_settings
from app.services.llm import get_embedding

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
    mat = np.stack([c["embedding"] for c in chunks]).astype("float32")
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return idx


def _embed_file(path: str, fname: str) -> List[Dict]:
    """Read a file, chunk it, embed each chunk, and return a list of dicts."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return []

    if not text:
        return []

    out: List[Dict] = []
    chunk_size = _settings.rag_embed_chunk_size
    for i, chunk in enumerate(
        text[j: j + chunk_size] for j in range(0, len(text), chunk_size)
    ):
        try:
            emb = np.array(get_embedding(chunk), dtype="float32")
            norm = np.linalg.norm(emb)
            if norm:
                emb /= norm
            out.append({"embedding": emb, "text": chunk, "source": fname, "chunk_id": i})
        except Exception as exc:
            logger.error("Embedding error %s#%d: %s", fname, i, exc)
            break
    return out


def _numpy_search(
    q_emb: np.ndarray, chunks: List[Dict], top_k: int
) -> List[Dict]:
    if not chunks:
        return []
    scores = [float(np.dot(q_emb, c["embedding"])) for c in chunks]
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": scores[i]}
        for i in indices
        if scores[i] > 0
    ]


def _faiss_search(
    q_emb: np.ndarray, chunks: List[Dict], index: Any, top_k: int
) -> List[Dict]:
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    return [
        {"text": chunks[int(i)]["text"], "source": chunks[int(i)]["source"], "score": float(s)}
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
    Return up to *k* most relevant chunks for *query* given the student's
    cognitive type.  Falls back to the global index if the per-type index
    doesn't fill the quota.
    """
    if k is None:
        k = _settings.rag_top_k

    code = cognitive_code.upper()
    load_cognitive_materials(code)
    load_global_materials()

    q_emb = np.array(get_embedding(query), dtype="float32")
    norm = np.linalg.norm(q_emb)
    if norm:
        q_emb /= norm

    cog = _cognitive_indices.get(code, {})
    hits = _search(q_emb, cog.get("chunks", []), cog.get("faiss"), k)

    if len(hits) < k:
        hits += _search(q_emb, _global_chunks, _global_faiss, k - len(hits))

    # Deduplicate
    seen: set = set()
    out: List[Dict] = []
    for r in hits:
        key = (r["source"], r["text"][:80])
        if key not in seen:
            seen.add(key)
            out.append(r)

    return out[:k]


def chunks_to_context(chunks: List[Dict], max_chars: int = None) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    if max_chars is None:
        max_chars = _settings.rag_chunk_max_chars
    if not chunks:
        return "Tidak ada konteks materi relevan."
    return "\n\n".join(
        f"[{c['source']}]\n{c['text'][:max_chars]}" for c in chunks
    )
