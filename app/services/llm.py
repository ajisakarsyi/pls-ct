"""
app/services/llm.py
────────────────────
LLM wrapper with ChatAnywhere-first, Ollama-fallback logic.

Chat routing (set once at startup via probe_and_set_chat_provider):
  CHAT_PROVIDER=auto   → probe ChatAnywhere; use it if key valid, else fall back to Ollama
  CHAT_PROVIDER=openai → always ChatAnywhere/OpenAI
  CHAT_PROVIDER=ollama → always local Ollama

Embeddings: ALWAYS Ollama-first.
  get_embedding() tries Ollama first, auto-falls back to OpenAI/ChatAnywhere only
  if Ollama is unreachable (e.g. ollama serve not running).
  This means RAG indexing never burns ChatAnywhere quota unless Ollama is down.
"""

import logging
import threading
import time
from typing import List

import numpy as np
from openai import OpenAI

from app.core import verbose as V
from app.core.config import get_settings
from app.core.prompts import SYSTEM_PROMPT
from app.core.rag_guard import assert_rag_allowed
from app.utils.latex import normalize_latex

logger    = logging.getLogger(__name__)
_settings = get_settings()

_provider_lock         = threading.Lock()
_active_chat_provider: str = "openai"   # overwritten by probe_and_set_chat_provider()

# ── REVISI PASCA-SIDANG (KEAMANAN) ────────────────────────────────────────
# API key TIDAK lagi tertanam di kode. Key ChatAnywhere lama yang pernah
# hardcoded di sini & di config.py sudah dihapus dan harus dianggap BOCOR —
# segera revoke. Client OpenAI hanya dibuat bila env OPENAI_API_KEY di-set.
_openai_client = None
if _settings.openai_api_key:
    _openai_client = OpenAI(
        api_key=_settings.openai_api_key,
        base_url=_settings.openai_api_base,
    )


def _require_openai_client() -> OpenAI:
    if _openai_client is None:
        raise RuntimeError(
            "Provider OpenAI diminta tetapi env OPENAI_API_KEY tidak di-set. "
            "Set key lewat environment, atau gunakan CHAT_PROVIDER=ollama "
            "(default sesuai skripsi)."
        )
    return _openai_client

_QUOTA_PATTERNS = (
    "429", "quota", "rate limit", "insufficient_quota",
    "too many requests", "暂时禁止", "免费api限制",
    "temporarily", "banned", "exceeded",
    "free account is limited", "requests per day",
    "567", "edgeone", "tencent cloud", "请求已被拦截",
    "restricted access", "security policy", "blocked", "安全策略",
)


def _is_quota_error(exc: Exception) -> bool:
    return any(p in str(exc).lower() for p in _QUOTA_PATTERNS)


def probe_and_set_chat_provider() -> str:
    """
    Probe ChatAnywhere and set the active provider once at startup.
    Falls back to Ollama if the key is exhausted or invalid.
    Returns the chosen provider: 'openai' or 'ollama'.
    """
    global _active_chat_provider
    cfg = getattr(_settings, "chat_provider", "openai").lower()

    if cfg == "ollama":
        with _provider_lock:
            _active_chat_provider = "ollama"
        logger.info("Chat provider: Ollama (forced by config)")
        return "ollama"

    if cfg == "openai":
        with _provider_lock:
            _active_chat_provider = "openai"
        logger.info("Chat provider: ChatAnywhere (forced by config)")
        return "openai"

    # AUTO: tanpa key → langsung Ollama (tidak ada lagi key tertanam)
    if _openai_client is None:
        with _provider_lock:
            _active_chat_provider = "ollama"
        logger.info("Chat provider: Ollama (OPENAI_API_KEY tidak di-set)")
        return "ollama"

    # AUTO: probe with a 1-token call
    logger.info("Probing OpenAI-compatible API key…")
    try:
        _openai_client.chat.completions.create(
            model=_settings.chat_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        with _provider_lock:
            _active_chat_provider = "openai"
        logger.info("✅ ChatAnywhere key OK → using %s", _settings.chat_model)
        return "openai"
    except Exception as exc:
        if _is_quota_error(exc):
            with _provider_lock:
                _active_chat_provider = "ollama"
            logger.warning("⚠️  ChatAnywhere key exhausted — falling back to local Ollama.")
            return "ollama"
        with _provider_lock:
            _active_chat_provider = "openai"
        logger.warning("ChatAnywhere probe inconclusive (%s), defaulting to openai.", exc)
        return "openai"


def get_active_provider() -> str:
    with _provider_lock:
        return _active_chat_provider


def force_ollama() -> None:
    global _active_chat_provider
    with _provider_lock:
        _active_chat_provider = "ollama"
    logger.warning("Chat provider forced to Ollama.")


def query_llm(prompt: str) -> str:
    """
    Send prompt to the active chat provider and return the normalised response.
    Automatically falls back to Ollama on quota errors mid-session.
    """
    provider = get_active_provider()
    V.step(f"query_llm → provider={provider} "
           f"model={_settings.chat_model if provider == 'openai' else _settings.ollama_chat_model} "
           f"| panjang prompt: {len(prompt)} char")
    if provider == "openai":
        return _chat_openai(prompt)
    return _chat_ollama(prompt)


def query_llm_ollama_raw(prompt: str) -> str:
    """
    KONDISI B (revisi pasca-sidang item 1-2): kirim prompt APA ADANYA
    langsung ke Ollama lokal.

    Jaminan "murni tanpa bantuan":
      • TANPA SYSTEM_PROMPT tutor — berbeda dari query_llm() yang jalur
        OpenAI-nya menyisipkan SYSTEM_PROMPT sebagai system message.
      • TANPA fallback ke provider OpenAI — model & jalur identik dengan
        yang dipakai evaluasi skripsi (llama3 lokal).
      • Prompt yang dikirim = prompt yang di-echo ke terminal & UI
        (field prompt_sent), sehingga dosen bisa memverifikasi tidak ada
        konteks tersembunyi.
    """
    V.step(f"query_llm_ollama_raw (Kondisi B) → ollama/"
           f"{_settings.ollama_chat_model} | panjang prompt: {len(prompt)} char "
           f"| TANPA system prompt, TANPA fallback")
    return _chat_ollama(prompt)


def _chat_openai(prompt: str) -> str:
    if _openai_client is None:
        logger.warning("OPENAI_API_KEY tidak di-set — memakai Ollama.")
        return _chat_ollama(prompt)
    for attempt in range(_settings.chat_retries):
        try:
            resp = _openai_client.chat.completions.create(
                model=_settings.chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=_settings.chat_temperature,
            )
            return normalize_latex(resp.choices[0].message.content.strip())
        except Exception as exc:
            if _is_quota_error(exc):
                force_ollama()
                logger.info("Quota error mid-session — switching to Ollama…")
                return _chat_ollama(prompt)
            logger.warning("OpenAI attempt %d/%d: %s", attempt + 1, _settings.chat_retries, exc)
            time.sleep(_settings.chat_retry_delay)
    return _chat_ollama(prompt)


def _chat_ollama(prompt: str) -> str:
    import requests as _req
    model      = getattr(_settings, "ollama_chat_model", "llama3")
    base_url   = getattr(_settings, "ollama_base_url", "http://localhost:11434")
    try:
        resp = _req.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            return normalize_latex(text)
    except Exception as exc:
        logger.warning("Ollama /api/generate failed: %s", exc)
    try:
        import requests as _req
        resp = _req.post(
            f"{base_url}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        if text:
            return normalize_latex(text)
    except Exception as exc:
        logger.error("Ollama also failed: %s", exc)
    return "[ERROR] Semua LLM tidak tersedia. Jalankan: ollama serve"


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS — Ollama-first, auto-fallback to ChatAnywhere jika Ollama mati
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> List[float]:
    """
    Return a normalised L2 embedding vector.

    Strategi (Ollama-first):
      1. Coba Ollama lokal terlebih dahulu — gratis, tidak ada kuota.
      2. Jika Ollama tidak bisa dijangkau (ConnectionError / timeout),
         fallback ke OpenAI/ChatAnywhere.
      3. Jika ChatAnywhere juga 429 / quota habis → raise langsung
         (rag.py sudah handle error ini dengan log + skip chunk).

    Tidak pernah mencoba ChatAnywhere jika Ollama berjalan normal.

    BUKTI KONDISI B (revisi pasca-sidang item 2): saat mode B aktif,
    assert_rag_allowed() melempar RAGBlockedError sehingga TIDAK ADA
    embedding yang bisa dihitung — dan setiap upaya tercatat sebagai
    embedding_calls_blocked pada no_rag_proof.
    """
    assert_rag_allowed("embedding")
    try:
        return _embed_ollama(text)
    except Exception as ollama_exc:
        # Hanya fallback ke OpenAI jika Ollama benar-benar tidak bisa dijangkau.
        # Jika Ollama running tapi model belum di-pull, error-nya bukan ConnectionError
        # tapi HTTP 404 — itu tetap dianggap "Ollama tidak tersedia" dan fallback ke OpenAI.
        logger.warning(
            "Ollama embedding gagal (%s) — fallback ke ChatAnywhere.",
            ollama_exc,
        )
        try:
            return _embed_openai(text)
        except Exception as openai_exc:
            # Keduanya gagal — lempar error asli Ollama agar rag.py tahu
            raise RuntimeError(
                f"Embedding gagal total. Ollama: {ollama_exc} | ChatAnywhere: {openai_exc}"
            ) from ollama_exc


def _embed_ollama(text: str) -> List[float]:
    import requests as _req
    base_url    = getattr(_settings, "ollama_base_url",    "http://localhost:11434")
    embed_model = getattr(_settings, "ollama_embed_model", "nomic-embed-text")
    resp = _req.post(
        f"{base_url}/api/embeddings",
        json={"model": embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    vec  = np.array(resp.json()["embedding"], dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()


def _embed_openai(text: str) -> List[float]:
    resp = _require_openai_client().embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    vec  = np.array(resp.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()