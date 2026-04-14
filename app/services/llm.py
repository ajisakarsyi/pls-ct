"""
app/services/llm.py
────────────────────
LLM wrapper with ChatAnywhere-first, Ollama-fallback logic.

Chat routing (set once at startup via probe_and_set_chat_provider):
  CHAT_PROVIDER=auto   → probe ChatAnywhere; use it if key valid, else fall back to Ollama
  CHAT_PROVIDER=openai → always ChatAnywhere/OpenAI
  CHAT_PROVIDER=ollama → always local Ollama

Embeddings:
  EMBEDDING_PROVIDER=openai  → OpenAI/ChatAnywhere embedding API  (default)
  EMBEDDING_PROVIDER=ollama  → local Ollama nomic-embed-text
"""

import logging
import threading
import time
from typing import List

import numpy as np
from openai import OpenAI

from app.core.config import get_settings
from app.core.prompts import SYSTEM_PROMPT
from app.utils.latex import normalize_latex

logger    = logging.getLogger(__name__)
_settings = get_settings()

_provider_lock         = threading.Lock()
_active_chat_provider: str = "openai"   # overwritten by probe_and_set_chat_provider()

_openai_client = OpenAI(
    api_key  = _settings.openai_api_key,
    base_url = _settings.openai_api_base,
)

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

    # AUTO: probe with a 1-token call
    logger.info("Probing ChatAnywhere API key…")
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
    if get_active_provider() == "openai":
        return _chat_openai(prompt)
    return _chat_ollama(prompt)


def _chat_openai(prompt: str) -> str:
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


def get_embedding(text: str) -> List[float]:
    """
    Return a normalised L2 embedding vector.
    Uses OpenAI/ChatAnywhere by default; Ollama if EMBEDDING_PROVIDER=ollama.
    """
    provider = getattr(_settings, "embedding_provider", "openai").lower()

    if provider == "ollama":
        return _embed_ollama(text)
    return _embed_openai(text)


def _embed_openai(text: str) -> List[float]:
    resp = _openai_client.embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    vec  = np.array(resp.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()


def _embed_ollama(text: str) -> List[float]:
    import requests as _req
    base_url   = getattr(_settings, "ollama_base_url",    "http://localhost:11434")
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