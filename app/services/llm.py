"""
app/services/llm.py
────────────────────
LLM wrapper with Ollama-first, ChatAnywhere-when-available logic.

Chat routing (decided once at startup):
  CHAT_PROVIDER=auto   → probe ChatAnywhere; use it if key valid, else Ollama
  CHAT_PROVIDER=ollama → always Ollama
  CHAT_PROVIDER=openai → always ChatAnywhere

Embeddings: ALWAYS local Ollama. Never calls remote embedding API.
"""

import logging
import threading
import time
from typing import List

import numpy as np
import requests
from openai import OpenAI

from app.core.config import get_settings
from app.core.prompts import SYSTEM_PROMPT
from app.utils.latex import normalize_latex

logger = logging.getLogger(__name__)
_settings = get_settings()

_provider_lock = threading.Lock()
_active_chat_provider: str = "ollama"

_openai_client = OpenAI(
    api_key=_settings.openai_api_key,
    base_url=_settings.openai_api_base,
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
    Probe ChatAnywhere and set the active provider.
    Called once at startup. Returns chosen provider: 'openai' or 'ollama'.
    """
    global _active_chat_provider
    cfg = _settings.chat_provider.lower()

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

    # AUTO: probe with a minimal 1-token call
    logger.info("Probing ChatAnywhere API key…")
    try:
        _openai_client.chat.completions.create(
            model=_settings.chat_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        with _provider_lock:
            _active_chat_provider = "openai"
        logger.info("✅ ChatAnywhere key OK → using %s for chat.", _settings.chat_model)
        return "openai"
    except Exception as exc:
        if _is_quota_error(exc):
            with _provider_lock:
                _active_chat_provider = "ollama"
            logger.warning(
                "⚠️  ChatAnywhere key exhausted — falling back to local Ollama (%s).",
                _settings.ollama_chat_model,
            )
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
                logger.info("Quota error mid-session, switching to Ollama…")
                return _chat_ollama(prompt)
            logger.warning("OpenAI attempt %d/%d: %s", attempt + 1, _settings.chat_retries, exc)
            time.sleep(_settings.chat_retry_delay)
    return _chat_ollama(prompt)


def _chat_ollama(prompt: str) -> str:
    model = _settings.ollama_chat_model
    try:
        resp = requests.post(
            f"{_settings.ollama_base_url}/api/generate",
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
        resp = requests.post(
            f"{_settings.ollama_base_url}/api/chat",
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
        logger.error("Ollama also failed: %s — run: ollama serve && ollama pull %s", exc, model)

    return "[ERROR] Semua LLM tidak tersedia. Jalankan: ollama serve"


def get_embedding(text: str) -> List[float]:
    """Always local Ollama — never calls remote embedding API."""
    resp = requests.post(
        f"{_settings.ollama_base_url}/api/embeddings",
        json={"model": _settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()