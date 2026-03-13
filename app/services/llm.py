"""
app/services/llm.py
────────────────────
Thin wrapper around LLM APIs.

Chat completions  → always uses OpenAI-compatible API (ChatAnywhere / OpenAI)
Embeddings        → routed based on EMBEDDING_PROVIDER in .env:
                    "ollama"  → local Ollama /api/embeddings  (default)
                    "openai"  → OpenAI/ChatAnywhere embeddings API

Setting EMBEDDING_PROVIDER=ollama avoids burning the ChatAnywhere free-tier
200 req/day limit on embeddings, which would block the app from starting.
"""

import logging
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

# ── Chat client (OpenAI-compatible — ChatAnywhere or real OpenAI) ──────────
_chat_client = OpenAI(
    api_key=_settings.openai_api_key,
    base_url=_settings.openai_api_base,
)
logger.info(
    "LLM chat client → %s  model=%s",
    _settings.openai_api_base, _settings.chat_model,
)

# ── Embedding provider selection ───────────────────────────────────────────
_USE_OLLAMA_EMBED = _settings.embedding_provider.lower() == "ollama"
logger.info(
    "Embedding provider → %s  (%s)",
    _settings.embedding_provider,
    f"ollama/{_settings.ollama_embed_model} @ {_settings.ollama_base_url}"
    if _USE_OLLAMA_EMBED
    else f"openai/{_settings.embedding_model}",
)


# ── Public: chat ───────────────────────────────────────────────────────────

def query_llm(prompt: str) -> str:
    """
    Send *prompt* to the chat-completions endpoint and return the
    normalised text response. Retries up to settings.chat_retries on failure.
    """
    for attempt in range(_settings.chat_retries):
        try:
            resp = _chat_client.chat.completions.create(
                model=_settings.chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=_settings.chat_temperature,
            )
            raw = resp.choices[0].message.content.strip()
            return normalize_latex(raw)
        except Exception as exc:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
            time.sleep(_settings.chat_retry_delay)
    return "[ERROR] LLM API tidak tersedia."


# ── Public: embeddings ─────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    """
    Return a normalised L2 embedding vector for *text*.
    Routes to Ollama or OpenAI based on EMBEDDING_PROVIDER setting.
    """
    if _USE_OLLAMA_EMBED:
        return _embed_ollama(text)
    return _embed_openai(text)


def _embed_ollama(text: str) -> List[float]:
    """Embed via local Ollama — zero API quota consumed."""
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


def _embed_openai(text: str) -> List[float]:
    """Embed via OpenAI/ChatAnywhere API."""
    resp = _chat_client.embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()