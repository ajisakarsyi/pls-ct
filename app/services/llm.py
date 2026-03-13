"""
app/services/llm.py
────────────────────
Thin wrapper around the OpenAI chat-completions and embeddings APIs.
All retry logic and LaTeX normalisation live here.
"""

import time
import logging
from typing import List

import numpy as np
from openai import OpenAI

from app.core.config import get_settings
from app.core.prompts import SYSTEM_PROMPT
from app.utils.latex import normalize_latex

logger   = logging.getLogger(__name__)
_settings = get_settings()

_client = OpenAI(
    api_key  = _settings.openai_api_key,
    base_url = _settings.openai_api_base,
)
logger.info("LLM client initialised — base: %s  model: %s",
            _settings.openai_api_base, _settings.chat_model)


def query_llm(prompt: str) -> str:
    """
    Send *prompt* to the chat-completions endpoint and return the
    normalised text response.  Retries up to ``settings.chat_retries`` times.
    """
    for attempt in range(_settings.chat_retries):
        try:
            resp = _client.chat.completions.create(
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


def get_embedding(text: str) -> List[float]:
    """
    Return a normalised L2 embedding vector for *text*.
    Raises on failure — callers should handle exceptions.
    """
    resp = _client.embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    vec  = np.array(resp.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()
