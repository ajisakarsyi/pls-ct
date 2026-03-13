"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables.
All configuration for the app lives here — no magic strings scattered around.
"""

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── OpenAI / ChatAnywhere (used for chat completions) ─────────────────
    openai_api_key: str = "YOUR_GPT_API_KEY_HERE"
    openai_api_base: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    chat_temperature: float = 0.7
    chat_retries: int = 3
    chat_retry_delay: int = 2

    # ── Embedding provider ────────────────────────────────────────────────
    # EMBEDDING_PROVIDER=ollama  → use local Ollama (avoids ChatAnywhere 200/day limit)
    # EMBEDDING_PROVIDER=openai  → use OpenAI/ChatAnywhere embedding API
    embedding_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_chunk_max_chars: int = 400
    rag_embed_chunk_size: int = 800
    rag_top_k: int = 4

    # ── Session / history ─────────────────────────────────────────────────
    max_history_chars: int = 1200

    # ── Paths (resolved relative to project root) ─────────────────────────
    base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    @property
    def materials_dir(self) -> str:
        return os.path.join(self.base_dir, "materials")

    @property
    def history_dir(self) -> str:
        return os.path.join(self.base_dir, "logs", "history")

    @property
    def eval_results_dir(self) -> str:
        return os.path.join(self.base_dir, "logs", "eval_results")

    @property
    def static_dir(self) -> str:
        return os.path.join(self.base_dir, "static")

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()