import os
from functools import lru_cache
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── OpenAI / ChatAnywhere (chat completions) ──────────────────────────
    openai_api_key:      str   = ""
    openai_api_base:     str   = "https://api.chatanywhere.org/v1"
    chat_model:          str   = "gpt-3.5-turbo"
    embedding_model:     str   = "text-embedding-3-small"
    chat_temperature:    float = 0.7
    chat_retries:        int   = 3
    chat_retry_delay:    int   = 2

    # ── Chat provider routing ─────────────────────────────────────────────
    # auto   → probe ChatAnywhere at startup; fall back to Ollama if quota exhausted
    # openai → always use ChatAnywhere/OpenAI
    # ollama → always use local Ollama
    chat_provider:       str   = "auto"

    # ── Embedding provider ────────────────────────────────────────────────
    # openai → OpenAI/ChatAnywhere embedding API (default)
    # ollama → local Ollama nomic-embed-text (avoids ChatAnywhere 200/day limit)
    embedding_provider:  str   = "openai"

    # ── Ollama (fallback / local) ─────────────────────────────────────────
    ollama_base_url:     str   = "http://localhost:11434"
    ollama_chat_model:   str   = "llama3"
    ollama_embed_model:  str   = "nomic-embed-text"

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_chunk_max_chars:  int = 400
    rag_embed_chunk_size: int = 800
    rag_top_k:            int = 4

    # ── Session / history ─────────────────────────────────────────────────
    max_history_chars: int = 1200

    # ── Server ────────────────────────────────────────────────────────────
    host:      str  = "0.0.0.0"
    port:      int  = 8000
    reload:    bool = True
    log_level: str  = "info"

    # ── Paths ─────────────────────────────────────────────────────────────
    base_dir: str = field(
        default_factory=lambda: os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
    )

    @property
    def materials_dir(self) -> str:
        return os.path.join(self.base_dir, "materials")

    @property
    def history_dir(self) -> str:
        return os.path.join(self.base_dir, "history_logs")

    @property
    def static_dir(self) -> str:
        return os.path.join(self.base_dir, "static")

    @property
    def rl_logs_dir(self) -> str:
        return os.path.join(self.base_dir, "rl_logs")

    @property
    def rl_plots_dir(self) -> str:
        return os.path.join(self.base_dir, "rl_plots")

    @property
    def eval_results_dir(self) -> str:
        return os.path.join(self.base_dir, "logs", "eval_results")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
