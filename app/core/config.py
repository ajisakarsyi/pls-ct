"""
app/core/config.py
───────────────────
Konfigurasi terpusat LogiCT — REVISI PASCA-SIDANG.

Perubahan penting dibanding versi sidang:

1. KEAMANAN — API key TIDAK lagi ditulis di kode. Key ChatAnywhere yang
   sebelumnya hardcoded di file ini (dan di services/llm.py) sudah DIHAPUS
   dan wajib dianggap BOCOR → segera revoke di dashboard ChatAnywhere.
   Bila ingin memakai provider OpenAI-compatible, set env OPENAI_API_KEY.

2. SELARAS SKRIPSI — nilai default kini persis dengan metodologi Bab 3:
     rag_top_k            = 6     (K=6, Pers. 2)
     rag_embed_chunk_size = 1200  (chunk 1.200 karakter, Bab 3.4.2)
     rag_chunk_max_chars  = 600   (potongan konteks per chunk di prompt —
                                   konsisten dengan evaluation/runner.py)
     chat_provider        = "ollama" (llama3 lokal, Bab 3.3)
     embedding_provider   = "ollama" (nomic-embed-text, Bab 3.4.2)

3. TRANSPARANSI — flag verbose/live metrics untuk revisi item 3
   (terminal menampilkan semua aktivitas latar + metrik live saat demo).

Semua nilai bisa dioverride lewat environment variable dengan nama
UPPER_CASE yang sama (mis. RAG_TOP_K=8 python main.py).
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


@dataclass
class Settings:
    # ── Provider utama: OLLAMA LOKAL (sesuai skripsi Bab 3.3) ─────────────
    # "ollama" → llama3 + nomic-embed-text lokal (default, sesuai skripsi)
    # "openai" → provider OpenAI-compatible; WAJIB set env OPENAI_API_KEY
    # "auto"   → coba OpenAI dulu bila key tersedia, fallback Ollama
    chat_provider:      str = field(default_factory=lambda: _env_str("CHAT_PROVIDER", "ollama"))
    embedding_provider: str = field(default_factory=lambda: _env_str("EMBEDDING_PROVIDER", "ollama"))

    # ── Ollama (lokal) ────────────────────────────────────────────────────
    ollama_base_url:    str = field(default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model:  str = field(default_factory=lambda: _env_str("OLLAMA_CHAT_MODEL", "llama3"))
    ollama_embed_model: str = field(default_factory=lambda: _env_str("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # ── OpenAI-compatible (opsional; TANPA default key!) ──────────────────
    # Key HANYA dari environment. Tidak ada fallback tertanam di kode.
    openai_api_key:   str   = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))
    openai_api_base:  str   = field(default_factory=lambda: _env_str("OPENAI_API_BASE", "https://api.openai.com/v1"))
    chat_model:       str   = field(default_factory=lambda: _env_str("CHAT_MODEL", "gpt-3.5-turbo"))
    embedding_model:  str   = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "text-embedding-3-small"))
    chat_temperature: float = field(default_factory=lambda: _env_float("CHAT_TEMPERATURE", 0.7))
    chat_retries:     int   = field(default_factory=lambda: _env_int("CHAT_RETRIES", 3))
    chat_retry_delay: int   = field(default_factory=lambda: _env_int("CHAT_RETRY_DELAY", 2))

    # ── RAG (selaras Bab 3.4.2 skripsi) ───────────────────────────────────
    rag_top_k:            int = field(default_factory=lambda: _env_int("RAG_TOP_K", 6))
    rag_embed_chunk_size: int = field(default_factory=lambda: _env_int("RAG_EMBED_CHUNK_SIZE", 1200))
    rag_chunk_max_chars:  int = field(default_factory=lambda: _env_int("RAG_CHUNK_MAX_CHARS", 600))

    # Ambang evaluasi (sumber definisi: evaluation/metrics.py — Pers. 2 & 8)
    theta_retrieval: float = field(default_factory=lambda: _env_float("THETA_RETRIEVAL", 0.25))
    theta_coverage:  float = field(default_factory=lambda: _env_float("THETA_COVERAGE", 0.35))

    # ── Transparansi demo (revisi pasca-sidang item 3) ────────────────────
    # verbose            → cetak semua aktivitas latar ke terminal
    # demo_live_metrics  → hitung P@K/Coverage/MeanSim/Diversity + scan
    #                      uncertainty/contradiction utk tiap jawaban demo
    # demo_full_metrics  → tambahkan juga entailment LLM-as-Judge live
    #                      (lambat: +m panggilan Ollama per jawaban)
    verbose:           bool = field(default_factory=lambda: _env_bool("LOGICT_VERBOSE", True))
    demo_live_metrics: bool = field(default_factory=lambda: _env_bool("DEMO_LIVE_METRICS", True))
    demo_full_metrics: bool = field(default_factory=lambda: _env_bool("DEMO_FULL_METRICS", False))

    # ── Session / history ─────────────────────────────────────────────────
    max_history_chars: int = field(default_factory=lambda: _env_int("MAX_HISTORY_CHARS", 1200))

    # ── Server ────────────────────────────────────────────────────────────
    # Default "localhost" — bisa dibuka langsung di browser dan membuat log
    # bawaan uvicorn ("Uvicorn running on http://...") langsung jelas tanpa
    # perlu diterjemahkan. Untuk mengizinkan akses dari perangkat lain di
    # jaringan yang sama (mis. demo dari HP), set env HOST=0.0.0.0 manual.
    host:      str  = field(default_factory=lambda: _env_str("HOST", "localhost"))
    port:      int  = field(default_factory=lambda: _env_int("PORT", 8000))
    reload:    bool = field(default_factory=lambda: _env_bool("RELOAD", True))
    log_level: str  = field(default_factory=lambda: _env_str("LOG_LEVEL", "info"))

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