"""
app/main.py
────────────
FastAPI application factory for CSIPBLLM PLS-CT v5.0.0.

Modules:
  /          → Student tutoring UI (cold start RL)
  /asah-otak → Student weekly quiz (Asah Otak)
  /admin/questions → Admin question bank dashboard

Architecture:
  /chat        → RL selects LT (cold start)  → RAG context  → LLM answer
  /evaluate    → LLM evaluates  → RL reward update (resolved only)
  /rl/*        → RL introspection (Q-values, plots, evaluation)
  /questions/* → Question bank (admin CRUD + student quiz)
  /history     → Conversation logs
"""

import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.core.config import get_settings
from app.core.cognitive import VALID_COGNITIVE_TYPES
from app.services.rag import load_global_materials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    # Ensure required directories exist
    for d in (
        settings.history_dir,
        settings.rl_logs_dir,
        settings.rl_plots_dir,
        settings.eval_results_dir,
        settings.data_dir,
    ):
        os.makedirs(d, exist_ok=True)

    app = FastAPI(
        title="CSIPBLLM Personalized Learning System",
        description=(
            "Sistem tutor adaptif berbasis **cold start** — mahasiswa tidak perlu memilih Tipe Kognitif.\n\n"
            "• **RAG** — jawaban berbasis materi per LT\n"
            "• **RL Contextual Bandit** (ε=0.50 awal) — eksplorasi-eksploitasi LT otomatis\n"
            "• **Asah Otak** — kuis mingguan berbasis soal AI yang dikurasi admin\n"
            "• **8 Learning Types** × **6 level penguasaan** = 48 kode kognitif\n\n"
            "**Alur utama:** `POST /chat` → `POST /evaluate` (ulangi jika salah)\n\n"
            "Run: `python main.py`  |  Docs: `/docs`"
        ),
        version="5.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Static files ──────────────────────────────────────────────────────
    if os.path.isdir(settings.static_dir):
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Student tutoring UI (main cold-start interface) ───────────────────
    @app.get("/", include_in_schema=False)
    def serve_index():
        path = os.path.join(settings.static_dir, "index.html")
        return FileResponse(path) if os.path.exists(path) else \
               JSONResponse({"message": "CSIPBLLM API running — visit /docs"})

    # ── Asah Otak — student weekly quiz ──────────────────────────────────
    @app.get("/asah-otak", include_in_schema=False)
    def serve_asah_otak():
        path = os.path.join(settings.static_dir, "asah_otak.html")
        return FileResponse(path) if os.path.exists(path) else \
               JSONResponse({"error": "asah_otak.html not found"}, status_code=404)

    # ── Admin — question bank dashboard ──────────────────────────────────
    @app.get("/admin/questions", include_in_schema=False)
    def serve_admin_dashboard():
        path = os.path.join(settings.static_dir, "admin_questions.html")
        return FileResponse(path) if os.path.exists(path) else \
               JSONResponse({"error": "admin_questions.html not found"}, status_code=404)

    # ── Provider status (debug) ───────────────────────────────────────────
    @app.get("/provider-status", include_in_schema=False)
    def provider_status():
        from app.services.llm import get_active_provider
        return {
            "chat_provider":      get_active_provider(),
            "embed_provider":     settings.embedding_provider,
            "ollama_embed_model": settings.ollama_embed_model,
            "ollama_base_url":    settings.ollama_base_url,
            "chat_model":         settings.chat_model,
        }

    # ── Startup ───────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        from app.services.llm import probe_and_set_chat_provider

        # 1. Decide chat provider (ChatAnywhere if key OK, else Ollama)
        provider = probe_and_set_chat_provider()

        # 2. Build RAG index
        logger.info("Building RAG index…")
        load_global_materials()

        logger.info(
            "━━ Server ready ━━  chat=%s | embed=%s | cognitive_types=%d",
            provider, settings.embedding_provider, len(VALID_COGNITIVE_TYPES),
        )
        logger.info("  Tutoring UI    → http://%s:%s/", settings.host, settings.port)
        logger.info("  Asah Otak      → http://%s:%s/asah-otak", settings.host, settings.port)
        logger.info("  Admin panel    → http://%s:%s/admin/questions", settings.host, settings.port)
        logger.info("  API docs       → http://%s:%s/docs", settings.host, settings.port)

    return app


app = create_app()
