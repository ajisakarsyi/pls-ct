"""
app/main.py  — updated for MVP Feature 1 + Feature 2
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

    app = FastAPI(
        title="CSIPBLLM Personalized Learning System",
        version="3.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Static files ──────────────────────────────────────────────────────
    if os.path.isdir(settings.static_dir):
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    # ── All API routes (tutor + question bank) ────────────────────────────
    app.include_router(api_router)

    # ── Student frontend (existing) ───────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def serve_index():
        path = os.path.join(settings.static_dir, "index.html")
        return FileResponse(path) if os.path.exists(path) else \
               JSONResponse({"error": "index.html not found"}, status_code=404)

    # ── Student Asah Otak page (new) ────────────────────────────────────────
    @app.get("/asah-otak", include_in_schema=False)
    def serve_asah_otak():
        path = os.path.join(settings.static_dir, "asah_otak.html")
        return FileResponse(path) if os.path.exists(path) else                JSONResponse({"error": "asah_otak.html not found in static/"}, status_code=404)

    # ── Admin question bank dashboard (new) ───────────────────────────────
    @app.get("/admin/questions", include_in_schema=False)
    def serve_admin_dashboard():
        path = os.path.join(settings.static_dir, "admin_questions.html")
        return FileResponse(path) if os.path.exists(path) else \
               JSONResponse({"error": "admin_questions.html not found in static/"}, status_code=404)

    # ── Provider status (debug) ───────────────────────────────────────────
    @app.get("/provider-status", include_in_schema=False)
    def provider_status():
        from app.services.llm import get_active_provider
        return {
            "chat_provider":      get_active_provider(),
            "embed_provider":     "ollama (always)",
            "ollama_embed_model": settings.ollama_embed_model,
            "ollama_base_url":    settings.ollama_base_url,
        }

    # ── Startup ───────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        from app.services.llm import probe_and_set_chat_provider

        # 1. Decide chat provider (ChatAnywhere if key valid, else Ollama)
        provider = probe_and_set_chat_provider()

        # 2. Build RAG index for Feature 2 (always local Ollama embeddings)
        logger.info("Building RAG index…")
        load_global_materials()

        # 3. Ensure data/ folder exists for question bank
        data_dir = os.path.join(settings.base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        logger.info(
            "━━ Server ready ━━  chat=%s | embed=ollama/%s | cognitive_types=%d",
            provider, settings.ollama_embed_model, len(VALID_COGNITIVE_TYPES),
        )
        logger.info("  Feature 1 admin dashboard : http://%s:%s/admin/questions",
                    settings.host, settings.port)
        logger.info("  Asah Otak (students)      : http://%s:%s/asah-otak",
                    settings.host, settings.port)
        logger.info("  Feature 2 student chat    : http://%s:%s/",
                    settings.host, settings.port)
        logger.info("  API docs                  : http://%s:%s/docs",
                    settings.host, settings.port)

    return app


app = create_app()