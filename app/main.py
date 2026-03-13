"""
app/main.py
────────────
FastAPI application factory for CSIPBLLM (Combined RAG + RL).

Startup:
  - Preloads global RAG materials index (shared embeddings)
  - Initialises RL session registry with correct log directory
  - Serves static frontend from /static

Architecture:
  /chat        → RL selects cognitive type  → RAG retrieves context  → LLM answers
  /evaluate    → LLM strict-evaluates answer  → RL records reward, updates Q-table
  /rl/*        → RL introspection (phase, Q-values, plots, summary …)
  /history     → conversation logs (history_logs/)
  /cognitive-types → reference list of all 48 codes
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
    ):
        os.makedirs(d, exist_ok=True)

    app = FastAPI(
        title="CSIPBLLM Personalized Learning System",
        description=(
            "Sistem tutor adaptif yang menggabungkan:\n"
            "• **RAG** (Retrieval-Augmented Generation) — jawaban berbasis materi\n"
            "• **RL Contextual Bandit** (seeding 10 soal → free phase) — pemilihan LT optimal\n"
            "• **48 tipe kognitif** (Level 1–6 × P/T × A/G × I/R)\n\n"
            "Run: `uvicorn app.main:app --reload`  |  Docs: `/docs`"
        ),
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Static files ──────────────────────────────────────────────────────
    if os.path.isdir(settings.static_dir):
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Frontend entry-point ──────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def serve_index():
        index_path = os.path.join(settings.static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"message": "CSIPBLLM API running — visit /docs"})

    # ── Startup ───────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        logger.info("Preloading global RAG materials index…")
        load_global_materials()
        logger.info(
            "Server ready — %d cognitive types | RAG + RL active.",
            len(VALID_COGNITIVE_TYPES),
        )

    return app


app = create_app()
