"""
app/main.py
────────────
FastAPI application factory.

Import ``create_app`` in tests or run the module directly via uvicorn.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
        description=(
            "Adaptive AI tutor with personalised cognitive profiles, "
            "RAG-augmented responses, and scaffolded evaluation."
        ),
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS Middleware ───────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files ──────────────────────────────────────────────────────
    if os.path.isdir(settings.static_dir):
        app.mount(
            "/static",
            StaticFiles(directory=settings.static_dir),
            name="static",
        )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Frontend entry-point ──────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def serve_index():
        index_path = os.path.join(settings.static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"error": "index.html not found"}, status_code=404)

    # ── Startup event ─────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        logger.info("Preloading global RAG materials index…")
        load_global_materials()
        logger.info(
            "Server ready — %d cognitive types supported.",
            len(VALID_COGNITIVE_TYPES),
        )

    return app


# Expose a module-level ``app`` instance so uvicorn can be pointed at
# ``app.main:app`` directly.
app = create_app()
