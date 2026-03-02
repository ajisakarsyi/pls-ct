"""
app/api/__init__.py
────────────────────
Aggregate all route modules into a single APIRouter so main.py stays clean.
"""

from fastapi import APIRouter

from .routes.reference import router as reference_router
from .routes.tutor import router as tutor_router
from .routes.history import router as history_router

api_router = APIRouter()
api_router.include_router(reference_router)
api_router.include_router(tutor_router)
api_router.include_router(history_router)

__all__ = ["api_router"]
