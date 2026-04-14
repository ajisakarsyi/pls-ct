"""
app/api/__init__.py
────────────────────
Aggregate all route modules into a single APIRouter.
"""

from fastapi import APIRouter

from .routes.reference     import router as reference_router
from .routes.tutor         import router as tutor_router
from .routes.history       import router as history_router
from .routes.rl            import router as rl_router
from .routes.question_bank import router as question_bank_router

api_router = APIRouter()
api_router.include_router(reference_router)
api_router.include_router(tutor_router)
api_router.include_router(history_router)
api_router.include_router(rl_router)
api_router.include_router(question_bank_router)

__all__ = ["api_router"]
