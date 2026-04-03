"""
app/api/__init__.py  — UPDATED VERSION
────────────────────────────────────────
Add the question_bank router to the existing api_router.

INSTRUCTIONS: Replace your existing app/api/__init__.py with this file.
Or if you prefer, just add the two marked lines to your existing __init__.py.
"""

from fastapi import APIRouter

from app.api.routes.tutor     import router as tutor_router
from app.api.routes.history   import router as history_router
from app.api.routes.reference import router as reference_router
from app.api.routes.question_bank import router as question_bank_router   # ← ADD THIS

api_router = APIRouter()

api_router.include_router(tutor_router)
api_router.include_router(history_router)
api_router.include_router(reference_router)
api_router.include_router(question_bank_router)   # ← ADD THIS
