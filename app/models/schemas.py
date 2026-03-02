"""
app/models/schemas.py
──────────────────────
Pydantic request and response schemas for all API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.cognitive import DEFAULT_COGNITIVE_TYPE


# ── Request models ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="Pertanyaan atau pesan mahasiswa")
    cognitive: str = Field(
        DEFAULT_COGNITIVE_TYPE, description="Kode tipe kognitif, misal '3TGR'"
    )
    session_id: str = Field("default", description="Identifikasi sesi")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Apa itu algoritma?",
                "cognitive": "2TAR",
                "session_id": "mahasiswa-01",
            }
        }
    }


class EvalRequest(BaseModel):
    answer: str = Field(..., description="Jawaban mahasiswa")
    correct_answer: str = Field(..., description="Jawaban referensi / kunci (penjelasan tutor)")
    active_question: str = Field(
        "", description="Pertanyaan spesifik yang sedang dijawab mahasiswa"
    )
    wrong_count: int = Field(0, description="Jumlah percobaan salah sejauh ini")
    cognitive: str = Field(DEFAULT_COGNITIVE_TYPE, description="Kode tipe kognitif")
    session_id: str = Field("default", description="Identifikasi sesi")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "32/3",
                "correct_answer": "Penjelasan tutor tentang integral...",
                "active_question": "Berapa luas area di bawah f(x) = -x^2 + 4x pada interval [0,4]?",
                "wrong_count": 0,
                "cognitive": "2TAR",
                "session_id": "mahasiswa-01",
            }
        }
    }


# ── Response models ────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    reply: str
    followup_question: str
    cognitive: str
    session_id: str


class EvalResponse(BaseModel):
    is_correct: bool
    feedback: str
    hint_level: str
    followup_question: str
    cognitive: str
    session_id: str


class CognitiveTypeItem(BaseModel):
    code: str
    label: str


class CognitiveTypesResponse(BaseModel):
    cognitive_types: List[CognitiveTypeItem]


class HistoryResponse(BaseModel):
    history: Optional[List[Dict[str, Any]]] = None
    data: Optional[str] = None
