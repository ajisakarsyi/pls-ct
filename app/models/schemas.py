"""
app/models/schemas.py
──────────────────────
Pydantic request and response schemas for all API endpoints.
Includes RL-specific fields (category, rl_* response fields).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str            = Field(...,         description="Pertanyaan atau pesan mahasiswa")
    cognitive:  Optional[str]  = Field(None,        description="Opsional. Jika tidak diisi, RL Agent memilih LT secara otomatis (cold start).")
    session_id: str            = Field("default",   description="Identifikasi sesi")
    category:   str            = Field("Penggalang", description="Kategori siswa: SiKecil | Siaga | Penggalang | Penegak")
    mode:       str            = Field("A",         description="Kondisi eksperimen skripsi: 'A' = LLM+RAG+profil kognitif (default); 'B' = LLM murni tanpa RAG (prompt buta Lampiran 4, dibungkus NoRAGGuard).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message":    "Apa itu algoritma?",
                "session_id": "mahasiswa-01",
                "category":   "Penggalang",
                "mode":       "A",
            }
        }
    }


class EvalRequest(BaseModel):
    answer:           str            = Field(...,          description="Jawaban mahasiswa")
    correct_answer:   str            = Field(...,          description="Jawaban referensi / kunci (penjelasan tutor)")
    active_question:  str            = Field("",           description="Pertanyaan spesifik yang sedang dijawab mahasiswa")
    wrong_count:      int            = Field(0,            description="Jumlah percobaan salah sejauh ini")
    cognitive:        Optional[str]  = Field(None,         description="Opsional. Jika None, RL Agent menggunakan LT yang dipilih pada /chat terakhir.")
    session_id:       str            = Field("default",    description="Identifikasi sesi")
    t_answer_seconds: Optional[float]= Field(None,         description="Waktu menjawab (detik). Jika None, dihitung otomatis.")
    category:         str            = Field("Penggalang", description="Kategori siswa")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer":           "32/3",
                "correct_answer":   "Penjelasan tutor tentang integral...",
                "active_question":  "Berapa luas area di bawah f(x) = -x^2 + 4x pada interval [0,4]?",
                "wrong_count":      0,
                "session_id":       "mahasiswa-01",
                "t_answer_seconds": None,
                "category":         "Penggalang",
            }
        }
    }


# ── Response models ────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    reply:             str
    followup_question: str
    cognitive:         str
    session_id:        str
    # ── Transparansi Kondisi A/B (revisi pasca-sidang item 1-3) ────────
    mode:         str                            = "A"
    rag_used:     bool                           = True
    retrieved:    Optional[List[Dict[str, Any]]] = None  # rank/source/topic/score/preview per chunk
    live_metrics: Optional[Dict[str, Any]]       = None  # Pers. 2/6/8/10 + scan Pers. 18 utk jawaban ini
    no_rag_proof: Optional[Dict[str, Any]]       = None  # laporan NoRAGGuard (hanya mode B)
    prompt_sent:  Optional[str]                  = None  # prompt persis yang dikirim ke LLM
    # RL fields
    rl_selected_lt:    Optional[str]            = None
    rl_selected:       Optional[bool]           = None
    rl_epsilon:        Optional[float]          = None
    rl_q_values:       Optional[Dict[str, Any]] = None
    rl_phase:          Optional[Dict[str, Any]] = None


class EvalResponse(BaseModel):
    is_correct:         bool
    feedback:           str
    hint_level:         str
    followup_question:  str
    cognitive:          str
    session_id:         str
    # RL fields
    rl:                Optional[Dict[str, Any]] = None
    lt_change:         Optional[Dict[str, Any]] = None
    total_lt_changes:  Optional[int]            = None


class CognitiveTypeItem(BaseModel):
    code:  str
    label: str


class CognitiveTypesResponse(BaseModel):
    cognitive_types: List[CognitiveTypeItem]


class HistoryResponse(BaseModel):
    history: Optional[List[Dict[str, Any]]] = None
    data:    Optional[str]                  = None
