"""
app/api/routes/question_bank.py
─────────────────────────────────
REST endpoints for the two-feature MVP:

Feature 1 — Weekly AI Question Bank (admin + student):
  POST /questions/generate          → trigger AI batch generation
  GET  /questions/pending           → admin: list pending questions (with answers)
  GET  /questions/all               → admin: all questions any status
  POST /questions/{id}/approve      → admin: approve question
  POST /questions/{id}/reject       → admin: reject question
  DELETE /questions/{id}            → admin: permanently delete
  GET  /questions/stats             → admin: dashboard summary
  GET  /questions/live              → student: approved questions (NO correct_answer)
  GET  /questions/live/{week_id}    → student: approved questions for specific week

Feature 2 — On-demand chat (existing, unchanged):
  POST /chat        → already in app/api/routes/tutor.py
  POST /evaluate    → already in app/api/routes/tutor.py

correct_answer visibility:
  Admin endpoints  → full question dict including correct_answer
  Student endpoints → correct_answer stripped via question_bank._sanitize()
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.ai.question_generator import generate_weekly_batch, get_week_id
from app.ai import question_bank as qb

router = APIRouter(prefix="/questions", tags=["Question Bank"])
logger = logging.getLogger(__name__)


# ── Request schemas ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    week_id:  Optional[str] = None   # override week, default = current week
    n_mcq:    int = 6
    n_open:   int = 4
    topics:   Optional[list] = None  # override topic list


class ApproveRequest(BaseModel):
    admin_note: str = ""


class RejectRequest(BaseModel):
    reason: str = ""


# ══════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@router.post("/generate", summary="[Admin] Generate weekly AI questions")
def generate_questions(req: GenerateRequest):
    """
    Trigger AI generation of a weekly batch.
    All generated questions start with status='pending'.
    Returns a summary — generation may take 1-3 minutes for 10 questions.
    """
    week_id = req.week_id or get_week_id()

    # Check if this week already has questions
    existing = qb.get_all()
    week_existing = [q for q in existing if q.get("week_id") == week_id]
    if week_existing:
        return {
            "message": f"Week {week_id} already has {len(week_existing)} questions. "
                       f"Delete them first or use a different week_id.",
            "week_id":  week_id,
            "existing": len(week_existing),
            "generated": 0,
        }

    logger.info("Admin triggered question generation for week=%s", week_id)
    questions = generate_weekly_batch(
        week_id=week_id,
        n_mcq=req.n_mcq,
        n_open=req.n_open,
        topics=req.topics,
    )

    added = qb.add_batch(questions)

    return {
        "message":   f"Generated {added} questions for week {week_id}.",
        "week_id":   week_id,
        "generated": added,
        "breakdown": {
            "mcq":  sum(1 for q in questions if q["type"] == "mcq"),
            "open": sum(1 for q in questions if q["type"] == "open"),
        },
        "topics": list({q["topic"] for q in questions}),
    }


@router.get("/pending", summary="[Admin] List pending questions (with answers)")
def list_pending():
    """Returns all pending questions including correct_answer. Admin only."""
    questions = qb.get_pending()
    return {"count": len(questions), "questions": questions}


@router.get("/all", summary="[Admin] List all questions any status (with answers)")
def list_all(
    status: Optional[str] = Query(None, description="Filter: pending | approved | rejected"),
    week_id: Optional[str] = Query(None),
):
    """Returns all questions including correct_answer. Admin only."""
    questions = qb.get_all(status=status)
    if week_id:
        questions = [q for q in questions if q.get("week_id") == week_id]
    return {"count": len(questions), "questions": questions}


@router.post("/{question_id}/approve", summary="[Admin] Approve a question")
def approve_question(question_id: str, req: ApproveRequest):
    q = qb.approve(question_id, admin_note=req.admin_note)
    if q is None:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found.")
    return {"message": "Question approved.", "question": q}


@router.post("/{question_id}/reject", summary="[Admin] Reject a question")
def reject_question(question_id: str, req: RejectRequest):
    q = qb.reject(question_id, reason=req.reason)
    if q is None:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found.")
    return {"message": "Question rejected.", "question": q}


@router.delete("/{question_id}", summary="[Admin] Permanently delete a question")
def delete_question(question_id: str):
    deleted = qb.delete(question_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found.")
    return {"message": f"Question {question_id} deleted."}


@router.get("/stats", summary="[Admin] Question bank summary stats")
def get_stats():
    return qb.get_stats()


# ══════════════════════════════════════════════════════════════════════════
# STUDENT ENDPOINTS — correct_answer NEVER included
# ══════════════════════════════════════════════════════════════════════════

@router.get("/live", summary="[Student] Get all live questions (no answers)")
def get_live_questions(
    week_id: Optional[str] = Query(None, description="Filter by week e.g. 2026-W12"),
    topic:   Optional[str] = Query(None, description="Filter by CT topic"),
    difficulty: Optional[str] = Query(None, description="mudah | sedang | sulit"),
    q_type: Optional[str] = Query(None, alias="type", description="mcq | open"),
):
    """
    Returns approved questions for students.
    correct_answer is NEVER included in this response.
    """
    questions = qb.get_live(week_id=week_id)

    if topic:
        questions = [q for q in questions if topic.lower() in q.get("topic", "").lower()]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if q_type:
        questions = [q for q in questions if q.get("type") == q_type]

    return {
        "count":     len(questions),
        "week_id":   week_id or "all",
        "questions": questions,
    }


@router.get("/live/current-week", summary="[Student] Get this week's live questions")
def get_current_week_questions():
    """Shortcut: returns approved questions for the current ISO week."""
    week_id   = get_week_id()
    questions = qb.get_live(week_id=week_id)
    return {
        "week_id":   week_id,
        "count":     len(questions),
        "questions": questions,
    }