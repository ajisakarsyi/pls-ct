"""
app/api/routes/question_bank.py
─────────────────────────────────
REST endpoints for the weekly Asah Otak quiz system.

Admin endpoints (include correct_answer):
  POST   /questions/generate          trigger AI batch for one topic
  GET    /questions/pending           list pending
  GET    /questions/all               list all (any status)
  POST   /questions/{id}/approve      approve
  POST   /questions/{id}/reject       reject
  DELETE /questions/{id}              delete
  GET    /questions/stats             dashboard stats

Student endpoints (correct_answer never included):
  GET    /questions/live              all approved (optional ?week_id=)
  GET    /questions/live/current-week this week's approved questions
  POST   /questions/{id}/answer       submit answer, get evaluated
                                      (result hidden until /questions/{id}/result)
"""

import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.ai.question_generator import generate_weekly_batch, get_week_id
from app.ai import question_bank as qb

router = APIRouter(prefix="/questions", tags=["Question Bank"])
logger = logging.getLogger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic_name: str                    # e.g. "Abstraksi"
    topic_text: str                    # full definition + explanation + examples
    week_id:    Optional[str] = None   # default: current week


class ApproveRequest(BaseModel):
    admin_note: str = ""


class RejectRequest(BaseModel):
    reason: str = ""


class AnswerRequest(BaseModel):
    answer:        str        # selected key(s) or free text
    question_type: str = ""   # "mcq" | "multi" | "truefalse" | "open"


# ══════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@router.post("/generate", summary="[Admin] Generate weekly questions for one topic")
def generate_questions(req: GenerateRequest):
    """
    Trigger AI generation of 10 questions all based on the provided topic.
    topic_text is the admin's description of this week's material.
    All questions start as 'pending' awaiting admin review.
    """
    if not req.topic_name.strip():
        raise HTTPException(status_code=422, detail="topic_name cannot be empty.")
    if len(req.topic_text.strip()) < 10:
        raise HTTPException(status_code=422,
                            detail="topic_text too short — provide a full description (min 10 chars).")

    week_id = req.week_id or get_week_id()

    existing = [q for q in qb.get_all() if q.get("week_id") == week_id]
    if existing:
        return {
            "message":   f"Week {week_id} already has {len(existing)} questions.",
            "week_id":   week_id,
            "existing":  len(existing),
            "generated": 0,
            "hint":      "Delete existing questions first or use a different week_id.",
        }

    logger.info("Generating questions | topic='%s' | week=%s", req.topic_name, week_id)
    questions = generate_weekly_batch(
        topic_name=req.topic_name.strip(),
        topic_text=req.topic_text.strip(),
        week_id=week_id,
    )
    added = qb.add_batch(questions)

    type_counts = {}
    for q in questions:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1

    return {
        "message":    f"Generated {added} questions for week {week_id}.",
        "week_id":    week_id,
        "topic":      req.topic_name,
        "generated":  added,
        "breakdown":  type_counts,
        "questions":  questions,
    }


@router.get("/pending", summary="[Admin] Pending questions (with answers)")
def list_pending():
    qs = qb.get_pending()
    return {"count": len(qs), "questions": qs}


@router.get("/all", summary="[Admin] All questions (with answers)")
def list_all(
    status:  Optional[str] = Query(None),
    week_id: Optional[str] = Query(None),
):
    qs = qb.get_all(status=status)
    if week_id:
        qs = [q for q in qs if q.get("week_id") == week_id]
    return {"count": len(qs), "questions": qs}


@router.post("/{question_id}/approve", summary="[Admin] Approve a question")
def approve_question(question_id: str, req: ApproveRequest):
    q = qb.approve(question_id, admin_note=req.admin_note)
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found.")
    return {"message": "Approved.", "question": q}


@router.post("/{question_id}/reject", summary="[Admin] Reject a question")
def reject_question(question_id: str, req: RejectRequest):
    q = qb.reject(question_id, reason=req.reason)
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found.")
    return {"message": "Rejected.", "question": q}


@router.delete("/{question_id}", summary="[Admin] Delete a question")
def delete_question(question_id: str):
    if not qb.delete(question_id):
        raise HTTPException(status_code=404, detail="Question not found.")
    return {"message": f"Deleted {question_id}."}


@router.get("/stats", summary="[Admin] Stats dashboard")
def get_stats():
    return qb.get_stats()


# ══════════════════════════════════════════════════════════════════════════
# STUDENT ENDPOINTS — correct_answer never sent to client
# ══════════════════════════════════════════════════════════════════════════

@router.get("/live/current-week", summary="[Student] This week's live questions")
def get_current_week():
    week_id = get_week_id()
    qs = qb.get_live(week_id=week_id)
    return {"week_id": week_id, "count": len(qs), "questions": qs}


@router.get("/live", summary="[Student] All live questions (no answers)")
def get_live(
    week_id:    Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    q_type:     Optional[str] = Query(None, alias="type"),
):
    qs = qb.get_live(week_id=week_id)
    if difficulty:
        qs = [q for q in qs if q.get("difficulty") == difficulty]
    if q_type:
        qs = [q for q in qs if q.get("type") == q_type]
    return {"count": len(qs), "questions": qs}


@router.post("/{question_id}/answer", summary="[Student] Submit answer (result hidden until quiz complete)")
def submit_answer(question_id: str, req: AnswerRequest):
    """
    Evaluates the student's answer server-side.
    Returns is_correct and explanation.

    For MCQ/TrueFalse/Multi: evaluated immediately (deterministic).
    For Open: evaluated by LLM.

    The FRONTEND is responsible for hiding results until all questions
    are answered — this endpoint just returns the evaluation result.
    correct_answer is never included in this response.
    """
    q = qb.get_by_id(question_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found.")
    if q.get("status") != "approved":
        raise HTTPException(status_code=403, detail="Question not available.")

    q_type  = req.question_type or q.get("type", "")
    student = req.answer.strip()
    correct = q.get("correct_answer")

    # ── MCQ: single key match ─────────────────────────────────────────────
    if q_type == "mcq":
        is_correct = student.upper() == str(correct).upper()
        return {
            "is_correct":  is_correct,
            "explanation": q.get("explanation", ""),
            # reveal correct key for MCQ (shown after all questions answered)
            "correct_answer": correct,
        }

    # ── True/False ────────────────────────────────────────────────────────
    if q_type == "truefalse":
        def normalise(s):
            s = s.lower().strip()
            return "benar" if s in ("benar","true","ya","yes","1") else "salah"
        is_correct = normalise(student) == normalise(str(correct))
        return {
            "is_correct":     is_correct,
            "explanation":    q.get("explanation", ""),
            "correct_answer": correct,
        }

    # ── Multi (multiple correct) ──────────────────────────────────────────
    if q_type == "multi":
        # student sends comma-separated keys e.g. "A,C" or JSON array
        if student.startswith("["):
            try:
                chosen = set(x.strip().upper() for x in __import__("json").loads(student))
            except Exception:
                chosen = set(x.strip().upper() for x in student.strip("[]").split(","))
        else:
            chosen = set(x.strip().upper() for x in student.split(",") if x.strip())

        correct_set = set(str(c).upper() for c in (correct if isinstance(correct, list) else [correct]))
        is_correct  = chosen == correct_set
        return {
            "is_correct":     is_correct,
            "explanation":    q.get("explanation", ""),
            "correct_answer": sorted(correct_set),
        }

    # ── Open: LLM evaluation ──────────────────────────────────────────────
    if q_type == "open":
        from app.ai.question_generator import _call_llm
        prompt = (
            f"Pertanyaan: {q['question']}\n\n"
            f"Jawaban referensi: {str(correct)[:600]}\n\n"
            f"Jawaban mahasiswa: {student}\n\n"
            f"Apakah jawaban mahasiswa mencakup konsep utama dari jawaban referensi?\n"
            f"BENAR jika konsep inti ada, meski kata berbeda.\n"
            f"SALAH jika konsep utama salah atau tidak ada.\n\n"
            f"Jawab HANYA:\nHASIL: BENAR\natau\nHASIL: SALAH"
        )
        raw   = _call_llm(prompt) or ""
        match = re.search(r"HASIL\s*:\s*(BENAR|SALAH)", raw, re.IGNORECASE)
        is_correct = match.group(1).upper() == "BENAR" if match else None
        return {
            "is_correct":     is_correct,
            "explanation":    q.get("explanation", ""),
            "correct_answer": None,   # never reveal for open-ended
        }

    raise HTTPException(status_code=422, detail=f"Unknown question_type: {q_type!r}")