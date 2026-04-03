"""
app/ai/question_bank.py
────────────────────────
Question bank manager — handles the full lifecycle of AI-generated questions:

  pending  →  approved (live for students)
           →  rejected (archived, not shown to students)

Storage: JSON file (data/question_bank.json) — swap for DB later
         by replacing _load() and _save() with your ORM calls.

Key design decisions for the AI engineer role:
  - correct_answer is stored but NEVER included in student-facing responses
  - _sanitize() strips correct_answer before any student endpoint returns data
  - Admin endpoints return full data including correct_answer
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Storage path ──────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_DATA_DIR   = os.path.join(_BASE_DIR, "data")
_BANK_FILE  = os.path.join(_DATA_DIR, "question_bank.json")
_LOCK       = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# STORAGE LAYER — swap these two functions for DB integration later
# ══════════════════════════════════════════════════════════════════════════

def _load() -> List[Dict]:
    """Load all questions from JSON file. Returns empty list if file missing."""
    if not os.path.exists(_BANK_FILE):
        return []
    try:
        with open(_BANK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load question bank: %s", exc)
        return []


def _save(questions: List[Dict]) -> None:
    """Persist all questions to JSON file."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# SECURITY — correct_answer stripping
# ══════════════════════════════════════════════════════════════════════════

def _sanitize(question: Dict) -> Dict:
    """
    Return a copy of the question with correct_answer removed.
    Always call this before returning data to student-facing endpoints.
    """
    q = dict(question)
    q.pop("correct_answer", None)
    return q


def _sanitize_list(questions: List[Dict]) -> List[Dict]:
    return [_sanitize(q) for q in questions]


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def add_batch(questions: List[Dict]) -> int:
    """
    Add a batch of newly generated questions to the bank.
    All questions must have status='pending'.
    Returns count of questions added.
    """
    with _LOCK:
        existing   = _load()
        existing_ids = {q["id"] for q in existing}
        new_qs     = [q for q in questions if q["id"] not in existing_ids]
        _save(existing + new_qs)
    logger.info("Added %d questions to bank (%d already existed)", len(new_qs), len(questions) - len(new_qs))
    return len(new_qs)


def get_all(status: Optional[str] = None) -> List[Dict]:
    """
    Get all questions. Admin-only — includes correct_answer.
    Filter by status: 'pending' | 'approved' | 'rejected' | None (all)
    """
    questions = _load()
    if status:
        questions = [q for q in questions if q.get("status") == status]
    return questions


def get_pending() -> List[Dict]:
    """All pending questions. Admin-only (includes correct_answer)."""
    return get_all(status="pending")


def get_live(week_id: Optional[str] = None) -> List[Dict]:
    """
    Approved questions for students — correct_answer stripped.
    Optionally filter by week_id.
    """
    questions = get_all(status="approved")
    if week_id:
        questions = [q for q in questions if q.get("week_id") == week_id]
    return _sanitize_list(questions)


def get_by_id(question_id: str) -> Optional[Dict]:
    """Get single question by ID. Admin-only (includes correct_answer)."""
    for q in _load():
        if q["id"] == question_id:
            return q
    return None


def approve(question_id: str, admin_note: str = "") -> Optional[Dict]:
    """
    Approve a pending question. Returns the updated question or None if not found.
    """
    with _LOCK:
        questions = _load()
        for q in questions:
            if q["id"] == question_id:
                if q["status"] != "pending":
                    logger.warning("Tried to approve question %s with status=%s", question_id, q["status"])
                    return q
                q["status"]       = "approved"
                q["approved_at"]  = datetime.now(timezone.utc).isoformat()
                q["admin_note"]   = admin_note
                _save(questions)
                logger.info("Question approved: %s topic=%s", question_id, q.get("topic"))
                return q
    return None


def reject(question_id: str, reason: str = "") -> Optional[Dict]:
    """
    Reject a pending question. Returns the updated question or None if not found.
    """
    with _LOCK:
        questions = _load()
        for q in questions:
            if q["id"] == question_id:
                q["status"]      = "rejected"
                q["rejected_at"] = datetime.now(timezone.utc).isoformat()
                q["reject_reason"] = reason
                _save(questions)
                logger.info("Question rejected: %s reason=%s", question_id, reason)
                return q
    return None


def delete(question_id: str) -> bool:
    """Permanently delete a question. Admin-only."""
    with _LOCK:
        questions = _load()
        before    = len(questions)
        questions = [q for q in questions if q["id"] != question_id]
        if len(questions) < before:
            _save(questions)
            return True
    return False


def get_stats() -> Dict:
    """Summary stats for the admin dashboard."""
    questions = _load()
    weeks     = {}
    for q in questions:
        wid = q.get("week_id", "unknown")
        weeks.setdefault(wid, {"pending": 0, "approved": 0, "rejected": 0})
        weeks[wid][q.get("status", "pending")] += 1

    return {
        "total":    len(questions),
        "pending":  sum(1 for q in questions if q.get("status") == "pending"),
        "approved": sum(1 for q in questions if q.get("status") == "approved"),
        "rejected": sum(1 for q in questions if q.get("status") == "rejected"),
        "by_week":  weeks,
        "topics":   list({q.get("topic", "") for q in questions}),
    }