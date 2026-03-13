"""
app/api/routes/tutor.py
────────────────────────
/chat  and  /evaluate  endpoints.

These are the two main endpoints students interact with.
RL selection (which LT/cognitive type to use) is handled here,
then the resolved cognitive_code is passed to the tutor service.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.core.cognitive import is_valid, DEFAULT_COGNITIVE_TYPE
from app.models.schemas import ChatRequest, ChatResponse, EvalRequest, EvalResponse
from app.services.tutor import generate_reply, evaluate_student_answer
from app.services.rl import (
    rl_registry,
    rl_select_cognitive,
    rl_record_response,
    parse_cognitive,
    build_cognitive_code,
    MASTERY_LABELS,
    RL_LTS,
)

router = APIRouter(tags=["Tutor"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Kirim pertanyaan dan terima respons tutor personal",
)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Main tutoring endpoint.

    The RL agent selects the best cognitive type / LT for this student
    during the free phase.  During the seeding phase (first 10 questions)
    the LT from the *cognitive* field is used as seeding_lt.

    Returns the tutor reply, a follow-up question, and RL metadata.
    """
    cognitive_code, rl_lt, rl_selected, rl_phase = rl_select_cognitive(
        session_id          = req.session_id,
        category            = req.category,
        requested_cognitive = req.cognitive,
    )
    if not is_valid(cognitive_code):
        cognitive_code = DEFAULT_COGNITIVE_TYPE

    agent = rl_registry.get_agent(req.session_id, category=req.category)

    result = generate_reply(
        message        = req.message,
        cognitive_code = cognitive_code,
        session_id     = req.session_id,
        rl_selected    = rl_selected,
        rl_phase       = rl_phase,
    )

    phase_info = {
        "phase":                rl_phase,
        "seeding_lt":           agent.seeding_lt,
        "seeding_sessions":     agent.seeding_questions,
        "seeding_remaining":    agent.seeding_remaining,
        "seeding_progress_pct": round(agent.seeding_progress_pct, 3),
        "global_question_count": agent.total_selections,
    }

    return ChatResponse(
        reply             = result["reply"],
        followup_question = result["followup_question"],
        cognitive         = cognitive_code,
        session_id        = req.session_id,
        rl_selected_lt    = rl_lt,
        rl_selected       = rl_selected,
        rl_epsilon        = round(agent.epsilon, 4),
        rl_q_values       = {lt: round(q, 5) for lt, q in agent.q_values().items()},
        rl_phase          = phase_info,
    )


@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Evaluasi jawaban mahasiswa dengan umpan balik scaffolded adaptif",
)
def evaluate(req: EvalRequest) -> EvalResponse:
    """
    Evaluate a student's answer.

    After evaluating correctness and generating feedback, the result is
    fed into the RL agent (record_response) to update Q-values and mastery.
    The updated RL state is returned alongside the feedback.
    """
    cognitive_code = req.cognitive.strip().upper()
    if not is_valid(cognitive_code):
        cognitive_code = DEFAULT_COGNITIVE_TYPE

    parsed = parse_cognitive(cognitive_code)
    mastery_level, lt = parsed if parsed else (1, "PAR")

    # ── 1. Evaluate answer ───────────────────────────────────────────────
    eval_result = evaluate_student_answer(
        answer          = req.answer.strip(),
        correct_answer  = req.correct_answer,
        active_question = req.active_question,
        wrong_count     = req.wrong_count,
        cognitive_code  = cognitive_code,
        session_id      = req.session_id,
    )

    # ── 2. Record in RL agent ────────────────────────────────────────────
    rl_step, next_cognitive, lt_change_info = rl_record_response(
        session_id       = req.session_id,
        category         = req.category,
        lt               = lt,
        mastery_level    = mastery_level,
        cognitive_code   = cognitive_code,
        is_correct       = eval_result["is_correct"],
        wrong_count      = req.wrong_count,
        t_answer_seconds = req.t_answer_seconds,
    )

    # ── 3. Build RL response block ───────────────────────────────────────
    agent = rl_registry.get_agent(req.session_id, category=req.category)
    rl_block = None
    if rl_step:
        rl_block = {
            "learning_type":    rl_step["learning_type"],
            "mastery_level":    rl_step["mastery_level"],
            "mastery_label":    rl_step["mastery_label"],
            "mastery_score":    rl_step["mastery_score"],
            "performance":      rl_step["performance"],
            "engagement":       rl_step["engagement"],
            "reward":           rl_step["reward"],
            "t_expected":       rl_step["t_expected"],
            "mlr_refitted":     rl_step["mlr_refitted"],
            "next_cognitive":   next_cognitive,
            "phase":            agent.phase,
            "seeding_remaining": agent.seeding_remaining,
        }

    return EvalResponse(
        is_correct        = eval_result["is_correct"],
        feedback          = eval_result["feedback"],
        hint_level        = eval_result["hint_level"],
        followup_question = eval_result["followup_question"],
        cognitive         = cognitive_code,
        session_id        = req.session_id,
        rl                = rl_block,
        lt_change         = lt_change_info,
        total_lt_changes  = len(agent.recommendation_history),
    )
