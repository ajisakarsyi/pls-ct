"""
app/api/routes/tutor.py
────────────────────────
/chat  and  /evaluate  endpoints.

Cold-start by default: the student does NOT choose a LT.
The RL agent selects the LT via epsilon-greedy from the very first question.
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
    summary="Kirim pertanyaan — RL Agent memilih LT secara otomatis (cold start)",
)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Main tutoring endpoint — cold start by default.

    The student sends ONLY their message and session_id.
    The RL agent selects the LT via epsilon-greedy from question 1.
    No seeding_lt is pre-assigned; the agent discovers the best LT
    through exploration-exploitation.

    The selected cognitive code is returned in the response so the
    frontend can display what the agent chose.

    REVISI PASCA-SIDANG (item 1): field ``mode`` memilih kondisi skripsi.
    mode="B" → LLM murni: RL DILEWATI seluruhnya (tidak ada pemilihan LT,
    tidak ada profil), prompt buta Lampiran 4, NoRAGGuard aktif; respons
    membawa no_rag_proof + prompt_sent + live_metrics sebagai bukti.
    """
    # ── KONDISI B: LLM murni — lewati RL & profil sepenuhnya ─────────────
    if str(req.mode).upper() == "B":
        result = generate_reply(
            message        = req.message,
            cognitive_code = "",          # tidak dipakai pada mode B
            session_id     = req.session_id,
            mode           = "B",
        )
        return ChatResponse(
            reply             = result["reply"],
            followup_question = result["followup_question"],
            cognitive         = result["cognitive"],      # "—"
            session_id        = req.session_id,
            mode              = "B",
            rag_used          = False,
            retrieved         = result["retrieved"],
            live_metrics      = result["live_metrics"],
            no_rag_proof      = result["no_rag_proof"],
            prompt_sent       = result["prompt_sent"],
        )

    # ── KONDISI A: alur asli (RL memilih LT → RAG → tutor) ───────────────
    # req.cognitive is None for cold-start (no user input)
    cognitive_code, rl_lt, rl_selected, rl_phase = rl_select_cognitive(
        session_id          = req.session_id,
        category            = req.category,
        requested_cognitive = req.cognitive,   # None → pure RL selection
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
        mode           = "A",
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
        mode              = "A",
        rag_used          = True,
        retrieved         = result.get("retrieved"),
        live_metrics      = result.get("live_metrics"),
        no_rag_proof      = None,
        prompt_sent       = result.get("prompt_sent"),
        rl_selected_lt    = rl_lt,
        rl_selected       = rl_selected,
        rl_epsilon        = round(agent.epsilon, 4),
        rl_q_values       = {lt: round(q, 5) for lt, q in agent.q_values().items()},
        rl_phase          = phase_info,
    )


@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Evaluasi jawaban mahasiswa — RL Agent diperbarui berdasarkan hasil",
)
def evaluate(req: EvalRequest) -> EvalResponse:
    """
    Evaluate a student's answer and update the RL agent.

    cognitive is optional — if not provided (cold start), the LT is derived
    from the agent's most recently selected LT so the reward is attributed
    to the correct action.
    """
    # ── Resolve cognitive code ────────────────────────────────────────────
    if req.cognitive:
        cognitive_code = req.cognitive.strip().upper()
        if not is_valid(cognitive_code):
            cognitive_code = DEFAULT_COGNITIVE_TYPE
    else:
        # Cold-start: derive from the agent's last selection
        agent_lookup = rl_registry.get_agent(req.session_id, category=req.category)
        last_lt      = agent_lookup.last_action_lt or "PAR"
        last_level   = agent_lookup.mastery_levels.get(last_lt, 1)
        cognitive_code = build_cognitive_code(last_level, last_lt)

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
            "learning_type":     rl_step["learning_type"],
            "mastery_level":     rl_step["mastery_level"],
            "mastery_label":     rl_step["mastery_label"],
            "mastery_score":     rl_step["mastery_score"],
            "performance":       rl_step["performance"],
            "engagement":        rl_step["engagement"],
            "reward":            rl_step["reward"],
            "t_expected":        rl_step["t_expected"],
            "mlr_refitted":      rl_step["mlr_refitted"],
            "next_cognitive":    next_cognitive,
            "phase":             agent.phase,
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
