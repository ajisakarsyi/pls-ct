"""
app/api/routes/tutor.py
────────────────────────
/chat  and  /evaluate  endpoints.

Cold-start by default: the student does NOT choose a LT.
The RL agent selects the LT via epsilon-greedy from the very first question.

[FIX v2 — Intent Intercept Layer on /evaluate]
  Problem: When a student had an active question and sent a message that was
  NOT an answer (e.g., "berikan 2 soal lagi tentang queue dan stack" or
  "kok dikasih soalnya deret?"), the /evaluate route would immediately pass
  the message to evaluate_student_answer(), grading it as a wrong answer.

  Fix: Before calling evaluate_student_answer(), /evaluate now calls
  classify_intent() from app.services.tutor. Based on the returned intent:

    A → student is answering       → evaluate_student_answer() as before
    B → student requests new content → clear active question, call generate_reply()
    C → student is complaining       → acknowledge, call generate_reply() with
                                       recovered topic for a corrected question
    D → student asks something new   → call generate_reply() directly

  The RL agent is NOT updated for intents B/C/D because no answer was given.
  This preserves full backward compatibility: genuine answers always get
  intent=A and see zero behavior change.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.core.cognitive import is_valid, DEFAULT_COGNITIVE_TYPE
from app.models.schemas import ChatRequest, ChatResponse, EvalRequest, EvalResponse
from app.services.tutor import (
    generate_reply,
    evaluate_student_answer,
    classify_intent,       # [FIX v2]
    get_session_topic,     # [FIX v2]
)
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
        "phase":                 rl_phase,
        "seeding_lt":            agent.seeding_lt,
        "seeding_sessions":      agent.seeding_questions,
        "seeding_remaining":     agent.seeding_remaining,
        "seeding_progress_pct":  round(agent.seeding_progress_pct, 3),
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
    summary="Evaluasi jawaban mahasiswa — RL Agent diperbarui berdasarkan hasil",
)
def evaluate(req: EvalRequest) -> EvalResponse:
    """
    Evaluate a student's answer and update the RL agent.

    [FIX v2] Before evaluating, classify the student's intent:
      - A (genuine answer)    → evaluate normally, update RL.
      - B (request new content) / C (complaint) / D (new question)
        → redirect to generate_reply(), RL agent NOT updated.

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
        agent_lookup   = rl_registry.get_agent(req.session_id, category=req.category)
        last_lt        = agent_lookup.last_action_lt or "PAR"
        last_level     = agent_lookup.mastery_levels.get(last_lt, 1)
        cognitive_code = build_cognitive_code(last_level, last_lt)

    parsed = parse_cognitive(cognitive_code)
    mastery_level, lt = parsed if parsed else (1, "PAR")

    # ── [FIX v2] Intent intercept ─────────────────────────────────────────
    # Only classify when there is an active question; otherwise skip.
    if req.active_question and req.active_question.strip():
        intent = classify_intent(
            message=req.answer.strip(),
            active_question=req.active_question.strip(),
        )
    else:
        intent = "A"

    # ── [FIX v2] Non-answer redirect ──────────────────────────────────────
    if intent != "A":
        # Student is NOT answering — redirect to generate_reply().
        # RL agent is NOT updated (no reward signal for non-answers).

        if intent == "C":
            # Complaint about off-topic question: recover session topic and
            # generate a corrected question on the actual discussion topic.
            active_topic = get_session_topic(req.session_id)
            redirect_message = (
                f"Baik, saya akan memberikan pertanyaan yang lebih sesuai "
                f"dengan topik yang sedang kita bahas: {active_topic}. "
                f"Pesan asli mahasiswa: {req.answer}"
            ) if active_topic else req.answer
        else:
            # B or D: treat student's message as a new chat input directly.
            redirect_message = req.answer

        chat_result = generate_reply(
            message        = redirect_message,
            cognitive_code = cognitive_code,
            session_id     = req.session_id,
            rl_selected    = None,
            rl_phase       = None,
        )

        agent = rl_registry.get_agent(req.session_id, category=req.category)

        _intent_labels = {
            "B": "Permintaan soal baru diterima.",
            "C": "Masukan kamu diterima — pertanyaan diperbarui.",
            "D": "Pertanyaan baru kamu diterima.",
        }
        feedback_prefix = _intent_labels.get(intent, "")

        return EvalResponse(
            is_correct        = False,
            feedback          = f"{feedback_prefix}\n\n{chat_result['reply']}",
            hint_level        = "Navigasi",
            followup_question = chat_result["followup_question"],
            cognitive         = cognitive_code,
            session_id        = req.session_id,
            rl                = None,
            lt_change         = None,
            total_lt_changes  = len(agent.recommendation_history),
        )

    # ── 1. Evaluate answer (intent == "A") ───────────────────────────────
    eval_result = evaluate_student_answer(
        answer          = req.answer.strip(),
        correct_answer  = req.correct_answer,
        active_question = req.active_question,
        wrong_count     = req.wrong_count,
        cognitive_code  = cognitive_code,
        session_id      = req.session_id,
    )

    # ── 2. Record in RL agent ─────────────────────────────────────────────
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

    # ── 3. Build RL response block ────────────────────────────────────────
    agent    = rl_registry.get_agent(req.session_id, category=req.category)
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
