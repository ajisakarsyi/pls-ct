"""
app/services/tutor.py
──────────────────────
Core tutoring business logic:
  - generate_reply()          — tutor explanation + follow-up question
  - evaluate_student_answer() — strict two-step evaluation + scaffolded feedback

RL selection (which cognitive type to use) is handled upstream in the route
handler via app/services/rl.py.  The cognitive_code is passed in here already
resolved, so this module stays clean and testable.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from app.core.cognitive import cognitive_label
from app.core.prompts import (
    CHAT_PROMPT_TEMPLATE,
    CHAT_CODE_PROMPT_TEMPLATE,
    CHECK_UNDERSTANDING_LEAD,
    FOLLOWUP_PROMPT_TEMPLATE,
    EVALUATE_PROMPT_WITH_QUESTION,
    EVALUATE_PROMPT_WITHOUT_QUESTION,
    FEEDBACK_PROMPT_TEMPLATE,
    SCAFFOLD_LEVELS,
    SCAFFOLD_DEFAULT,
)
from app.services.llm import query_llm
from app.services.rag import retrieve, chunks_to_context
from app.services.session import (
    get_session,
    format_history,
    log_interaction,
)
from app.utils.code_detector import is_code_like

logger = logging.getLogger(__name__)

# Scaffolding levels sourced from app.core.prompts (SCAFFOLD_LEVELS, SCAFFOLD_DEFAULT)
_SCAFFOLD        = SCAFFOLD_LEVELS
_DEFAULT_SCAFFOLD = SCAFFOLD_DEFAULT


def generate_reply(
    message:        str,
    cognitive_code: str,
    session_id:     str,
    rl_selected:    Optional[bool] = None,
    rl_phase:       Optional[str]  = None,
) -> Dict:
    """
    Generate a tutor explanation for *message* and a follow-up question.
    cognitive_code is already resolved (by RL or explicit request) before
    this function is called.

    Returns a dict with keys: reply, followup_question, cognitive, session_id.
    """
    label       = cognitive_label(cognitive_code)
    history     = get_session(session_id)
    history_txt = format_history(history)

    chunks  = retrieve(message, cognitive_code)
    context = chunks_to_context(chunks)

    template = CHAT_CODE_PROMPT_TEMPLATE if is_code_like(message) else CHAT_PROMPT_TEMPLATE
    prompt   = template.format(
        label=label, history=history_txt, code=cognitive_code,
        context=context, message=message,
        check_understanding_lead=CHECK_UNDERSTANDING_LEAD,
    )

    reply   = query_llm(prompt)
    followup = _generate_followup(message, reply, context, label)

    history.add_user_message(message)
    history.add_ai_message(reply)
    log_interaction(
        session_id        = session_id,
        cognitive         = cognitive_code,
        user_message      = message,
        reply             = reply,
        followup_question = followup,
        rl_selected       = rl_selected,
        rl_phase          = rl_phase,
    )

    return {
        "reply":             reply,
        "followup_question": followup,
        "cognitive":         cognitive_code,
        "session_id":        session_id,
    }


def evaluate_student_answer(
    answer:          str,
    correct_answer:  str,
    active_question: str,
    wrong_count:     int,
    cognitive_code:  str,
    session_id:      str,
) -> Dict:
    """
    Strictly evaluate *answer* and return scaffolded feedback.

    Returns a dict with keys:
      is_correct, feedback, hint_level, followup_question, cognitive, session_id.
    """
    label       = cognitive_label(cognitive_code)
    history     = get_session(session_id)
    history_txt = format_history(history)

    chunks  = retrieve(
        f"Kunci: {correct_answer}. Jawaban mahasiswa: {answer}",
        cognitive_code,
    )
    context = chunks_to_context(chunks)

    is_correct, reasoning = _strict_evaluate(
        answer=answer, correct_answer=correct_answer,
        active_question=active_question, context=context,
        history_txt=history_txt, label=label, cognitive_code=cognitive_code,
    )

    if is_correct:
        feedback  = "✅ Jawaban kamu benar."
        followup  = ""
    else:
        hint_level, feedback_instruction = _SCAFFOLD.get(wrong_count, _DEFAULT_SCAFFOLD)
        feedback  = _generate_feedback(
            answer=answer, correct_answer=correct_answer,
            reasoning=reasoning, context=context, history_txt=history_txt,
            label=label, cognitive_code=cognitive_code,
            hint_level=hint_level, feedback_instruction=feedback_instruction,
        )
        followup = _generate_followup(correct_answer, feedback, context, label)

    hint_level = _SCAFFOLD.get(wrong_count, _DEFAULT_SCAFFOLD)[0]

    history.add_user_message(f"[JAWABAN MAHASISWA] {answer}")
    history.add_ai_message(f"[UMPAN BALIK TUTOR] {feedback}")

    return {
        "is_correct":        is_correct,
        "feedback":          feedback,
        "hint_level":        hint_level,
        "followup_question": followup,
        "cognitive":         cognitive_code,
        "session_id":        session_id,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _strict_evaluate(
    answer: str, correct_answer: str, active_question: str,
    context: str, history_txt: str, label: str, cognitive_code: str,
) -> Tuple[bool, str]:
    if active_question and active_question.strip():
        prompt = EVALUATE_PROMPT_WITH_QUESTION.format(
            label=label, code=cognitive_code, context=context, history=history_txt,
            active_question=active_question,
            correct_answer=correct_answer[:800],
            answer=answer,
        )
    else:
        prompt = EVALUATE_PROMPT_WITHOUT_QUESTION.format(
            label=label, code=cognitive_code, context=context, history=history_txt,
            correct_answer=correct_answer[:800],
            answer=answer,
        )

    raw   = query_llm(prompt)
    match = re.search(r"HASIL:\s*(BENAR|SALAH)", raw, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        reasoning  = re.sub(r"HASIL:\s*(BENAR|SALAH)", "", raw, flags=re.IGNORECASE).strip()
    else:
        is_correct = False
        reasoning  = raw.strip()

    return is_correct, reasoning


def _generate_followup(
    original_question: str, tutor_reply: str, context: str, label: str,
) -> str:
    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        label=label, original_question=original_question, reply=tutor_reply[:600], context=context,
    )
    result = query_llm(prompt).strip()
    lines  = [ln.strip() for ln in result.split("\n") if ln.strip()]
    return lines[-1] if lines else result


def _generate_feedback(
    answer: str, correct_answer: str, reasoning: str, context: str,
    history_txt: str, label: str, cognitive_code: str,
    hint_level: str, feedback_instruction: str,
) -> str:
    prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        label=label, code=cognitive_code, context=context, history=history_txt,
        answer=answer, correct_answer=correct_answer, reasoning=reasoning,
        hint_level=hint_level, feedback_instruction=feedback_instruction,
    )
    return query_llm(prompt)
