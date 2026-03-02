"""
app/services/tutor.py
──────────────────────
Core tutoring business logic:
  - generate_reply()     — tutor explanation + follow-up question
  - evaluate_answer()    — strict two-step evaluation + scaffolded feedback
"""

import logging
import re
from typing import Dict, Tuple

from app.core.cognitive import cognitive_label
from app.core.prompts import (
    CHAT_PROMPT_TEMPLATE,
    CHAT_CODE_PROMPT_TEMPLATE,
    FOLLOWUP_PROMPT_TEMPLATE,
    EVALUATE_PROMPT_TEMPLATE,
    FEEDBACK_PROMPT_TEMPLATE,
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

# Scaffolding levels indexed by wrong_count
_SCAFFOLD: Dict[int, Tuple[str, str]] = {
    0: (
        "Evaluasi Awal",
        "Tunjukkan bagian yang kurang tepat secara umum (1-2 kalimat). "
        "Jangan jelaskan terlalu banyak — cukup arahkan.",
    ),
    1: (
        "Petunjuk Terarah",
        "Berikan petunjuk spesifik tentang konsep yang salah (2-3 kalimat). "
        "Sebutkan aspek mana yang perlu diperbaiki tanpa memberi jawaban langsung.",
    ),
    2: (
        "Dukungan Remedial",
        "Uraikan konsep yang salah secara bertahap dalam 3 poin singkat. "
        "Boleh memberi contoh kecil untuk memperjelas.",
    ),
}
_DEFAULT_SCAFFOLD = (
    "Panduan Langkah-demi-Langkah",
    "Jelaskan ulang konsep secara lengkap dengan analogi sederhana, maksimal 3 paragraf. "
    "Pastikan mahasiswa memahami di mana letak kesalahannya.",
)


def generate_reply(
    message: str,
    cognitive_code: str,
    session_id: str,
) -> Dict:
    """
    Generate a tutor explanation for *message* and a follow-up question.

    Returns a dict with keys: reply, followup_question, cognitive, session_id.
    """
    label = cognitive_label(cognitive_code)
    history = get_session(session_id)
    history_txt = format_history(history)

    chunks = retrieve(message, cognitive_code)
    context = chunks_to_context(chunks)

    # Choose prompt template based on whether the message looks like code
    template = CHAT_CODE_PROMPT_TEMPLATE if is_code_like(message) else CHAT_PROMPT_TEMPLATE
    prompt = template.format(
        label=label,
        history=history_txt,
        code=cognitive_code,
        context=context,
        message=message,
    )

    reply = query_llm(prompt)

    # Generate a separate technical follow-up question
    followup = _generate_followup(message, reply, context, label)

    # Persist to session + logs
    history.add_user_message(message)
    history.add_ai_message(reply)
    log_interaction(session_id, cognitive_code, message, reply, followup)

    return {
        "reply": reply,
        "followup_question": followup,
        "cognitive": cognitive_code,
        "session_id": session_id,
    }


def evaluate_student_answer(
    answer: str,
    correct_answer: str,
    active_question: str,
    wrong_count: int,
    cognitive_code: str,
    session_id: str,
) -> Dict:
    """
    Strictly evaluate *answer* and return feedback.

    Returns a dict with keys:
      is_correct, feedback, hint_level, followup_question, cognitive, session_id.
    """
    label = cognitive_label(cognitive_code)
    history = get_session(session_id)
    history_txt = format_history(history)

    chunks = retrieve(
        f"Kunci: {correct_answer}. Jawaban mahasiswa: {answer}",
        cognitive_code,
    )
    context = chunks_to_context(chunks)

    # Two-step strict evaluation
    is_correct, reasoning = _strict_evaluate(
        answer=answer,
        correct_answer=correct_answer,
        active_question=active_question,
        context=context,
        history_txt=history_txt,
        label=label,
        cognitive_code=cognitive_code,
    )

    if is_correct:
        feedback = "✅ Jawaban kamu benar."
        followup = ""
    else:
        hint_level, feedback_instruction = _SCAFFOLD.get(wrong_count, _DEFAULT_SCAFFOLD)
        feedback = _generate_feedback(
            answer=answer,
            correct_answer=correct_answer,
            reasoning=reasoning,
            context=context,
            history_txt=history_txt,
            label=label,
            cognitive_code=cognitive_code,
            hint_level=hint_level,
            feedback_instruction=feedback_instruction,
        )
        followup = _generate_followup(correct_answer, feedback, context, label)

    hint_level = _SCAFFOLD.get(wrong_count, _DEFAULT_SCAFFOLD)[0]

    history.add_user_message(f"[JAWABAN MAHASISWA] {answer}")
    history.add_ai_message(f"[UMPAN BALIK TUTOR] {feedback}")

    return {
        "is_correct": is_correct,
        "feedback": feedback,
        "hint_level": hint_level,
        "followup_question": followup,
        "cognitive": cognitive_code,
        "session_id": session_id,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _strict_evaluate(
    answer: str,
    correct_answer: str,
    active_question: str,
    context: str,
    history_txt: str,
    label: str,
    cognitive_code: str,
) -> Tuple[bool, str]:
    if active_question and active_question.strip():
        evaluation_scope = (
            f"PERTANYAAN YANG SEDANG DIJAWAB:\n{active_question}\n\n"
            f"KONTEKS MATERI (penjelasan tutor sebelumnya):\n{correct_answer[:800]}"
        )
        task_instruction = (
            "TUGAS PENILAIAN:\n"
            "1. Fokus pada pertanyaan yang sedang dijawab — BUKAN penjelasan tutor secara keseluruhan.\n"
            "2. Hitung atau verifikasi kebenaran jawaban mahasiswa terhadap pertanyaan tersebut.\n"
            "3. Untuk soal numerik/matematis: periksa apakah hasil akhirnya benar secara matematis.\n"
            "4. Untuk soal konseptual: periksa apakah jawaban mencakup poin utama yang ditanyakan.\n"
            "5. JANGAN menolak jawaban benar hanya karena singkat atau tidak menjelaskan proses."
        )
    else:
        evaluation_scope = (
            f"KUNCI / REFERENSI (penjelasan tutor):\n{correct_answer[:800]}"
        )
        task_instruction = (
            "TUGAS PENILAIAN:\n"
            "1. Bandingkan jawaban mahasiswa dengan penjelasan tutor secara konseptual.\n"
            "2. Jawaban BENAR jika mencakup konsep utama, meskipun dengan kata berbeda.\n"
            "3. Jawaban SALAH jika konsep utama hilang, keliru, atau tidak relevan.\n"
            "4. Jangan anggap benar hanya karena terdengar logis — harus sesuai kunci."
        )

    prompt = EVALUATE_PROMPT_TEMPLATE.format(
        label=label,
        code=cognitive_code,
        context=context,
        history=history_txt,
        evaluation_scope=evaluation_scope,
        answer=answer,
        task_instruction=task_instruction,
    )

    raw = query_llm(prompt)
    match = re.search(r"HASIL:\s*(BENAR|SALAH)", raw, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        reasoning = re.sub(r"HASIL:\s*(BENAR|SALAH)", "", raw, flags=re.IGNORECASE).strip()
    else:
        is_correct = False
        reasoning = raw.strip()

    return is_correct, reasoning


def _generate_followup(
    original_question: str,
    tutor_reply: str,
    context: str,
    label: str,
) -> str:
    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        label=label,
        reply=tutor_reply[:600],
        context=context,
    )
    result = query_llm(prompt).strip()
    lines = [ln.strip() for ln in result.split("\n") if ln.strip()]
    return lines[-1] if lines else result


def _generate_feedback(
    answer: str,
    correct_answer: str,
    reasoning: str,
    context: str,
    history_txt: str,
    label: str,
    cognitive_code: str,
    hint_level: str,
    feedback_instruction: str,
) -> str:
    prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        label=label,
        code=cognitive_code,
        context=context,
        history=history_txt,
        answer=answer,
        correct_answer=correct_answer,
        reasoning=reasoning,
        hint_level=hint_level,
        feedback_instruction=feedback_instruction,
    )
    return query_llm(prompt)
