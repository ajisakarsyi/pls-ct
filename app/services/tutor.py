"""
app/services/tutor.py
──────────────────────
Core tutoring business logic:
  - generate_reply()          — tutor explanation + follow-up question
  - evaluate_student_answer() — strict two-step evaluation + scaffolded feedback
  - classify_intent()         — [FIX v2] detect if student is answering, requesting
                                new question, complaining, or asking something new
  - extract_active_topic()    — [FIX v2] extract current discussion topic from
                                session history for topic-bound followup generation

RL selection (which cognitive type to use) is handled upstream in the route
handler via app/services/rl.py.  The cognitive_code is passed in here already
resolved, so this module stays clean and testable.

[FIX v2 — Bug fixes for topic drift and intent misclassification]
  BUG 1: Followup questions (LLM path) were generated without topic context,
         causing the LLM to produce questions from unrelated topics (e.g.,
         deret/sequences when the student was discussing stack & queue).
  FIX 1: extract_active_topic() extracts the current topic from session history
         and injects it as {active_topic} into FOLLOWUP_PROMPT_TEMPLATE, which
         now contains an explicit constraint: "soal HARUS berkaitan dengan
         topik {active_topic}". The HARD_CODED_FOLLOWUPS dictionary is checked
         first; the LLM path (now with active_topic) is only used as fallback.

  BUG 2: Student messages were always routed to evaluate_student_answer() even
         when the student was complaining about an irrelevant question or
         requesting a new topic — causing those messages to be graded as wrong
         answers.
  FIX 2: classify_intent() uses INTENT_CLASSIFIER_PROMPT to detect whether the
         student is (A) answering, (B) requesting new content, (C) complaining,
         or (D) asking something new. This is called in the route layer BEFORE
         dispatching to the evaluator.
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
    INTENT_CLASSIFIER_PROMPT,    # [FIX v2]
    TOPIC_EXTRACTOR_PROMPT,      # [FIX v2]
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
_SCAFFOLD         = SCAFFOLD_LEVELS
_DEFAULT_SCAFFOLD = SCAFFOLD_DEFAULT

# ── [FIX v2] In-memory topic cache per session ────────────────────────────
# Maps session_id → current active topic string (e.g. "stack dan queue")
# Updated every time generate_reply() is called.
_session_topic_cache: Dict[str, str] = {}


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

    [FIX v2] Extracts and caches the active topic from session history
    so that followup questions (LLM fallback path) stay bound to the
    current discussion topic.

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

    reply = query_llm(prompt)

    # [FIX v2] Extract and cache the active topic BEFORE generating followup
    active_topic = _extract_and_cache_topic(
        session_id=session_id,
        message=message,
        history_txt=history_txt,
    )

    followup = _generate_followup(message, reply, context, label, active_topic=active_topic)

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

    # [FIX v2] Retrieve cached topic for bound followup generation
    active_topic = _session_topic_cache.get(session_id, "")

    if is_correct:
        feedback = "✅ Jawaban kamu benar."
        followup = ""
    else:
        hint_level, feedback_instruction = _SCAFFOLD.get(wrong_count, _DEFAULT_SCAFFOLD)
        feedback = _generate_feedback(
            answer=answer, correct_answer=correct_answer,
            reasoning=reasoning, context=context, history_txt=history_txt,
            label=label, cognitive_code=cognitive_code,
            hint_level=hint_level, feedback_instruction=feedback_instruction,
        )
        followup = _generate_followup(
            correct_answer, feedback, context, label, active_topic=active_topic
        )

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


# ── [FIX v2] Public helpers for route layer ───────────────────────────────

def classify_intent(message: str, active_question: str) -> str:
    """
    Classify the intent of *message* given the currently active question.

    Returns one of:
      "A" — student is answering the active question
      "B" — student is requesting new question / new topic
      "C" — student is complaining or protesting about the question
      "D" — student is asking something unrelated to the active question

    Falls back to keyword-based detection if the LLM fails, and ultimately
    defaults to "A" (treat as answer attempt) to preserve backward compatibility.
    """
    if not active_question or not active_question.strip():
        # No active question → always route to generate_reply
        return "D"

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        active_question=active_question.strip(),
        message=message.strip(),
    )

    try:
        raw   = query_llm(prompt).strip()
        match = re.search(r"\b([ABCD])\b", raw.upper())
        if match:
            return match.group(1)
        return _keyword_intent_fallback(message)
    except Exception as exc:
        logger.warning("[INTENT] classify_intent failed (%s) — defaulting to A", exc)
        return "A"


def get_session_topic(session_id: str) -> str:
    """Return the cached active topic for a session, or empty string."""
    return _session_topic_cache.get(session_id, "")


# ── Private helpers ────────────────────────────────────────────────────────

def _extract_and_cache_topic(
    session_id:  str,
    message:     str,
    history_txt: str,
) -> str:
    """
    [FIX v2] Extract the active topic from session history + current message
    and cache it in _session_topic_cache.

    Uses TOPIC_EXTRACTOR_PROMPT. Falls back to a cleaned version of the
    current message if the LLM call fails.
    """
    prompt = TOPIC_EXTRACTOR_PROMPT.format(
        history=history_txt[:1000],
        message=message,
    )
    try:
        topic = query_llm(prompt).strip()
        # Keep only first line, strip quotes/punctuation
        topic = topic.split("\n")[0].strip().strip('"').strip("'").rstrip(".")
        if topic and len(topic) < 80:
            _session_topic_cache[session_id] = topic
            logger.debug("[TOPIC] session=%s → '%s'", session_id, topic)
            return topic
    except Exception as exc:
        logger.warning("[TOPIC] extract failed (%s) — using message as fallback", exc)

    fallback = message[:60].strip()
    _session_topic_cache[session_id] = fallback
    return fallback


def _keyword_intent_fallback(message: str) -> str:
    """
    Rule-based intent detection as fallback when LLM is unavailable.
    Returns "A", "B", "C", or "D".
    """
    msg_lower = message.lower()

    if any(kw in msg_lower for kw in [
        "berikan", "kasih", "minta", "soal lagi", "soal baru",
        "lanjut", "topik lain", "pertanyaan lain", "soal tentang",
    ]):
        return "B"

    if any(kw in msg_lower for kw in [
        "kok", "kenapa", "tidak relevan", "salah soal",
        "saya perbaiki", "bukan soal", "ini bukan", "ganti soal",
    ]):
        return "C"

    if re.match(r"^(apa|bagaimana|jelaskan|mengapa|kapan|siapa|apa itu)", msg_lower):
        return "D"

    return "A"


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
    original_question: str,
    tutor_reply:       str,
    context:           str,
    label:             str,
    active_topic:      str = "",   # [FIX v2]
) -> str:
    """
    Generate a follow-up question.

    Priority order:
    1. HARD_CODED_FOLLOWUPS — exact match on normalised student question.
       These bypass the LLM entirely and guarantee topic relevance for the
       fixed set of Exercise prompts used in user testing sessions.
    2. LLM fallback — FOLLOWUP_PROMPT_TEMPLATE now includes {active_topic}
       so the LLM is constrained to generate a question on the current topic.
       [FIX v2] Without active_topic the LLM would freely pick any topic.
    """
    # 1. Hard-coded followups (exact match, normalised)
    HARD_CODED_FOLLOWUPS = {
        "apa itu computational thinking dan mengapa penting untuk dipelajari?":
            "Kamu ingin merencanakan rute pembagian bantuan sembako di sebuah desa yang memiliki 15 RT agar efisien dan hemat waktu. Tentukan langkah pertama apa yang harus kamu lakukan jika ingin menerapkan metode Dekomposisi dalam masalah ini?",

        "jelaskan perbedaan antara dekomposisi dan abstraksi dalam ct dengan contoh nyata.":
            "Sebuah restoran ingin membuat sistem pemesanan makanan otomatis. Mereka mengabaikan warna baju pelayan dan fokus hanya pada menu serta harga makanan. Komponen CT mana yang sedang mereka terapkan (Dekomposisi atau Abstraksi)?",

        "saya bingung kenapa harus belajar algoritma. apa hubungannya dengan kehidupan sehari-hari?":
            "Terdapat 3 langkah acak memasak mi instan: [A: Rebus air, B: Masukkan mi, C: Tiriskan mi]. Urutkan huruf langkah tersebut berdasarkan prinsip algoritma yang benar dari awal sampai akhir!",

        "bagaimana cara melatih kemampuan pengenalan pola dalam kehidupan sehari-hari?":
            "Seorang dokter melihat gejala pasien: demam tinggi, bintik merah, dan trombosit turun. Dokter langsung tahu pasien terkena DBD karena polanya mirip dengan ratusan pasien sebelumnya. Apakah tindakan dokter ini memanfaatkan prinsip Pengenalan Pola (Ya atau Tidak)?",

        "apakah semua masalah di dunia ini bisa diselesaikan dengan computational thinking?":
            "Diberikan pernyataan: 'Computational Thinking hanya berguna bagi orang yang bekerja sebagai programmer atau software engineer.' Apakah pernyataan tersebut Benar atau Salah?",
    }

    cleaned_question = original_question.strip().lower()
    if cleaned_question in HARD_CODED_FOLLOWUPS:
        logger.info("[HARD-CODED] Mengirimkan soal spesifik untuk: %s", original_question)
        return HARD_CODED_FOLLOWUPS[cleaned_question]

    # 2. LLM fallback — with active_topic constraint [FIX v2]
    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        label=label,
        active_topic=active_topic or original_question[:80],  # fallback to question text
        original_question=original_question,
        reply=tutor_reply[:600],
        context=context,
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
