"""
app/ai/question_generator.py
─────────────────────────────
AI layer that generates Computational Thinking questions for the weekly
Asah Otak quiz bank.

Produces a mix of MCQ (pilihan ganda) and open-ended questions.
Each question includes:
  - question text
  - type: "mcq" | "open"
  - options (MCQ only): {"A": ..., "B": ..., "C": ..., "D": ...}
  - correct_answer: the correct option key (MCQ) or reference answer (open)
    → stored server-side only, NEVER sent to end-users
  - explanation (pembahasan): why the answer is correct
  - difficulty: "mudah" | "sedang" | "sulit"
  - topic: CT topic tag
  - week_id: ISO week string e.g. "2026-W12"

Uses ChatAnywhere if available, otherwise falls back to local Ollama.
LLM response is parsed as strict JSON — malformed responses are retried.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests as _req

logger = logging.getLogger(__name__)

# ── CT topics pool — rotated across weeks ─────────────────────────────────
CT_TOPICS = [
    "algoritma dan pseudocode",
    "dekomposisi masalah",
    "abstraksi",
    "pengenalan pola (pattern recognition)",
    "bubble sort",
    "selection sort",
    "insertion sort",
    "merge sort",
    "quick sort",
    "binary search",
    "linear search",
    "rekursi",
    "iterasi dan perulangan",
    "kompleksitas waktu Big-O",
    "kompleksitas ruang",
    "stack dan queue",
    "linked list",
    "array dan list",
    "hash table dan dictionary",
    "binary tree dan traversal",
    "graph dan BFS/DFS",
    "dynamic programming",
    "greedy algorithm",
    "paradigma pemrograman",
    "abstract data type (ADT)",
]

DIFFICULTIES = ["mudah", "sedang", "sulit"]

# Target mix per batch of 10: 6 MCQ + 4 open-ended
MCQ_COUNT  = 6
OPEN_COUNT = 4


# ══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════

_MCQ_PROMPT = """\
Kamu adalah pembuat soal Computational Thinking untuk mahasiswa universitas Indonesia.

Buat SATU soal pilihan ganda (MCQ) dengan ketentuan berikut:
- Topik: {topic}
- Tingkat kesulitan: {difficulty}
- Bahasa: Indonesia
- Soal harus menguji pemahaman konsep, bukan sekadar hafalan
- Opsi jawaban harus masuk akal dan tidak trivial

Balas HANYA dengan JSON valid berikut, tidak ada teks lain:
{{
  "question": "teks pertanyaan di sini",
  "options": {{
    "A": "opsi A",
    "B": "opsi B",
    "C": "opsi C",
    "D": "opsi D"
  }},
  "correct_answer": "A",
  "explanation": "penjelasan mengapa jawaban ini benar dan mengapa opsi lain salah (2-3 kalimat)"
}}"""

_OPEN_PROMPT = """\
Kamu adalah pembuat soal Computational Thinking untuk mahasiswa universitas Indonesia.

Buat SATU soal uraian (open-ended) dengan ketentuan berikut:
- Topik: {topic}
- Tingkat kesulitan: {difficulty}
- Bahasa: Indonesia
- Soal harus berbasis skenario atau penerapan konkret
- Jawaban referensi harus jelas dan dapat dinilai secara objektif

Balas HANYA dengan JSON valid berikut, tidak ada teks lain:
{{
  "question": "teks pertanyaan di sini (bisa berupa skenario 2-3 kalimat + pertanyaan spesifik)",
  "correct_answer": "jawaban referensi lengkap yang digunakan untuk menilai jawaban mahasiswa",
  "explanation": "pembahasan tambahan: konsep kunci yang harus ada dalam jawaban yang benar (2-3 kalimat)"
}}"""


# ══════════════════════════════════════════════════════════════════════════
# LLM CALLER — uses same provider logic as the main app
# ══════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str) -> Optional[str]:
    """
    Call LLM for question generation.
    Tries app's query_llm first (respects provider switching),
    falls back to direct Ollama call if import fails.
    """
    try:
        from app.services.llm import query_llm
        return query_llm(prompt)
    except Exception:
        pass

    # Direct Ollama fallback
    try:
        from app.core.config import get_settings
        s = get_settings()
        resp = _req.post(
            f"{s.ollama_base_url}/api/generate",
            json={"model": s.ollama_chat_model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.error("LLM call failed in question_generator: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> Optional[Dict]:
    """
    Extract the first valid JSON object from LLM output.
    Handles cases where LLM wraps JSON in markdown code blocks or adds preamble.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Try to fix common LLM JSON errors: trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# ══════════════════════════════════════════════════════════════════════════
# SINGLE QUESTION GENERATORS
# ══════════════════════════════════════════════════════════════════════════

def _generate_mcq(topic: str, difficulty: str, week_id: str, max_retries: int = 3) -> Optional[Dict]:
    """Generate one MCQ question. Retries on malformed JSON."""
    prompt = _MCQ_PROMPT.format(topic=topic, difficulty=difficulty)

    for attempt in range(max_retries):
        raw = _call_llm(prompt)
        data = _extract_json(raw or "")

        if not data:
            logger.warning("MCQ attempt %d: could not extract JSON for topic=%s", attempt + 1, topic)
            continue

        # Validate required fields
        if not all(k in data for k in ["question", "options", "correct_answer", "explanation"]):
            logger.warning("MCQ attempt %d: missing fields: %s", attempt + 1, list(data.keys()))
            continue

        options = data.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            logger.warning("MCQ attempt %d: invalid options: %s", attempt + 1, options)
            continue

        correct = str(data["correct_answer"]).strip().upper()
        if correct not in options:
            # Try to recover if model gave full answer text instead of key
            for k, v in options.items():
                if str(v).strip().lower() == correct.lower():
                    correct = k
                    break
            else:
                correct = list(options.keys())[0]  # fallback to first option

        return {
            "id":             str(uuid.uuid4()),
            "type":           "mcq",
            "question":       data["question"].strip(),
            "options":        {k: str(v) for k, v in options.items()},
            "correct_answer": correct,
            "explanation":    data.get("explanation", "").strip(),
            "topic":          topic,
            "difficulty":     difficulty,
            "week_id":        week_id,
            "status":         "pending",
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

    logger.error("Failed to generate MCQ for topic=%s after %d attempts", topic, max_retries)
    return None


def _generate_open(topic: str, difficulty: str, week_id: str, max_retries: int = 3) -> Optional[Dict]:
    """Generate one open-ended question."""
    prompt = _OPEN_PROMPT.format(topic=topic, difficulty=difficulty)

    for attempt in range(max_retries):
        raw = _call_llm(prompt)
        data = _extract_json(raw or "")

        if not data:
            logger.warning("Open attempt %d: could not extract JSON for topic=%s", attempt + 1, topic)
            continue

        if not all(k in data for k in ["question", "correct_answer", "explanation"]):
            logger.warning("Open attempt %d: missing fields", attempt + 1)
            continue

        return {
            "id":             str(uuid.uuid4()),
            "type":           "open",
            "question":       data["question"].strip(),
            "options":        None,
            "correct_answer": data["correct_answer"].strip(),
            "explanation":    data.get("explanation", "").strip(),
            "topic":          topic,
            "difficulty":     difficulty,
            "week_id":        week_id,
            "status":         "pending",
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

    logger.error("Failed to generate open question for topic=%s after %d attempts", topic, max_retries)
    return None


# ══════════════════════════════════════════════════════════════════════════
# BATCH GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def get_week_id(dt: Optional[datetime] = None) -> str:
    """Return ISO week string like '2026-W12'."""
    d = dt or datetime.now(timezone.utc)
    return f"{d.year}-W{d.isocalendar()[1]:02d}"


def generate_weekly_batch(
    week_id: Optional[str] = None,
    n_mcq: int = MCQ_COUNT,
    n_open: int = OPEN_COUNT,
    topics: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Generate a full weekly batch of CT questions.

    Returns list of question dicts (all status='pending').
    The caller is responsible for storing them.

    Args:
        week_id:  override week string, default = current week
        n_mcq:    number of MCQ questions (default 6)
        n_open:   number of open-ended questions (default 4)
        topics:   override topic list; if None, picks from CT_TOPICS pool
    """
    wid    = week_id or get_week_id()
    pool   = topics or CT_TOPICS
    total  = n_mcq + n_open

    # Select topics — spread across difficulties
    import random
    selected_topics = random.sample(pool, min(total, len(pool)))
    if len(selected_topics) < total:
        # Repeat pool if needed
        selected_topics *= (total // len(selected_topics) + 1)
    selected_topics = selected_topics[:total]

    # Assign difficulties — roughly balanced
    difficulty_cycle = (
        DIFFICULTIES * (total // len(DIFFICULTIES) + 1)
    )[:total]
    random.shuffle(difficulty_cycle)

    questions = []
    errors    = 0

    logger.info("Generating weekly batch: week=%s MCQ=%d open=%d", wid, n_mcq, n_open)

    # Generate MCQs
    for i in range(n_mcq):
        q = _generate_mcq(selected_topics[i], difficulty_cycle[i], wid)
        if q:
            questions.append(q)
            logger.info("  [%d/%d] MCQ generated: topic=%s difficulty=%s",
                        len(questions), total, q["topic"], q["difficulty"])
        else:
            errors += 1

    # Generate open-ended
    for i in range(n_open):
        idx = n_mcq + i
        q = _generate_open(selected_topics[idx], difficulty_cycle[idx], wid)
        if q:
            questions.append(q)
            logger.info("  [%d/%d] Open generated: topic=%s difficulty=%s",
                        len(questions), total, q["topic"], q["difficulty"])
        else:
            errors += 1

    logger.info(
        "Batch complete: %d generated, %d failed. week=%s",
        len(questions), errors, wid,
    )
    return questions
