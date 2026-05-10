"""
app/ai/question_generator.py
─────────────────────────────
Generates a weekly batch of 10 CT questions, all sharing ONE topic.

The topic is provided as a free-text description (definition, explanation,
examples) written by the admin/teacher. The AI uses ONLY this topic text
as its knowledge source — no external topics are mixed in.

Three question types per batch:
  mcq       — single correct answer from A/B/C/D
  multi     — multiple correct answers (checkboxes), 2-4 correct out of 4-5 options
  truefalse — True/False with justification

Batch composition (10 questions, 1 topic):
  4 × mcq
  3 × multi
  3 × truefalse
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests as _req

logger = logging.getLogger(__name__)

# ── Batch composition ─────────────────────────────────────────────────────
BATCH_COMPOSITION = [
    ("mcq",       4),
    ("multi",     3),
    ("truefalse", 3),
]
DIFFICULTIES = ["mudah", "sedang", "sulit"]


# ══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES  — all grounded in the provided topic_text
# ══════════════════════════════════════════════════════════════════════════

_MCQ_PROMPT = """\
Kamu adalah pembuat soal untuk mata kuliah Computational Thinking.

MATERI MINGGU INI:
{topic_text}

Buat SATU soal pilihan ganda dengan SATU jawaban benar berdasarkan materi di atas.
Tingkat kesulitan: {difficulty}
Bahasa: Indonesia
Soal harus menguji pemahaman konsep dari materi, bukan hafalan.

Balas HANYA dengan JSON valid ini, tidak ada teks lain:
{{
  "question": "teks pertanyaan",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "A",
  "explanation": "penjelasan singkat mengapa jawaban ini benar (2-3 kalimat)"
}}"""

_MULTI_PROMPT = """\
Kamu adalah pembuat soal untuk mata kuliah Computational Thinking.

MATERI MINGGU INI:
{topic_text}

Buat SATU soal pilihan ganda dengan LEBIH DARI SATU jawaban benar berdasarkan materi di atas.
Tingkat kesulitan: {difficulty}
Bahasa: Indonesia
Sediakan 5 opsi (A-E), dengan 2-3 jawaban yang benar.
Soal harus jelas bahwa mahasiswa perlu memilih SEMUA jawaban yang benar.

Balas HANYA dengan JSON valid ini, tidak ada teks lain:
{{
  "question": "teks pertanyaan (tambahkan: 'Pilih SEMUA jawaban yang benar.')",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}},
  "correct_answers": ["A", "C"],
  "explanation": "penjelasan mengapa opsi-opsi tersebut benar dan yang lain salah (2-3 kalimat)"
}}"""

_TF_PROMPT = """\
Kamu adalah pembuat soal untuk mata kuliah Computational Thinking.

MATERI MINGGU INI:
{topic_text}

Buat SATU pernyataan Benar/Salah berdasarkan materi di atas.
Tingkat kesulitan: {difficulty}
Bahasa: Indonesia
Pernyataan harus spesifik dan dapat diverifikasi dari materi yang diberikan.
Jangan buat pernyataan yang ambigu.

Balas HANYA dengan JSON valid ini, tidak ada teks lain:
{{
  "question": "pernyataan yang harus dinilai Benar atau Salah",
  "correct_answer": "Benar",
  "explanation": "penjelasan mengapa pernyataan ini benar atau salah berdasarkan materi (2-3 kalimat)"
}}"""


# ══════════════════════════════════════════════════════════════════════════
# LLM CALLER
# ══════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str) -> Optional[str]:
    try:
        from app.services.llm import query_llm
        return query_llm(prompt)
    except Exception:
        pass
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
        logger.error("LLM call failed: %s", exc)
        return None


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# ══════════════════════════════════════════════════════════════════════════
# SINGLE QUESTION GENERATORS
# ══════════════════════════════════════════════════════════════════════════

def _base(q_type: str, topic_name: str, difficulty: str, week_id: str) -> Dict:
    return {
        "id":         str(uuid.uuid4()),
        "type":       q_type,
        "topic":      topic_name,
        "difficulty": difficulty,
        "week_id":    week_id,
        "status":     "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _gen_mcq(topic_text: str, topic_name: str, difficulty: str, week_id: str) -> Optional[Dict]:
    prompt = _MCQ_PROMPT.format(topic_text=topic_text[:2000], difficulty=difficulty)
    for _ in range(3):
        data = _extract_json(_call_llm(prompt) or "")
        if not data:
            continue
        if not all(k in data for k in ["question", "options", "correct_answer", "explanation"]):
            continue
        opts = data["options"]
        if not isinstance(opts, dict) or len(opts) < 2:
            continue
        correct = str(data["correct_answer"]).strip().upper()
        if correct not in opts:
            correct = list(opts.keys())[0]
        q = _base("mcq", topic_name, difficulty, week_id)
        q.update({
            "question":       data["question"].strip(),
            "options":        {k: str(v) for k, v in opts.items()},
            "correct_answer": correct,
            "explanation":    data.get("explanation", "").strip(),
        })
        return q
    return None


def _gen_multi(topic_text: str, topic_name: str, difficulty: str, week_id: str) -> Optional[Dict]:
    prompt = _MULTI_PROMPT.format(topic_text=topic_text[:2000], difficulty=difficulty)
    for _ in range(3):
        data = _extract_json(_call_llm(prompt) or "")
        if not data:
            continue
        if not all(k in data for k in ["question", "options", "correct_answers", "explanation"]):
            continue
        opts = data["options"]
        if not isinstance(opts, dict) or len(opts) < 3:
            continue
        corrects = [str(c).strip().upper() for c in data["correct_answers"]]
        corrects = [c for c in corrects if c in opts]
        if not corrects:
            corrects = [list(opts.keys())[0]]
        q = _base("multi", topic_name, difficulty, week_id)
        q.update({
            "question":        data["question"].strip(),
            "options":         {k: str(v) for k, v in opts.items()},
            "correct_answers": corrects,   # list, e.g. ["A","C"]
            "correct_answer":  corrects,   # alias for unified handling
            "explanation":     data.get("explanation", "").strip(),
        })
        return q
    return None


def _gen_truefalse(topic_text: str, topic_name: str, difficulty: str, week_id: str) -> Optional[Dict]:
    prompt = _TF_PROMPT.format(topic_text=topic_text[:2000], difficulty=difficulty)
    for _ in range(3):
        data = _extract_json(_call_llm(prompt) or "")
        if not data:
            continue
        if not all(k in data for k in ["question", "correct_answer", "explanation"]):
            continue
        answer = str(data["correct_answer"]).strip().lower()
        # Normalise to "Benar" or "Salah"
        if answer in ("true", "benar", "ya", "yes", "1"):
            answer = "Benar"
        else:
            answer = "Salah"
        q = _base("truefalse", topic_name, difficulty, week_id)
        q.update({
            "question":       data["question"].strip(),
            "options":        {"Benar": "Benar", "Salah": "Salah"},
            "correct_answer": answer,
            "explanation":    data.get("explanation", "").strip(),
        })
        return q
    return None


_GENERATORS = {
    "mcq":       _gen_mcq,
    "multi":     _gen_multi,
    "truefalse": _gen_truefalse,
}


# ══════════════════════════════════════════════════════════════════════════
# BATCH GENERATOR — single topic per session
# ══════════════════════════════════════════════════════════════════════════

def get_week_id(dt=None) -> str:
    from datetime import datetime, timezone
    d = dt or datetime.now(timezone.utc)
    return f"{d.year}-W{d.isocalendar()[1]:02d}"


def generate_weekly_batch(
    topic_name: str,
    topic_text: str,
    week_id: Optional[str] = None,
    composition: Optional[List] = None,
) -> List[Dict]:
    """
    Generate a weekly batch of questions all based on ONE topic.

    Args:
        topic_name:  Short label, e.g. "Abstraksi"
        topic_text:  Full description the admin provides — definition,
                     explanation, examples. This is the SOLE reference.
        week_id:     Override ISO week string. Default: current week.
        composition: List of (type, count) tuples. Default: BATCH_COMPOSITION.

    Returns:
        List of question dicts with status='pending'.
    """
    wid   = week_id or get_week_id()
    comp  = composition or BATCH_COMPOSITION
    total = sum(c for _, c in comp)

    # Build difficulty list — balanced across total questions
    import random
    diffs = (DIFFICULTIES * (total // len(DIFFICULTIES) + 1))[:total]
    random.shuffle(diffs)

    questions: List[Dict] = []
    diff_idx  = 0

    logger.info("Generating %d questions | topic='%s' | week=%s", total, topic_name, wid)

    for q_type, count in comp:
        gen_fn = _GENERATORS[q_type]
        for _ in range(count):
            diff = diffs[diff_idx % len(diffs)]
            diff_idx += 1
            q = gen_fn(topic_text, topic_name, diff, wid)
            if q:
                questions.append(q)
                logger.info("  ✓ [%s] %s | diff=%s", q_type, topic_name, diff)
            else:
                logger.error("  ✗ Failed to generate %s question", q_type)

    logger.info("Batch done: %d/%d generated | week=%s", len(questions), total, wid)
    return questions