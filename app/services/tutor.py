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
import os
import re
from typing import Dict, Optional, Tuple

from app.core import verbose as V
from app.core.cognitive import VALID_COGNITIVE_TYPES, cognitive_label
from app.core.config import get_settings
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
from app.core.rag_guard import NoRAGGuard
from app.services.live_metrics import (
    compute_live_metrics_a,
    compute_live_metrics_b,
)
from app.services.llm import query_llm, query_llm_ollama_raw
from app.services.rag import retrieve, chunks_to_context
from app.services.session import (
    get_session,
    format_history,
    log_interaction,
)
from app.utils.code_detector import is_code_like

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── Prompt buta Kondisi B — HARUS identik dengan Lampiran 4 skripsi dan
#    evaluation/runner.py::_chat_without_rag. Tanpa konteks, tanpa profil,
#    tanpa instruksi tutor, tanpa riwayat. ──────────────────────────────────
BLIND_PROMPT_TEMPLATE = (
    "Jawab pertanyaan berikut sebaik mungkin:\n\n"
    "{query}\n\n"
    "Berikan jawaban dalam Bahasa Indonesia."
)

# Scaffolding levels sourced from app.core.prompts (SCAFFOLD_LEVELS, SCAFFOLD_DEFAULT)
_SCAFFOLD        = SCAFFOLD_LEVELS
_DEFAULT_SCAFFOLD = SCAFFOLD_DEFAULT


def generate_reply(
    message:        str,
    cognitive_code: str,
    session_id:     str,
    rl_selected:    Optional[bool] = None,
    rl_phase:       Optional[str]  = None,
    mode:           str            = "A",
) -> Dict:
    """
    ═══════════════════════════════════════════════════════════════
    FUNGSI INTI TUTOR — generate_reply()   [REVISI PASCA-SIDANG]
    File: app/services/tutor.py
    ═══════════════════════════════════════════════════════════════

    Parameter *mode* (revisi item 1) memilih kondisi eksperimen skripsi:

      mode="A" (default) — LLM + RAG + profil kognitif + instruksi tutor.
        Alur: label profil → riwayat sesi → retrieve() Top-K → prompt
        template → query_llm() → follow-up. Chunk yang dipakai, skornya,
        topiknya, metrik live (Pers. 2/6/8/10 + scan Pers. 18), dan prompt
        yang dikirim SEMUANYA dikembalikan (transparansi item 3).

      mode="B" — LLM MURNI (baseline skripsi, prompt buta Lampiran 4).
        TANPA retrieval, TANPA profil kognitif, TANPA instruksi tutor,
        TANPA riwayat sesi, TANPA follow-up. Tiga lapis bukti kemurnian
        (revisi item 2):
          1. prompt_sent — prompt persis yang dikirim, bisa diperiksa;
          2. NoRAGGuard — retrieval/embedding DIBLOKIR secara teknis
             selama permintaan; upaya apa pun melempar RAGBlockedError;
          3. no_rag_proof — laporan guard (jumlah upaya terblokir = 0
             membuktikan tidak ada jalur kode yang menyentuh RAG).
        Jalur LLM: query_llm_ollama_raw() — tanpa system prompt, tanpa
        fallback provider lain; identik dengan evaluation/runner.py.

    Q: "Bagaimana profil kognitif mempengaruhi jawaban LLM?" (mode A)
    A: Profil kognitif diinjeksikan ke prompt melalui dua mekanisme:
       a. Label deskriptif ({label}) → instruksi gaya penjelasan
       b. Konteks dokumen profil ({context}) → chunk dari file kode
          kognitif (mis. 3TAR.txt)

    Q: "Di mana log interaksi disimpan?"
    A: log_interaction() di app/services/session.py
       → history_logs/ sebagai JSON per sesi (mode ikut dicatat).
    ═══════════════════════════════════════════════════════════════
    """
    if str(mode).upper() == "B":
        return _generate_reply_kondisi_b(message, session_id)
    return _generate_reply_kondisi_a(
        message, cognitive_code, session_id, rl_selected, rl_phase
    )


def _generate_reply_kondisi_a(
    message:        str,
    cognitive_code: str,
    session_id:     str,
    rl_selected:    Optional[bool] = None,
    rl_phase:       Optional[str]  = None,
) -> Dict:
    """Kondisi A: RAG + profil kognitif + instruksi tutor (alur asli)."""
    V.banner(f"KONDISI A — RAG + profil kognitif ({cognitive_code})")
    # Langkah 1: label profil kognitif untuk prompt
    label       = cognitive_label(cognitive_code)
    history     = get_session(session_id)
    history_txt = format_history(history)
    V.step(f"Profil kognitif: {cognitive_code} → {label}")
    V.step(f"Riwayat sesi: {len(history_txt)} char")

    # Langkah 3: RAG — ambil Top-K chunk paling relevan
    # Lihat app/services/rag.py → retrieve() untuk detail
    chunks  = retrieve(message, cognitive_code)
    context = chunks_to_context(chunks)

    template = CHAT_CODE_PROMPT_TEMPLATE if is_code_like(message) else CHAT_PROMPT_TEMPLATE
    prompt   = template.format(
        label=label, history=history_txt, code=cognitive_code,
        context=context, message=message,
        check_understanding_lead=CHECK_UNDERSTANDING_LEAD,
    )
    V.prompt_echo("PROMPT KONDISI A (konteks RAG + profil + instruksi tutor)",
                  prompt)

    reply    = query_llm(prompt)
    # Soal lanjutan dibangun dari konteks MATERI saja (chunk profil kognitif
    # dikeluarkan) agar soal teknikal-topikal, bukan tentang gaya belajar.
    followup = _generate_followup(message, reply,
                                  _material_only_context(chunks), label)

    live = (compute_live_metrics_a(message, chunks, reply)
            if _settings.demo_live_metrics else None)

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
        # ── transparansi (revisi pasca-sidang item 1 & 3) ──────────────
        "mode":         "A",
        "rag_used":     True,
        "retrieved":    [{
            "rank":    i + 1,
            "source":  c.get("source"),
            "topic":   c.get("topic"),
            "score":   c.get("score"),
            "preview": (c.get("text") or "")[:200],
        } for i, c in enumerate(chunks)],
        "live_metrics": live,
        "no_rag_proof": None,          # guard hanya aktif pada Kondisi B
        "prompt_sent":  prompt,
    }


def _generate_reply_kondisi_b(message: str, session_id: str) -> Dict:
    """
    Kondisi B: LLM murni — prompt buta Lampiran 4, dibungkus NoRAGGuard.

    Sengaja TANPA riwayat sesi (jawaban sebelumnya di sesi bisa berasal
    dari Kondisi A yang mengandung materi RAG — menyertakannya akan
    mengkontaminasi kemurnian B), TANPA profil, TANPA follow-up.
    """
    V.banner("KONDISI B — LLM MURNI (prompt buta, NoRAGGuard aktif)")
    prompt = BLIND_PROMPT_TEMPLATE.format(query=message)
    V.prompt_echo("PROMPT KONDISI B (persis Lampiran 4 — tanpa konteks, "
                  "tanpa profil, tanpa instruksi, tanpa riwayat)", prompt)

    with NoRAGGuard() as guard:
        reply = query_llm_ollama_raw(prompt)
        proof = guard.report()
    V.step(f"no_rag_proof: {proof['retrieval_calls_blocked']} retrieval + "
           f"{proof['embedding_calls_blocked']} embedding diblokir "
           f"(0 = tidak ada jalur kode menyentuh RAG)")

    live = (compute_live_metrics_b(reply)
            if _settings.demo_live_metrics else None)

    log_interaction(
        session_id        = session_id,
        cognitive         = "B-MURNI",
        user_message      = message,
        reply             = reply,
        followup_question = "",
        rl_selected       = None,
        rl_phase          = None,
    )

    return {
        "reply":             reply,
        "followup_question": "",       # mode B tidak beralur tutor
        "cognitive":         "—",
        "session_id":        session_id,
        "mode":         "B",
        "rag_used":     False,
        "retrieved":    [],
        "live_metrics": live,
        "no_rag_proof": proof,
        "prompt_sent":  prompt,
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
        followup = _generate_followup(correct_answer, feedback,
                                      _material_only_context(chunks), label)

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


def _material_only_context(chunks) -> str:
    """
    Konteks untuk PEMBUATAN SOAL lanjutan: hanya chunk materi teknikal.

    Chunk profil kognitif (file bernama kode kognitif, mis. 2PAR.txt)
    dikeluarkan — isinya deskripsi gaya belajar, bukan materi, dan bila
    ikut serta llama3 cenderung membuat "soal" tentang gaya belajar atau
    menyalin catatan meta dari dokumen. Bila semua chunk ternyata profil
    (kasus langka), kembalikan seluruhnya agar konteks tidak kosong.
    Hanya dipakai jalur followup — prompt jawaban utama Kondisi A TIDAK
    berubah (menjaga konsistensi dengan desain skripsi).
    """
    def _is_profile(c) -> bool:
        stem = os.path.splitext(str(c.get("source", "")))[0].upper()
        return stem in VALID_COGNITIVE_TYPES or (
            "_" in stem and stem.split("_")[0] in VALID_COGNITIVE_TYPES
        )
    material = [c for c in chunks if not _is_profile(c)]
    return chunks_to_context(material if material else chunks)


# ── Ekstraksi & validasi pertanyaan lanjutan ──────────────────────────────
_PREAMBLE_RE = re.compile(
    r"^\s*(\*\*|__)?\s*(berikut( adalah)?\b.*?[:：]?|pertanyaan"
    r"( lanjutan)?(\s*\d+)?|soal(\s*lanjutan)?(\s*\d+)?|question(\s*\d+)?|"
    r"latihan(\s*\d+)?)\s*(\*\*|__)?\s*[:：.]?\s*$",
    re.IGNORECASE,
)
_NOTE_RE = re.compile(
    r"^\s*[\(\[]?\s*(note|catatan|nb|keterangan|instruksi|hint|petunjuk"
    r"\s+penilai)\b|the student is expected|expected answer|answer format",
    re.IGNORECASE,
)


def _extract_question(raw: str) -> Optional[str]:
    """
    Bersihkan keluaran LLM menjadi SATU teks soal Bahasa yang valid.

    Perbaikan atas heuristik lama `lines[-1]` (ambil baris terakhir) yang
    justru memilih catatan meta seperti "(Note: The student is expected …)"
    ketika llama3 menambahkannya SETELAH soal. Langkah:
      1. Buang pagar kode, penomoran, judul/preamble ("Pertanyaan 1",
         "Berikut soal…"), dan baris catatan meta (Note/Catatan/…).
      2. Gabungkan baris tersisa; potong semua yang berada SETELAH tanda
         tanya terakhir (membuang ekor non-soal yang lolos filter baris).
      3. Validasi: mengandung '?', panjang wajar, bukan pola catatan.
    Return None bila tidak ada soal valid — pemanggil akan retry/fallback.
    """
    if not raw or not raw.strip():
        return None
    text = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
    kept = []
    for ln in text.split("\n"):
        # buang dekorasi markdown (** __ ` #) di tepi baris sebelum dicek
        ln = re.sub(r"^[\s#>*_`]+|[\s*_`]+$", "", ln.strip())
        if not ln or _PREAMBLE_RE.match(ln) or _NOTE_RE.search(ln):
            continue
        kept.append(re.sub(r"^\s*(?:\d+[\.\)]|[-*•>])\s*", "", ln))
    if not kept:
        return None
    joined = re.sub(r"\s+", " ", " ".join(kept)).strip()
    qpos = joined.rfind("?")
    if qpos == -1:
        return None
    joined = joined[: qpos + 1].strip().strip('"').strip()
    if len(joined) < 15 or _NOTE_RE.search(joined):
        return None
    return joined


def _generate_followup(
    original_question: str, tutor_reply: str, context: str, label: str,
) -> str:
    """
    Hasilkan SATU pertanyaan lanjutan teknikal (Bahasa Indonesia, diakhiri
    '?'). Keluaran LLM divalidasi via _extract_question; bila tidak valid,
    dicoba ulang sekali dengan peringatan format; bila tetap gagal, dipakai
    pertanyaan cadangan deterministik agar UI tidak pernah menampilkan
    catatan meta / teks kosong.
    """
    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        label=label, original_question=original_question,
        reply=tutor_reply[:600], context=context,
    )
    for attempt in (1, 2):
        raw = query_llm(prompt).strip()
        question = _extract_question(raw)
        if question:
            if attempt == 2:
                V.note("Followup: percobaan ke-2 berhasil setelah keluaran "
                       "pertama tidak valid.")
            return question
        V.note(f"Followup percobaan {attempt} tidak valid "
               f"(keluaran mentah: {raw[:120]!r}) — "
               + ("mencoba ulang dengan peringatan format."
                  if attempt == 1 else "memakai pertanyaan cadangan."))
        prompt += (
            "\n\nPERINGATAN: Keluaranmu sebelumnya melanggar format. "
            "Ulangi. Tulis HANYA satu soal berbahasa Indonesia yang diakhiri "
            "tanda tanya (?). Tanpa judul, tanpa nomor, tanpa catatan, tanpa "
            "'(Note: ...)', tanpa teks bahasa Inggris."
        )
    # Cadangan deterministik — tetap topikal & berakhir '?'
    topic = re.sub(r"\s+", " ", original_question).strip()
    if len(topic) > 140:
        topic = topic[:140].rsplit(" ", 1)[0] + "…"
    return (f"Berdasarkan penjelasan di atas, uraikan dengan bahasamu "
            f"sendiri langkah demi langkah bagaimana konsep tersebut "
            f"diterapkan untuk menjawab: \"{topic}\" — apa hasil akhirnya?")


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
