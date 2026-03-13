"""
app/services/session.py
────────────────────────
In-memory per-session chat history using LangChain's ChatMessageHistory.
Handles writing conversation logs to JSON and CSV in history_logs/.

Stores optional RL fields (rl_selected, rl_phase) alongside each interaction
so conversation logs carry full context for the RAG evaluator.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory

from app.core.config import get_settings

logger    = logging.getLogger(__name__)
_settings = get_settings()

_session_store:     Dict[str, ChatMessageHistory] = {}
_conversation_log:  List[Dict[str, Any]]          = []

_HISTORY_BASE = "conversation_log"


def get_session(session_id: str) -> ChatMessageHistory:
    """Return (or create) the in-memory history for *session_id*."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def format_history(history: ChatMessageHistory, max_chars: int = None) -> str:
    """Render history as plain text for inclusion in an LLM prompt."""
    if max_chars is None:
        max_chars = _settings.max_history_chars

    if not getattr(history, "messages", None):
        return "Tidak ada riwayat sebelumnya."

    lines: List[str] = []
    for msg in history.messages:
        role    = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        prefix  = (
            "[Mahasiswa]" if role == "human"
            else "[Tutor]"  if role == "ai"
            else "[Riwayat]"
        )
        lines.append(f"{prefix} {content}")

    text = "\n".join(lines)
    return text if len(text) <= max_chars else "...\n" + text[-max_chars:]


def log_interaction(
    session_id:        str,
    cognitive:         str,
    user_message:      str,
    reply:             str,
    followup_question: str,
    rl_selected:       Optional[bool] = None,
    rl_phase:          Optional[str]  = None,
) -> None:
    """Append an interaction to the in-memory log and flush to disk."""
    entry: Dict[str, Any] = {
        "timestamp":        datetime.now().isoformat(),
        "session_id":       session_id,
        "cognitive":        cognitive,
        "rl_selected":      rl_selected,
        "rl_phase":         rl_phase,
        "user_message":     user_message,
        "reply":            reply,
        "followup_question": followup_question,
    }
    _conversation_log.append(entry)
    _flush_logs()


def _flush_logs() -> None:
    """Write the full conversation log to JSON + CSV."""
    if not _conversation_log:
        return

    os.makedirs(_settings.history_dir, exist_ok=True)
    json_path = os.path.join(_settings.history_dir, f"{_HISTORY_BASE}.json")
    csv_path  = os.path.join(_settings.history_dir, f"{_HISTORY_BASE}.csv")

    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_conversation_log, fh, indent=4, ensure_ascii=False)

        fieldnames = list(_conversation_log[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in _conversation_log:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()
                    }
                )
    except Exception as exc:
        logger.error("Failed to flush logs: %s", exc)


def get_all_logs() -> List[Dict[str, Any]]:
    """Return all in-memory conversation logs."""
    return list(_conversation_log)
