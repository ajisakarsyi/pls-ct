"""
app/api/routes/history.py
──────────────────────────
Endpoints for reading and downloading conversation logs.
Logs are stored in history_logs/ (kept from RL project).
"""

import os

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.services.session import get_all_logs

router    = APIRouter(tags=["History"])
_settings = get_settings()

_HISTORY_BASE = "conversation_log"


@router.get("/history", summary="Ambil riwayat percakapan")
def get_history(format: str = "json"):  # noqa: A002
    logs = get_all_logs()
    if not logs:
        return {"history": []}

    if format == "json":
        return {"history": logs}

    lines = []
    for i, conv in enumerate(logs, 1):
        lines += [
            f"[Percakapan {i}]",
            f"Profil Kognitif : {conv.get('cognitive', '-')}",
            f"RL Phase        : {conv.get('rl_phase', '-')}",
            f"Pertanyaan      : {conv['user_message']}",
            f"Jawaban         :\n{conv['reply']}",
            "-" * 60,
        ]
    return {"data": "\n".join(lines)}


@router.get(
    "/download-history",
    summary="Unduh riwayat percakapan sebagai JSON atau CSV",
)
def download_history(format: str = "json"):  # noqa: A002
    hist_dir = _settings.history_dir
    if format.lower() == "csv":
        path      = os.path.join(hist_dir, f"{_HISTORY_BASE}.csv")
        media_type = "text/csv"
        filename  = f"{_HISTORY_BASE}.csv"
    else:
        path      = os.path.join(hist_dir, f"{_HISTORY_BASE}.json")
        media_type = "application/json"
        filename  = f"{_HISTORY_BASE}.json"

    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Belum ada riwayat percakapan.",
        )
    return FileResponse(path, media_type=media_type, filename=filename)
