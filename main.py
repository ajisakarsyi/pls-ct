"""
main.py
────────
Project root entry-point.

Run with:
    python main.py
or:
    uvicorn main:app --reload
"""

import uvicorn

from app.main import app  # noqa: F401 — exposed for uvicorn
from app.core.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    print(f"\n🚀  Server   →  http://{settings.host}:{settings.port}")
    print(f"🧠  Model    →  {settings.chat_model}")
    print(f"🌐  API Base →  {settings.openai_api_base}")
    print(f"📚  Docs     →  http://{settings.host}:{settings.port}/docs\n")

    uvicorn.run(
        "app.main:app",
        host      = settings.host,
        port      = settings.port,
        reload    = settings.reload,
        log_level = settings.log_level,
    )
