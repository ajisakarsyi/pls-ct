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
    # "0.0.0.0" adalah alamat bind (menerima koneksi dari interface mana
    # pun), tapi bukan alamat yang bisa dibuka di browser — tampilkan
    # "localhost" di sini supaya orang baru tidak bingung harus mengetik
    # apa di address bar.
    display_host = "localhost" if settings.host in ("0.0.0.0", "") else settings.host
    # Tampilkan model/provider yang BENAR-BENAR aktif, bukan selalu info
    # OpenAI — default sistem ini adalah Ollama lokal (CHAT_PROVIDER=ollama).
    if settings.chat_provider == "ollama":
        model_info = f"{settings.ollama_chat_model} (Ollama lokal)"
    elif settings.chat_provider == "openai":
        model_info = f"{settings.chat_model} ({settings.openai_api_base})"
    else:  # "auto" — provider sesungguhnya baru ditentukan saat startup
        model_info = f"otomatis (Ollama {settings.ollama_chat_model} / OpenAI {settings.chat_model})"
    print(f"\n🚀  Server   →  http://{display_host}:{settings.port}")
    print(f"🧠  Model    →  {model_info}")
    print(f"📚  Docs     →  http://{display_host}:{settings.port}/docs\n")

    uvicorn.run(
        "app.main:app",
        host      = settings.host,
        port      = settings.port,
        reload    = settings.reload,
        log_level = settings.log_level,
    )