"""
patch_config_ollama.py
───────────────────────
Run this ONCE from your project root to add Ollama settings to config.py.

Usage:
    python patch_config_ollama.py
"""

import sys, os, shutil

CONFIG_PATH = os.path.join("app", "core", "config.py")

if not os.path.exists(CONFIG_PATH):
    print(f"ERROR: {CONFIG_PATH} not found. Run from your project root.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if "ollama_base_url" in content:
    print("config.py already has Ollama settings. Nothing to do.")
    sys.exit(0)

backup = CONFIG_PATH + ".bak"
shutil.copy2(CONFIG_PATH, backup)
print(f"Backed up original to {backup}")

OLLAMA_BLOCK = """\

    # ── Ollama (embeddings always, chat as fallback) ──────────────────────
    ollama_base_url: str    = "http://localhost:11434"
    ollama_chat_model: str  = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # ── Chat provider ─────────────────────────────────────────────────────
    # auto   -> probe ChatAnywhere on startup; use it if available, else Ollama
    # openai -> always ChatAnywhere
    # ollama -> always local Ollama
    chat_provider: str = "auto"

"""

inserted = False
for marker in ["    # ── RAG", "    rag_chunk", "    embedding_model", "    # ── Session"]:
    if marker in content:
        content = content.replace(marker, OLLAMA_BLOCK + marker, 1)
        inserted = True
        break

if not inserted:
    content = content.replace(
        "    model_config = SettingsConfigDict(",
        OLLAMA_BLOCK + "    model_config = SettingsConfigDict(",
        1,
    )

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("config.py patched successfully.")
print("  ollama_base_url    = http://localhost:11434")
print("  ollama_chat_model  = llama3")
print("  ollama_embed_model = nomic-embed-text")
print("  chat_provider      = auto")
