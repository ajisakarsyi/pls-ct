"""
patch_config_chat_provider.py
──────────────────────────────
Adds the missing chat_provider field to config.py.
Run from your project root: python patch_config_chat_provider.py
"""
import sys, os, shutil

CONFIG_PATH = os.path.join("app", "core", "config.py")

if not os.path.exists(CONFIG_PATH):
    print(f"ERROR: {CONFIG_PATH} not found. Run from project root.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if "chat_provider" in content:
    print("chat_provider already exists in config.py. Nothing to do.")
    sys.exit(0)

# Back up
shutil.copy2(CONFIG_PATH, CONFIG_PATH + ".bak2")
print(f"Backed up to {CONFIG_PATH}.bak2")

# Insert chat_provider right after ollama_embed_model line
OLD = '    ollama_embed_model: str = "nomic-embed-text"'
NEW = '    ollama_embed_model: str = "nomic-embed-text"\n    chat_provider: str      = "auto"'

if OLD not in content:
    # Try without leading spaces (different formatting)
    OLD = 'ollama_embed_model: str = "nomic-embed-text"'
    NEW = 'ollama_embed_model: str = "nomic-embed-text"\n    chat_provider: str      = "auto"'

if OLD not in content:
    print("ERROR: Could not find ollama_embed_model line to insert after.")
    print("Please manually add this line to your Settings class in config.py:")
    print('    chat_provider: str = "auto"')
    sys.exit(1)

content = content.replace(OLD, NEW, 1)

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ chat_provider added to config.py")
print('   Default: "auto" (probe ChatAnywhere, fall back to Ollama if exhausted)')
