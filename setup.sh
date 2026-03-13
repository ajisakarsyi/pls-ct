#!/bin/bash
# ============================================================
# CSIPBLLM — Combined RAG + RL Setup (Linux / macOS)
# Run once on first install:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   CSIPBLLM Combined RAG + RL — First-time Setup ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Install Python 3.10+ first."
  exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python $PYTHON_VERSION found."

# ── 2. Create virtual environment ──────────────────────────────
if [ ! -d "venv" ]; then
  echo ""
  echo "📦  Creating virtual environment..."
  python3 -m venv venv
  echo "✅  Virtual environment 'venv/' created."
else
  echo "✅  Virtual environment 'venv/' already exists, skipping."
fi

# ── 3. Activate and install packages ───────────────────────────
echo ""
echo "📥  Installing dependencies from requirements.txt..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅  All dependencies installed."

# ── 4. Create .env from .env.example if missing ────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚙️   '.env' created from '.env.example'."
  echo "    ➡️  REQUIRED: Open '.env' and fill in your OPENAI_API_KEY!"
else
  echo "✅  '.env' already exists."
fi

# ── 5. Create required directories ─────────────────────────────
mkdir -p history_logs rl_logs rl_plots plots logs/eval_results
echo "✅  Directories ready: history_logs/ rl_logs/ rl_plots/ plots/ logs/eval_results/"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:                   ║"
echo "║                                                  ║"
echo "║   1. Edit .env → fill in OPENAI_API_KEY         ║"
echo "║   2. Run: ./start.sh                            ║"
echo "║   3. Open: http://localhost:8000                ║"
echo "║   4. Docs: http://localhost:8000/docs           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
