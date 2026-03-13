#!/bin/bash
# ============================================================
# CSIPBLLM — Combined RAG + RL Start Server (Linux / macOS)
# Run every time you want to start the server:
#   ./start.sh
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   CSIPBLLM Combined RAG + RL — Starting Server ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Check venv ────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "❌  Virtual environment not found."
  echo "    Run './setup.sh' first."
  exit 1
fi

# ── Activate venv ─────────────────────────────────────────────
source venv/bin/activate

# ── Load .env ─────────────────────────────────────────────────
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
  echo "✅  Environment variables loaded from .env"
else
  echo "⚠️   .env not found — using defaults."
  echo "    Create .env from .env.example and fill in your API key."
fi

# ── Validate API key ──────────────────────────────────────────
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ]; then
  echo ""
  echo "⚠️   WARNING: OPENAI_API_KEY is not set!"
  echo "    Server will start but GPT calls will fail."
  echo "    Edit .env and fill in a valid OPENAI_API_KEY."
  echo ""
fi

# ── Create directories ────────────────────────────────────────
mkdir -p history_logs rl_logs rl_plots plots logs/eval_results

# ── Start server ──────────────────────────────────────────────
echo ""
echo "🚀  Server         →  http://localhost:8000"
echo "📚  Swagger Docs   →  http://localhost:8000/docs"
echo "🔄  Auto-reload active. Press Ctrl+C to stop."
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
