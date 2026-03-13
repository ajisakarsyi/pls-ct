# CSIPBLLM — Combined RAG + RL Personalized Learning System

**v4.0.0** — RAG-based tutoring + RL Contextual Bandit adaptive pedagogy.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/ajisakarsyi/pls-ct.git
cd pls-ct
pip install -r requirements.txt

# 2. Run the server
python main.py
# or
uvicorn app.main:app --reload

# 3. Open browser
# UI   → http://localhost:8000
# Docs → http://localhost:8000/docs
```

No `.env` file needed — API key and config are in `app/core/config.py`.

---

## Architecture

```
Student sends /chat
  └─► RL Agent selects best LT (seeding: forced LT, free: ε-greedy)
       └─► RAG retrieves relevant materials for that cognitive type
            └─► LLM generates personalised explanation + follow-up question

Student answers /evaluate
  └─► LLM strictly evaluates correctness (two-step)
       └─► Scaffolded feedback (3 levels + full remediation)
            └─► RL records reward, updates Q-table (M/P/E + MLR weights)
```

---

## API Endpoints

### Tutor
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a question, get personalised RAG+RL response |
| `POST` | `/evaluate` | Evaluate student answer, update RL Q-table |

### RL
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/rl/recommend/{id}` | Greedy best LT — no side effects |
| `GET` | `/rl/select/{id}` | ε-greedy select (advances counter) |
| `GET` | `/rl/phase/{id}` | Seeding vs free phase status |
| `GET` | `/rl/summary/{id}` | Full session analytics |
| `GET` | `/rl/changes/{id}` | LT recommendation change log |
| `GET` | `/rl/log/{id}` | Step-by-step reward/M/P/E log |
| `GET` | `/rl/plots/{id}` | All 7 plots as base64 PNGs |
| `GET` | `/rl/sessions` | List all active sessions |
| `POST` | `/rl/refit` | Force MLR weight refit |
| `DELETE` | `/rl/session/{id}` | Delete a session |

### Reference / History
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/cognitive-types` | All 48 valid cognitive codes |
| `GET` | `/history` | Conversation log |
| `GET` | `/download-history` | Download log as JSON or CSV |

---

## Simulation Tools

```bash
python simulate_rl.py                              # 1 session, 15 Q
python simulate_rl.py --sessions 3 --questions 30
python simulate_rl.py --fixed-lt TGI --questions 25
python simulate_rl.py --story --assigned-lt PAI --true-lt TAR
python simulate_rl.py --per-lt 50
python simulate_rl.py --cross-lt 100

python testgila.py                   # 8-student cross-LT benchmark
python rag_evaluator.py              # RAG eval suite (server must be running)
```

---

## Project Structure

```
pls-ct-combined/
├── main.py                    # Entry point
├── app/
│   ├── main.py                # FastAPI factory
│   ├── core/
│   │   ├── config.py          # All settings (key, model, paths) — edit here
│   │   ├── cognitive.py       # 48 cognitive type codes
│   │   └── prompts.py         # LLM prompt templates
│   ├── models/schemas.py      # Pydantic request/response models
│   ├── services/
│   │   ├── llm.py             # OpenAI chat + embedding wrapper
│   │   ├── rag.py             # FAISS/NumPy RAG retrieval
│   │   ├── session.py         # Chat history + conversation logs
│   │   ├── tutor.py           # Tutoring logic
│   │   └── rl.py              # RL service (registry, selection, recording, plots)
│   ├── api/routes/
│   │   ├── tutor.py           # POST /chat, POST /evaluate
│   │   ├── rl.py              # GET /rl/*
│   │   ├── history.py         # GET /history
│   │   └── reference.py       # GET /cognitive-types
│   └── utils/
│       ├── latex.py           # LaTeX normalisation
│       └── code_detector.py   # Code snippet detection
├── pedagogy_selector.py       # RL agent
├── rl_metrics.py              # M/P/E metrics + MLR
├── simulate_rl.py             # CLI simulator + plots
├── testgila.py                # 8-student cross-LT benchmark
├── rag_evaluator.py           # RAG evaluation suite
├── materials/                 # 48 cognitive-type + shared topic files
├── static/                    # Frontend
├── history_logs/              # Conversation logs
├── rl_logs/                   # RL step logs
└── requirements.txt
```
