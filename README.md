# CSIPBLLM — Combined RAG + RL Personalized Learning System

**v5.0.0** — RAG-based tutoring + RL Contextual Bandit adaptive pedagogy.

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
Student sends /chat  (no cognitive field — cold start)
  └─► RL Agent selects LT via ε-greedy exploration-exploitation
       └─► RAG retrieves relevant materials for that cognitive type
            └─► LLM generates personalised explanation + follow-up question

Student answers /evaluate
  └─► LLM strictly evaluates correctness (two-step)
       └─► Intermediate wrong answer → scaffolded hint, NO RL update
            └─► Resolved answer (correct or 5th attempt):
                 └─► RL records reward, updates Q-table (M/P/E + MLR weights)
                      └─► Mastery level promoted on correct answer
```

**Reward function:** `rt = β₀ + α·ΔP + β·ΔM + γ·E`

| Component | Description |
|-----------|-------------|
| `ΔP` | Change in performance (correctness rate) |
| `ΔM` | Change in mastery score (levels 1–6) |
| `E` | Engagement score — **zero on wrong answers** |
| `α β γ` | Weights refitted by MLR every 10 resolved questions |

---

## Cold Start

The system no longer requires the student to choose a Learning Type upfront. The RL agent starts with `ε=0.50` and discovers the best LT through interaction.

**Do not send a `cognitive` field in `/chat` or `/evaluate` requests.** The agent selects the LT automatically from question 1 and returns its choice in the response.

---

## API Endpoints

### Tutor

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a question — RL selects LT, RAG retrieves context, LLM replies |
| `POST` | `/evaluate` | Evaluate student answer, update RL on resolved questions only |

**`POST /chat` — request:**
```json
{
  "message":    "Apa itu algoritma?",
  "session_id": "student-001",
  "category":   "Penggalang"
}
```

**`POST /chat` — response (key fields):**
```json
{
  "reply":             "...",
  "followup_question": "...",
  "cognitive":         "2PAR",
  "rl_selected_lt":    "PAR",
  "rl_epsilon":        0.431,
  "rl_q_values":       { "PAR": 0.082, "TAR": 0.071, "...": "..." },
  "rl_phase":          { "phase": "free", "global_question_count": 4 }
}
```

**`POST /evaluate` — request:**
```json
{
  "answer":          "Algoritma adalah urutan langkah-langkah...",
  "correct_answer":  "<the reply field from /chat>",
  "active_question": "<the followup_question from /chat>",
  "wrong_count":     0,
  "session_id":      "student-001",
  "category":        "Penggalang"
}
```

> Increment `wrong_count` by 1 per failed attempt. RL only updates on resolved questions (`is_correct=true` or `wrong_count >= 4`).

**`POST /evaluate` — response (key fields):**
```json
{
  "is_correct":        true,
  "feedback":          "...",
  "hint_level":        "Evaluasi Awal",
  "followup_question": "...",
  "cognitive":         "2PAR",
  "rl": {
    "mastery_level":  2,
    "mastery_label":  "Pemahaman dasar",
    "reward":         0.1823,
    "next_cognitive": "3PAR"
  },
  "lt_change": { "changed": false, "current_lt": "PAR" }
}
```

### RL

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/rl/coldstart/{id}` | Initialise cold-start session — returns Q-values and plot URL |
| `GET` | `/rl/recommend/{id}` | Greedy best LT — no side effects |
| `GET` | `/rl/select/{id}` | ε-greedy select (advances counter) |
| `GET` | `/rl/phase/{id}` | Seeding vs free phase status |
| `GET` | `/rl/summary/{id}` | Full session analytics |
| `GET` | `/rl/changes/{id}` | LT recommendation change log |
| `GET` | `/rl/log/{id}` | Step-by-step reward/M/P/E log |
| `GET` | `/rl/plots/{id}` | All 8 plots as base64 PNGs |
| `GET` | `/rl/plots/{id}/single_line` | Single-line reward chart coloured by LT (auto-saved every 10 Q) |
| `GET` | `/rl/evaluate/{id}` | Run all 3 evaluation techniques |
| `GET` | `/rl/evaluate/{id}/kt_auc` | Knowledge Tracing AUC only |
| `GET` | `/rl/evaluate/{id}/reward_decomposition` | MLR weight stability + component contribution |
| `GET` | `/rl/evaluate/{id}/ope_dr` | Offline Policy Evaluation — doubly robust |
| `GET` | `/rl/sessions` | List all active sessions |
| `POST` | `/rl/refit` | Force MLR weight refit |
| `DELETE` | `/rl/session/{id}` | Delete a session |

### Reference / History

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/cognitive-types` | All 48 valid cognitive codes |
| `GET` | `/history` | Conversation log |
| `GET` | `/download-history` | Download log as JSON or TXT |

---

## Frontend Integration

```javascript
// Step 1 — send question (no cognitive field)
const chat = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message, session_id: studentId, category: 'Penggalang' }),
}).then(r => r.json());

displayReply(chat.reply);
displayFollowup(chat.followup_question);
showRLBadge(chat.rl_selected_lt, chat.rl_epsilon); // show what the agent picked

// Step 2 — submit answer
let wrongAttempts = 0;

const result = await fetch('/evaluate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    answer:          studentAnswer,
    correct_answer:  chat.reply,
    active_question: chat.followup_question,
    wrong_count:     wrongAttempts,
    session_id:      studentId,
  }),
}).then(r => r.json());

if (result.is_correct) {
  showSuccess(result.feedback);
  wrongAttempts = 0;
  // ready for next question — call /chat again
} else {
  wrongAttempts++;
  showHint(result.feedback, result.hint_level);
  // student tries again — only call /evaluate, not /chat
}
```

**Reward chart** — available after 10 resolved questions:
```html
<img src="/rl/plots/student-001/single_line" alt="RL reward chart" />
```

---

## Configuration

All settings in `app/core/config.py` and `pedagogy_selector.py`.

| Setting | Default | Notes |
|---------|---------|-------|
| `openai_api_key` | `sk-...` | ChatAnywhere proxy key |
| `openai_api_base` | `https://api.chatanywhere.org/v1` | Swap for `https://api.openai.com/v1` for direct OpenAI |
| `chat_model` | `gpt-3.5-turbo` | Upgrade to `gpt-4o` for better quality |
| `EPSILON_INIT` | `0.50` | Starting exploration rate |
| `EPSILON_DECAY` | `0.97` | Per-question decay |
| `EPSILON_MIN` | `0.05` | Minimum exploration floor |
| `N_MAX` | `5` | Max wrong attempts before answer is revealed |
| `MLR_REFIT_EVERY` | `10` | Refit α/β/γ every N resolved questions |

---

## Simulation Tools

```bash
python simulate_rl.py                              # 1 session, 15 Q
python simulate_rl.py --sessions 3 --questions 30
python simulate_rl.py --single-line --profile student_tgi --questions 40 --sessions 2
python simulate_rl.py --fixed-lt TGI --questions 25
python simulate_rl.py --story --assigned-lt PAI --true-lt TAR
python simulate_rl.py --per-lt 50
python simulate_rl.py --cross-lt 100
python simulate_rl.py --correlation

python rag_evaluator.py              # RAG eval suite (server must be running)
```

---

## Built-in Evaluation

Three research-backed techniques available via `/rl/evaluate/{session_id}` after sufficient session data.

| Technique | Metric | Source |
|-----------|--------|--------|
| Knowledge Tracing AUC | AUC-ROC of `mastery_score` predicting correctness on held-out questions. ≥ 0.72 beats BKT baseline. | Liu et al., NeurIPS 2022 |
| Reward Decomposition | Weight stability of α/β/γ across MLR refits; which component dominates reward. | Septon et al., AAMAS 2023 |
| OPE Doubly Robust | Estimates whether pure-exploit (ε=0) would earn more than current ε-greedy. | Zhan et al., KDD 2021 |

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
├── pedagogy_selector.py       # RL agent + SessionRegistry
├── rl_metrics.py              # M/P/E metrics + MLR
├── simulate_rl/               # CLI simulator package
│   ├── profiles.py            # Student profiles + StudentSimulator
│   ├── runners.py             # run_session, run_simulation, run_story, ...
│   ├── plots.py               # All matplotlib plot functions
│   └── terminal.py            # Terminal output helpers
├── simulate_rl.py             # Backward-compat shim
├── evaluation/
│   └── evaluator.py           # KT AUC, Reward Decomposition, OPE-DR
├── rag_evaluator.py           # RAG evaluation suite
├── materials/                 # 48 cognitive-type + shared topic files
├── static/                    # Frontend
├── history_logs/              # Conversation logs
├── rl_logs/                   # RL step logs per session
├── rl_plots/                  # Auto-saved reward plots per session
└── requirements.txt
```

---

## Developer Notes

**CORS** — not configured by default. Add to `app/main.py` if your frontend is on a different origin:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**Session persistence** — sessions are in-memory. Restarting the server resets all sessions.

**Mastery progression** — one correct answer promotes the LT's level (`1PAR → 2PAR → 3PAR`). The `next_cognitive` field in the `/evaluate` response always reflects the updated level.

**Reward chart** — auto-saved to `rl_plots/{session_id}_single_line.png` after every 10 resolved questions. Also always accessible at `GET /rl/plots/{session_id}/single_line`.