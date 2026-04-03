# CSIPBLLM — Personalized Learning System for Computational Thinking

> **Pengembangan Sistem Pembelajaran Terpersonalisasi Mata Kuliah CT Menggunakan Chatbot Berbasis LLM dan RAG**

An adaptive AI tutor powered by **FastAPI**, **OpenAI GPT**, **FAISS**, and **RAG** that personalises explanations and evaluations based on a student's cognitive profile across **48 cognitive type combinations**.

---

## 📁 Project Structure

```
pls-ct/
├── main.py                        # Uvicorn entry-point
├── requirements.txt
├── .env.example                   # Copy to .env and fill in secrets
├── .gitignore
│
├── app/                           # Application package
│   ├── main.py                    # FastAPI app factory (create_app)
│   ├── core/
│   │   ├── config.py              # Centralised settings (pydantic-settings)
│   │   ├── cognitive.py           # 48 cognitive types + label helpers
│   │   └── prompts.py             # All LLM prompt templates
│   ├── models/
│   │   └── schemas.py             # Pydantic request / response schemas
│   ├── services/
│   │   ├── llm.py                 # OpenAI chat + embedding wrapper
│   │   ├── rag.py                 # FAISS vector indexing + retrieval
│   │   ├── session.py             # Per-session history + conversation log
│   │   └── tutor.py               # Core tutoring business logic
│   ├── api/
│   │   └── routes/
│   │       ├── reference.py       # GET /cognitive-types
│   │       ├── tutor.py           # POST /chat, POST /evaluate
│   │       └── history.py         # GET /history, GET /download-history
│   └── utils/
│       ├── latex.py               # LaTeX → MathJax normaliser
│       └── code_detector.py       # Heuristic code-snippet detector
│
├── evaluation/                    # RAG evaluation suite
│   ├── metrics.py                 # Pure metric functions (Precision@K, etc.)
│   ├── faithfulness.py            # Faithfulness + hallucination detection
│   ├── test_cases.py              # Static 20-case test dataset
│   └── runner.py                  # Full evaluation orchestrator + reporting
│
├── scripts/
│   └── run_evaluation.py          # CLI runner for the evaluation suite
│
├── tests/
│   └── test_core.py               # Unit tests (no API calls needed)
│
├── static/
│   ├── index.html                 # Frontend UI
│   ├── css/style.css
│   └── js/script.js
│
├── materials/                     # RAG knowledge base
│   ├── {CODE}.txt                 # e.g. 3TGI.txt — per cognitive type
│   └── *.md / *.txt               # Shared topic files (global fallback)
│
└── logs/
    ├── history/                   # Auto-saved conversation logs (JSON + CSV)
    └── eval_results/              # Evaluation reports (JSON + CSV + TXT)
```

---

## ✨ Features

- 🧠 **48 Cognitive Types** — Every response is tailored to the student's learning profile (level, practical/theoretical, analytical/global, individual/relational)
- 📚 **Always-On RAG** — Retrieves relevant chunks from per-type + global material files before every LLM call
- 🤖 **Adaptive Evaluation** — Detects correct/incorrect answers and provides scaffolded hints that increase in detail with each wrong attempt
- 🔁 **Follow-up Question Generator** — Automatically generates a new technical case-study question after each explanation or evaluation
- 🧮 **LaTeX Normalisation** — Standardises all math notation to MathJax-compatible `\(...\)` / `\[...\]`
- 💾 **Conversation Logging** — All sessions saved automatically to JSON + CSV
- 🌐 **Web UI** — Built-in frontend with MathJax, Markdown rendering, and download buttons

---

## ⚙️ Requirements

- Python 3.10+
- An **OpenAI API key** (`gpt-3.5-turbo` + `text-embedding-3-small`)
- Optional: `faiss-cpu` for fast vector search (NumPy fallback otherwise)

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ajisakarsyi/pls-ct.git
cd pls-ct
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 3. Run the server

```bash
python main.py
# or
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 🧠 Cognitive Type System

Each student is assigned a 4-character code:

```
{Level}{PT}{AG}{IR}
```

| Dimension | Options | Meaning |
|---|---|---|
| **Level** | `1` – `6` | Bloom's Taxonomy level |
| **PT** | `P` / `T` | Praktis / Teoretis |
| **AG** | `A` / `G` | Analitis / Global |
| **IR** | `I` / `R` | Individual / Relasional |

**Example:** `3TGR` → Level 3, Theoretical, Global, Relational

48 unique profiles (6 × 2 × 2 × 2).  Full list: `GET /cognitive-types`

---

## 📡 API Reference

### `POST /chat`

```json
{ "message": "Apa itu algoritma?", "cognitive": "2TAR", "session_id": "s1" }
```

Response: `reply`, `followup_question`, `cognitive`, `session_id`

### `POST /evaluate`

```json
{
  "answer": "32/3",
  "correct_answer": "…tutor explanation…",
  "active_question": "Berapa luas area di bawah f(x)?",
  "wrong_count": 0,
  "cognitive": "2TAR",
  "session_id": "s1"
}
```

Response: `is_correct`, `feedback`, `hint_level`, `followup_question`, `cognitive`, `session_id`

**Scaffolding levels:**

| `wrong_count` | Level |
|---|---|
| 0 | Evaluasi Awal — brief direction |
| 1 | Petunjuk Terarah — specific pointer |
| 2 | Dukungan Remedial — step-by-step with examples |
| 3+ | Panduan Langkah-demi-Langkah — full re-explanation |

### `GET /cognitive-types`
### `GET /history?format=json|text`
### `GET /download-history?format=json|csv`

---

## 🔬 Running the Evaluation Suite

```bash
python scripts/run_evaluation.py
# or with a custom server URL:
python scripts/run_evaluation.py --base-url http://localhost:8000
```

Results are saved to `logs/eval_results/`. Metrics computed:

| Category | Metrics |
|---|---|
| Retrieval | Precision@K, Recall@K, MeanSim, Coverage, Source Diversity |
| Generation | Faithfulness score, Hallucination risk |
| Answer Quality | Boolean accuracy via `/evaluate` |
| Offline Stats | Interaction count, session distribution, reply length stats |

---

## 🧪 Running Unit Tests

```bash
pip install pytest
pytest tests/
```

The tests in `tests/test_core.py` cover all pure utility functions and require **no live API calls**.

---

## 🛠️ Configuration

All settings live in `app/core/config.py` and are loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | Custom OpenAI-compatible base URL |
| `CHAT_MODEL` | `gpt-3.5-turbo` | Chat completions model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8000` | Server port |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web framework and server |
| `openai` | GPT chat + embeddings |
| `faiss-cpu` | Vector similarity search |
| `numpy` | Embedding math / NumPy fallback |
| `langchain` + `langchain-community` | Session memory |
| `pydantic` + `pydantic-settings` | Validation + config |

---

## 📄 License

This project is part of an undergraduate research thesis at IPB University. For academic use.
