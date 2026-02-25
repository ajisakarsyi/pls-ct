# CSIPBLLM Personalized Learning System --- Backend (Ollama GPT-OSS)

Welcome! This README is designed for **regular users** who want to run
and use the backend system without needing deep technical knowledge.\
This backend powers an AI tutor with **personalized cognitive
profiles**, **RAG (Retrieval-Augmented Generation)**, and **adaptive
evaluation**.

------------------------------------------------------------------------

## 🚀 1. What This Program Does

This backend provides: - **AI chat tutor** using a local LLM via
**Ollama** - **Personalized learning** based on: - Cognitive style:
`PAR` / `TAR` - CQ preferences: `P` / `T` / `A` - **RAG context
retrieval** using your files in `/materials` - **Memory per session** -
**Automatic logging** to JSON + CSV - **Evaluation mode** with adaptive
scaffolding

Frontend can call: - `POST /chat` - `POST /evaluate`

------------------------------------------------------------------------

## 🧩 2. Requirements

### You need:

-   **Python 3.10+**
-   **Ollama installed**
-   An LLM model:
    -   Example: `deepseek-r1:8b`
-   An embedding model:
    -   Example: `mxbai-embed-large`
-   Install Python dependencies:

```{=html}
<!-- -->
```
    pip install -r requirements.txt

------------------------------------------------------------------------

## ⚙️ 3. Running the Backend

### Step 1 --- Start Ollama

Make sure Ollama is running.

    ollama serve

### Step 2 --- Pull the required model

    ollama pull deepseek-r1:8b
    ollama pull mxbai-embed-large

### Step 3 --- Run FastAPI server

From the backend folder:

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open:

    http://localhost:8000

------------------------------------------------------------------------

## 📂 4. Folder Structure Overview

    /static          → Frontend files (index.html)
    /materials       → Your RAG source files (.txt / .md)
    /history_logs    → Auto-generated conversation logs
    main.py          → Backend program

If `/materials` doesn't exist, create it:

    mkdir materials

Add `.txt` or `.md` files --- these will be used as **knowledge
sources**.

------------------------------------------------------------------------

## 💬 5. How to Use the Chat API

### Endpoint

`POST /chat`

### Example Request Body

    {
      "message": "Apa itu variabel dalam pemrograman?",
      "cognitive": "par",
      "cq1": "t",
      "cq2": "a",
      "mode": "accurate",
      "session_id": "siswa123"
    }

### Response includes:

-   Main explanation
-   Comparative explanation
-   Follow-up question
-   Whether code-like input detected
-   Whether RAG was used

------------------------------------------------------------------------

## 🧠 6. How Evaluation Works

### Endpoint

`POST /evaluate`

Example:

    {
      "answer": "variabel adalah tempat menyimpan data",
      "correct_answer": "wadah untuk menyimpan nilai",
      "wrong_count": 1,
      "session_id": "siswa123"
    }

The system: - Detects correctness - Generates hints depending on
mistakes - Provides one follow-up question - Generates
`[STATUS: CORRECT]` or `[STATUS: INCORRECT]`

------------------------------------------------------------------------

## 📊 7. Conversation Logs (Automatic)

Logs are saved automatically to:

    /history_logs/conversation_log.json
    /history_logs/conversation_log.csv

Each interaction includes: - Timestamp\
- User message\
- Cognitive profile\
- RAG sources\
- System replies

------------------------------------------------------------------------

## 🧠 8. RAG (Retrieval-Augmented Generation)

RAG activates only when:

    "mode": "accurate"

The backend: 1. Reads all `.txt` and `.md` files in `/materials` 2.
Chunks them into pieces 3. Embeds them with `mxbai-embed-large` 4. Finds
the most relevant chunks for each question

------------------------------------------------------------------------

## 🛠️ 9. Troubleshooting

### ❗ Ollama not detected

Ensure Ollama is running on: - `localhost:11434` - or `localhost:11435`

### ❗ Embedding model error

Re-pull the model:

    ollama pull mxbai-embed-large

### ❗ RAG not working

Check that `/materials` has readable files.

------------------------------------------------------------------------

## 📝 10. Credits

CSIPBLLM Backend --- Personalized & Cognitive AI Tutor\
Powered by **FastAPI + Ollama + LangChain + FAISS**

------------------------------------------------------------------------

Enjoy exploring your personalized AI learning system! 🎓✨
