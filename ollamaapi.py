# ============================================================
# CSIPBLLM PERSONALIZED LEARNING SYSTEM — BACKEND
# GPT API · RAG always-on · 48 tipe kognitif
# ============================================================

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
import json, os, time, re, csv
import numpy as np
from datetime import datetime

# ============================================================
# HYBRID LLM CONFIG (Ollama & ChatAnywhere)
# ============================================================
from openai import OpenAI
from dotenv import load_dotenv
import httpx # Digunakan untuk cek ketersediaan API cepat

load_dotenv()

# --- KONFIGURASI ---
TEST_MODE = True  # SET KE TRUE UNTUK TESTING (MEMAKSA OLLAMA)

# ChatAnywhere Config
CA_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-8qopXoq0Gcn3YbFToEWt4nEOyo5rigUmXlFcWzitTQZGzROg")
CA_API_BASE = "https://api.chatanywhere.tech/v1"
CA_MODEL    = "gpt-3.5-turbo"

# Ollama Config
OLLAMA_BASE  = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3:8b" # Sesuaikan dengan model yang Anda pull

# Embedding tetap gunakan ChatAnywhere (atau ganti ke Ollama jika ingin full lokal)
EMBEDDING_MODEL_NAME = "mxbai-embed-large"

# Inisialisasi Clients
client_ca     = OpenAI(api_key=CA_API_KEY, base_url=CA_API_BASE)
client_ollama = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
client = client_ollama

def is_chatanywhere_available() -> bool:
    """Fungsi untuk mengecek apakah API ChatAnywhere bisa diakses dan token ada"""
    if TEST_MODE: return False # Paksa lokal jika sedang testing
    try:
        # Cek simpel dengan timeout rendah
        with httpx.Client(timeout=3.0) as check_client:
            response = check_client.get(f"{CA_API_BASE}/models", headers={"Authorization": f"Bearer {CA_API_KEY}"})
            return response.status_code == 200
    except:
        return False

# Penentuan model default awal
CURRENT_MODE = "OLLAMA (Testing)" if TEST_MODE else "CHATANYWHERE (Produksi)"
print(f"[LLM] Mode Aktif: {CURRENT_MODE}")
# ============================================================
# TIPE KOGNITIF (48 kombinasi)
# Format : {1-6}{P|T}{A|G}{I|R}
# ============================================================
VALID_COGNITIVE_TYPES: List[str] = [
    f"{n}{pt}{ag}{ir}"
    for n  in ["1","2","3","4","5","6"]
    for pt in ["P","T"]
    for ag in ["A","G"]
    for ir in ["I","R"]
]
DEFAULT_COGNITIVE_TYPE = "1PAR"


def cognitive_label(code: str) -> str:
    code = code.upper()
    if len(code) != 4:
        return f"Tipe Kognitif {code}"
    pt = "Praktis"    if code[1] == "P" else "Teoretis"
    ag = "Analitis"   if code[2] == "A" else "Global"
    ir = "Individual" if code[3] == "I" else "Relasional"
    return f"Level {code[0]} — {pt}, {ag}, {ir} ({code})"


# ============================================================
# NORMALISASI LATEX
# LLM kadang menulis [ ... ] atau $$ ... $$ alih-alih \[...\]
# Fungsi ini menstandarkan semua notasi ke format MathJax yang benar
# ============================================================
def normalize_latex(text: str) -> str:
    if not text:
        return text

    # Lindungi blok kode agar tidak ikut diproses
    code_blocks: List[str] = []
    def stash_code(m):
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks)-1}\x00"
    text = re.sub(r"```[\s\S]*?```|`[^`]+`", stash_code, text)

    # $$ ... $$ → \[ ... \]
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: r"\[" + m.group(1) + r"\]", text, flags=re.DOTALL)

    # $ ... $ (inline) → \( ... \)  — hati-hati agar tidak menangkap $$ yang sudah dikonversi
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", lambda m: r"\(" + m.group(1) + r"\)", text)

    # \[ spasi ... spasi \] yang sudah benar — biarkan
    # Pola LLM yang sering salah: "[" + spasi + konten LaTeX + spasi + "]"
    # Tapi hanya jika konten mengandung perintah LaTeX (\frac, \int, dll.)
    latex_cmd = r"\\(?:frac|int|sum|prod|lim|sqrt|left|right|begin|end|alpha|beta|gamma|delta|theta|lambda|mu|sigma|omega|pi|infty|partial|nabla|cdot|times|div|pm|leq|geq|neq|approx|equiv|in|subset|cup|cap|forall|exists|mathbb|mathbf|mathrm|text|overline|underline|hat|vec|bar|dot|ddot|tilde)"
    text = re.sub(
        r"(?<!\[)\[[\s\n]*(" + latex_cmd + r"[\s\S]*?)[\s\n]*\](?!\])",
        lambda m: r"\[" + m.group(1) + r"\]",
        text
    )

    # Kembalikan blok kode
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{i}\x00", block)

    return text


# ============================================================
# FAISS
# ============================================================
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="CSIPBLLM Personalized Learning",
    description="Sistem tutor adaptif berbasis kognitif dengan RAG.",
    version="2.0.0",
)

BASE_DIR      = os.path.dirname(__file__)
STATIC_DIR    = os.path.join(BASE_DIR, "static")
MATERIALS_DIR = os.path.join(BASE_DIR, "materials")
HISTORY_DIR   = os.path.join(BASE_DIR, "history_logs")
os.makedirs(HISTORY_DIR, exist_ok=True)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html tidak ditemukan"}, status_code=404)


@app.get("/cognitive-types", summary="Daftar semua kode tipe kognitif", tags=["Referensi"])
def list_cognitive_types():
    return {
        "cognitive_types": [
            {"code": ct, "label": cognitive_label(ct)}
            for ct in VALID_COGNITIVE_TYPES
        ]
    }


# ============================================================
# RAG GLOBALS
# ============================================================
RAG_CHUNK_MAX_CHARS = 1000
MAX_HISTORY_CHARS   = 1200

cognitive_indices: Dict[str, Dict[str, Any]] = {}
cognitive_loaded:  Dict[str, bool]           = {}
global_materials_index: List[Dict]           = []
global_faiss_index: Any                      = None
global_materials_loaded                      = False


# ============================================================
# EMBEDDINGS
# ============================================================
class GPTEmbeddings:
    def embed_query(self, text: str) -> List[float]:
        # Tentukan client mana yang dipakai untuk embedding
        if is_chatanywhere_available():
            active_client = client_ca
            # Pastikan model embedding di ChatAnywhere/OpenAI tersedia
            # Biasanya: "text-embedding-3-small" atau "text-embedding-ada-002"
            active_model = "text-embedding-3-small" 
        else:
            active_client = client_ollama
            active_model = EMBEDDING_MODEL_NAME

        try:
            resp = active_client.embeddings.create(
                model=active_model, 
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"[Embedding Error] Gagal menggunakan {active_model}: {e}")
            # Fallback dimensi (1536 untuk OpenAI, 1024 untuk mxbai-embed-large)
            dim = 1536 if "text-embedding" in active_model else 1024
            return [0.0] * dim

embeddings_model = GPTEmbeddings()


def _build_faiss(chunks: List[Dict]) -> Any:
    if faiss is None or not chunks:
        return None
    mat = np.stack([c["embedding"] for c in chunks]).astype("float32")
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return idx


def _embed_file(path: str, fname: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"[RAG] ⚠️  Tidak bisa baca {path}: {e}")
        return []
    if not text:
        return []
    out = []
    for i, chunk in enumerate([text[j:j+800] for j in range(0, len(text), 800)]):
        try:
            emb  = np.array(embeddings_model.embed_query(chunk), dtype="float32")
            norm = np.linalg.norm(emb)
            if norm: emb /= norm
            out.append({"embedding": emb, "text": chunk, "source": fname, "chunk_id": i})
        except Exception as e:
            print(f"[RAG] ❌ Error embedding {fname}#{i}: {e}")
            break
    return out


# ============================================================
# MUAT MATERI
# ============================================================
def load_cognitive_materials(code: str) -> None:
    code = code.upper()
    if cognitive_loaded.get(code): return
    if not os.path.isdir(MATERIALS_DIR):
        cognitive_loaded[code] = True; return
    print(f"[RAG] 🔍 Mengindeks materi untuk: {code}")
    chunks: List[Dict] = []
    for root, _, files in os.walk(MATERIALS_DIR):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md")): continue
            stem = os.path.splitext(fname)[0].upper()
            if stem == code or stem.startswith(code + "_"):
                chunks.extend(_embed_file(os.path.join(root, fname), fname))
    cognitive_indices[code] = {"chunks": chunks, "faiss": _build_faiss(chunks)}
    cognitive_loaded[code]  = True
    print(f"[RAG] ✅ {code}: {len(chunks)} chunk diindeks.")


def load_global_materials() -> None:
    global global_materials_index, global_faiss_index, global_materials_loaded
    if global_materials_loaded: return
    if not os.path.isdir(MATERIALS_DIR):
        global_materials_loaded = True; return
    print("[RAG] 🌐 Membangun indeks global fallback…")
    chunks: List[Dict] = []
    cog_set = set(VALID_COGNITIVE_TYPES)
    for root, _, files in os.walk(MATERIALS_DIR):
        for fname in files:
            if not fname.lower().endswith((".txt", ".md")): continue
            stem = os.path.splitext(fname)[0].upper()
            if any(stem == ct or stem.startswith(ct + "_") for ct in cog_set): continue
            chunks.extend(_embed_file(os.path.join(root, fname), fname))
    global_materials_index = chunks
    global_faiss_index     = _build_faiss(chunks)
    global_materials_loaded = True
    print(f"[RAG] 🌐 Indeks global: {len(chunks)} chunk.")


# ============================================================
# RAG RETRIEVAL
# ============================================================
def retrieve_relevant_chunks(query: str, code: str, k: int = 4) -> List[Dict]:
    code = code.upper()
    load_cognitive_materials(code)
    load_global_materials()

    q_emb = np.array(embeddings_model.embed_query(query), dtype="float32")
    norm  = np.linalg.norm(q_emb)
    if norm: q_emb /= norm

    def _search(chunks, index, top_k):
        if not chunks: return []
        if index is not None:
            D, I = index.search(q_emb.reshape(1,-1), top_k)
            return [{"text": chunks[int(i)]["text"], "source": chunks[int(i)]["source"], "score": float(s)}
                    for i, s in zip(I[0], D[0]) if i >= 0 and s > 0]
        scores = [float(np.dot(q_emb, c["embedding"])) for c in chunks]
        return [{"text": chunks[i]["text"], "source": chunks[i]["source"], "score": scores[i]}
                for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]

    cog  = cognitive_indices.get(code, {})
    hits = _search(cog.get("chunks",[]), cog.get("faiss"), k)
    if len(hits) < k:
        hits += _search(global_materials_index, global_faiss_index, k-len(hits))

    seen, out = set(), []
    for r in hits:
        key = (r["source"], r["text"][:80])
        if key not in seen:
            seen.add(key); out.append(r)
    return out[:k]


# ============================================================
# RIWAYAT SESI
# ============================================================
from langchain_community.chat_message_histories import ChatMessageHistory

session_histories:    Dict[str, Any] = {}
conversation_history: List[Dict]     = []

HISTORY_FILE_BASE      = "conversation_log"
HISTORY_FILE_PATH_JSON = os.path.join(HISTORY_DIR, f"{HISTORY_FILE_BASE}.json")
HISTORY_FILE_PATH_CSV  = os.path.join(HISTORY_DIR, f"{HISTORY_FILE_BASE}.csv")


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]


def format_history_as_text(history, max_chars: int = MAX_HISTORY_CHARS) -> str:
    if not getattr(history, "messages", None):
        return "Tidak ada riwayat sebelumnya."
    lines = []
    for msg in history.messages:
        role    = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        prefix  = "[Mahasiswa]" if role == "human" else "[Tutor]" if role == "ai" else "[Riwayat]"
        lines.append(f"{prefix} {content}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else "...\n" + text[-max_chars:]


def save_conversation_history_to_file() -> None:
    if not conversation_history: return
    with open(HISTORY_FILE_PATH_JSON, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=4, ensure_ascii=False)
    fieldnames = list(conversation_history[0].keys())
    with open(HISTORY_FILE_PATH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in conversation_history:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list,dict)) else v for k,v in row.items()})


# ============================================================
# GPT CALL
# ============================================================
SYSTEM_PROMPT = (
    "Kamu adalah tutor pendidikan tinggi di Indonesia. "
    "Selalu jawab dalam Bahasa Indonesia yang jelas dan akademis. "
    "Untuk persamaan matematika, SELALU gunakan format LaTeX MathJax: "
    "inline dengan \\(...\\) dan blok persamaan dengan \\[...\\]. "
    "JANGAN gunakan $...$ atau $$...$$. "
    "JANGAN gunakan [...] sebagai pengganti \\[...\\]."
)

def query_gpt(prompt: str, retries: int = 2, delay: int = 1) -> str:
    # 1. Tentukan Client & Model
    if is_chatanywhere_available():
        active_client = client_ca
        active_model  = CA_MODEL
        tag = "[GPT-CA]"
    else:
        active_client = client_ollama
        active_model  = OLLAMA_MODEL
        tag = "[OLLAMA]"

    for attempt in range(retries):
        try:
            resp = active_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"{tag} Response generated.")
            return normalize_latex(raw)
        except Exception as e:
            print(f"{tag} ❌ Percobaan {attempt+1}: {e}")
            # Jika CA gagal di tengah jalan, coba switch ke Ollama di percobaan berikutnya
            active_client = client_ollama
            active_model  = OLLAMA_MODEL
            time.sleep(delay)
            
    return "[ERROR] Semua layanan LLM (Ollama & ChatAnywhere) tidak tersedia."


# ============================================================
# DETEKSI KODE
# ============================================================
CODE_REGEX = re.compile(
    r"```[\s\S]*?```|(\bfor\b|\bwhile\b|\bif\b|\bdef\b|\bprint\b|\breturn\b|;|==)"
)
def is_code_like(text: str) -> bool:
    return bool(CODE_REGEX.search(text or ""))


# ============================================================
# EVALUASI KETAT (dua langkah)
# ============================================================
def strict_evaluate(answer: str, correct_answer: str, active_question: str,
                    context: str, history_txt: str, label: str,
                    cognitive_code: str) -> Tuple[bool, str]:
    # Jika ada active_question (pertanyaan followup spesifik), evaluasi berdasarkan itu.
    # Jika tidak ada, evaluasi berdasarkan kesesuaian dengan penjelasan tutor secara umum.
    if active_question and active_question.strip():
        evaluation_scope = f"""PERTANYAAN YANG SEDANG DIJAWAB:
{active_question}

KONTEKS MATERI (penjelasan tutor sebelumnya):
{correct_answer[:800]}"""
        task_instruction = """TUGAS PENILAIAN:
1. Fokus pada pertanyaan yang sedang dijawab di atas — BUKAN penjelasan tutor secara keseluruhan.
2. Hitung atau verifikasi kebenaran jawaban mahasiswa terhadap pertanyaan tersebut.
3. Untuk soal numerik/matematis: periksa apakah hasil akhirnya benar secara matematis.
4. Untuk soal konseptual: periksa apakah jawaban mencakup poin utama yang ditanyakan.
5. JANGAN menolak jawaban benar hanya karena singkat atau tidak menjelaskan proses."""
    else:
        evaluation_scope = f"""KUNCI / REFERENSI (penjelasan tutor):
{correct_answer[:800]}"""
        task_instruction = """TUGAS PENILAIAN:
1. Bandingkan jawaban mahasiswa dengan penjelasan tutor secara konseptual.
2. Jawaban BENAR jika mencakup konsep utama, meskipun dengan kata berbeda.
3. Jawaban SALAH jika konsep utama hilang, keliru, atau tidak relevan.
4. Jangan anggap benar hanya karena terdengar logis — harus sesuai kunci."""

    prompt = f"""Kamu adalah penilai jawaban yang ketat dan objektif untuk tutor universitas di Indonesia.

Tipe kognitif mahasiswa: {label}

Materi referensi (profil {cognitive_code}):
{context}

Riwayat percakapan:
{history_txt}

---
{evaluation_scope}

JAWABAN MAHASISWA:
{answer}
---

{task_instruction}

Tulis penjelasan singkat penilaian (3-4 kalimat), lalu pada baris terakhir tulis TEPAT salah satu:
HASIL: BENAR
HASIL: SALAH"""

    raw   = query_gpt(prompt)
    match = re.search(r"HASIL:\s*(BENAR|SALAH)", raw, re.IGNORECASE)
    if match:
        is_correct = match.group(1).upper() == "BENAR"
        reasoning  = re.sub(r"HASIL:\s*(BENAR|SALAH)", "", raw, flags=re.IGNORECASE).strip()
    else:
        is_correct = False
        reasoning  = raw.strip()
    return is_correct, reasoning


# ============================================================
# GENERATE PERTANYAAN LANJUTAN TEKNIKAL (studi kasus)
# Dipisah dari feedback agar fokus dan terkontrol
# ============================================================
def generate_technical_followup(original_question: str, tutor_reply: str,
                                 context: str, label: str) -> str:
    prompt = f"""Kamu adalah tutor universitas di Indonesia yang sedang merancang soal latihan teknikal.

Tipe kognitif mahasiswa: {label}

Topik yang baru dijelaskan:
{tutor_reply[:600]}

Materi referensi:
{context}

TUGAS:
Buat SATU pertanyaan lanjutan berbentuk studi kasus teknikal singkat.
Pertanyaan harus:
- Berbasis skenario nyata atau penerapan konkret (bukan definisi ulang konsep).
- Mendorong mahasiswa menerapkan konsep yang baru dipelajari pada situasi spesifik.
- Cukup spesifik sehingga ada jawaban yang benar/salah secara teknikal.
- Singkat: maksimal 3 kalimat total (skenario + pertanyaan).
- Diakhiri tanda tanya (?).

Contoh format yang BAIK:
"Sebuah array berisi [5, 3, 8, 1, 9, 2] dan kamu diminta mengurutkannya menggunakan Bubble Sort. Pada iterasi pertama, elemen mana yang akan berpindah posisi, dan mengapa?"

Contoh format yang BURUK (terlalu konseptual):
"Apa perbedaan antara Bubble Sort dan Selection Sort?"

Tulis HANYA pertanyaannya, tanpa penjelasan tambahan."""

    result = query_gpt(prompt).strip()
    # Ambil hanya kalimat terakhir jika ada banyak paragraf
    lines = [l.strip() for l in result.split("\n") if l.strip()]
    return lines[-1] if lines else result


# ============================================================
# MODEL REQUEST
# ============================================================
class ChatRequest(BaseModel):
    message:    str = Field(...,                    description="Pertanyaan atau pesan mahasiswa")
    cognitive:  str = Field(DEFAULT_COGNITIVE_TYPE, description="Kode tipe kognitif, misal '3TGR'")
    session_id: str = Field("default",              description="Identifikasi sesi")

    model_config = {"json_schema_extra": {"example": {
        "message": "Apa itu algoritma?", "cognitive": "2TAR", "session_id": "mahasiswa-01"
    }}}


class EvalRequest(BaseModel):
    answer:            str = Field(...,                    description="Jawaban mahasiswa")
    correct_answer:    str = Field(...,                    description="Jawaban referensi / kunci (penjelasan tutor)")
    active_question:   str = Field("",                    description="Pertanyaan spesifik yang sedang dijawab mahasiswa (followup atau pertanyaan awal)")
    wrong_count:       int = Field(0,                     description="Jumlah percobaan salah sejauh ini")
    cognitive:         str = Field(DEFAULT_COGNITIVE_TYPE, description="Kode tipe kognitif, misal '3TGR'")
    session_id:        str = Field("default",             description="Identifikasi sesi")

    model_config = {"json_schema_extra": {"example": {
        "answer": "32/3", "correct_answer": "Penjelasan tutor tentang integral...",
        "active_question": "Berapa luas area di bawah f(x) = -x^2 + 4x pada interval [0,4]?",
        "wrong_count": 0, "cognitive": "2TAR", "session_id": "mahasiswa-01"
    }}}


# ============================================================
# /chat
# ============================================================
@app.post("/chat", summary="Kirim pertanyaan dan terima respons tutor", tags=["Tutor"])
def chat_endpoint(req: ChatRequest):
    session_id     = req.session_id
    history        = get_session_history(session_id)
    cognitive_code = req.cognitive.upper()

    if cognitive_code not in VALID_COGNITIVE_TYPES:
        return JSONResponse(
            {"error": f"Tipe kognitif '{cognitive_code}' tidak valid. Lihat GET /cognitive-types."},
            status_code=422,
        )

    label       = cognitive_label(cognitive_code)
    rag_chunks  = retrieve_relevant_chunks(req.message, cognitive_code)
    context     = "\n\n".join(
        f"[{c['source']}]\n{c['text'][:RAG_CHUNK_MAX_CHARS]}" for c in rag_chunks
    ) or "Tidak ada konteks materi relevan."
    history_txt = format_history_as_text(history)
    check_understanding_lead = (
    "Di akhir penjelasan, kamu WAJIB menambahkan kalimat penutup untuk memicu evaluasi mandiri. "
    "Gunakan pola: 'Untuk memastikan kamu memahami konsep [Topik] ini, coba jelaskan dengan bahasamu sendiri "
    "bagaimana cara kerja [Bagian Spesifik], atau jawab pertanyaan studi kasus yang akan saya berikan di bawah ini.'"
)

    # ── PENJELASAN TUTOR (singkat, maks 3 poin) ───────────────
    if is_code_like(req.message):
        prompt = f"""Kamu adalah tutor Computational Thinking untuk mahasiswa universitas di Indonesia.
Tipe kognitif mahasiswa: {label}

Riwayat percakapan:
{history_txt}

Materi referensi (profil {cognitive_code}):
{context}

Pertanyaan mahasiswa (berupa kode):
{req.message}

INSTRUKSI WAJIB:
- Jawaban hanya boleh berasal dari materi referensi.
- Jangan menambahkan informasi dari pengetahuan umum.
- Gunakan istilah persis seperti materi.
- Jika materi menyebut daftar, tampilkan sesuai urutan materi.
- Analisis kode secara bertahap sesuai gaya kognitif mahasiswa.
- Jangan langsung memberikan jawaban final — arahkan mahasiswa untuk berpikir.
- Penjelasan 4 poin/paragraf pendek. Padat dan langsung ke inti.
- Gunakan \\(...\\) untuk matematika inline dan \\[...\\] untuk persamaan blok.
- {check_understanding_lead}
- JANGAN tambahkan pertanyaan di akhir — pertanyaan lanjutan akan dibuat terpisah."""
    else:
        prompt = f"""Kamu adalah tutor untuk mahasiswa universitas di Indonesia.
Tipe kognitif mahasiswa: {label}

Riwayat percakapan:
{history_txt}

Materi referensi (profil {cognitive_code}):
{context}

Pertanyaan mahasiswa:
{req.message}

INSTRUKSI WAJIB:
- Jawaban hanya boleh berasal dari materi referensi.
- Jangan menambahkan informasi dari pengetahuan umum.
- Gunakan istilah persis seperti materi.
- Jika materi menyebut daftar, tampilkan sesuai urutan materi.
- Jelaskan konsep sesuai gaya kognitif mahasiswa ({label}).
- Gunakan contoh konkret yang relevan dengan konteks Indonesia.
- Jangan langsung memberikan jawaban final — bantu mahasiswa memahami konsep.
- Penjelasan 4 poin/paragraf pendek. Padat dan langsung ke inti.
- Gunakan \\(...\\) untuk matematika inline dan \\[...\\] untuk persamaan blok.
- {check_understanding_lead}
- JANGAN tambahkan pertanyaan di akhir — pertanyaan lanjutan akan dibuat terpisah."""


    reply = query_gpt(prompt)

    # ── PERTANYAAN LANJUTAN TEKNIKAL (terpisah) ───────────────
    followup = generate_technical_followup(req.message, reply, context, label)

    history.add_user_message(req.message)
    history.add_ai_message(reply)

    conversation_history.append({
        "timestamp":         datetime.now().isoformat(),
        "session_id":        session_id,
        "cognitive":         cognitive_code,
        "user_message":      req.message,
        "reply":             reply,
        "followup_question": followup,
    })
    save_conversation_history_to_file()

    return {
        "reply":             reply,
        "followup_question": followup,
        "cognitive":         cognitive_code,
        "session_id":        session_id,
    }


# ============================================================
# /evaluate
# ============================================================
@app.post("/evaluate", summary="Evaluasi jawaban mahasiswa dengan umpan balik adaptif", tags=["Tutor"])
def evaluate_answer(req: EvalRequest):
    session_id     = req.session_id
    history        = get_session_history(session_id)
    cognitive_code = req.cognitive.upper()
    if cognitive_code not in VALID_COGNITIVE_TYPES:
        cognitive_code = DEFAULT_COGNITIVE_TYPE
    label  = cognitive_label(cognitive_code)
    answer = req.answer.strip()

    # Level scaffolding — detail bertambah seiring percobaan salah
    scaffold = {
        0: ("Evaluasi Awal",
            "Tunjukkan bagian yang kurang tepat secara umum (1-2 kalimat). "
            "Jangan jelaskan terlalu banyak — cukup arahkan."),
        1: ("Petunjuk Terarah",
            "Berikan petunjuk spesifik tentang konsep yang salah (2-3 kalimat). "
            "Sebutkan aspek mana yang perlu diperbaiki tanpa memberi jawaban langsung."),
        2: ("Dukungan Remedial",
            "Uraikan konsep yang salah secara bertahap dalam 3 poin singkat. "
            "Boleh memberi contoh kecil untuk memperjelas."),
    }
    default_scaffold = (
        "Panduan Langkah-demi-Langkah",
        "Jelaskan ulang konsep secara lengkap dengan analogi sederhana, maksimal 3 paragraf. "
        "Pastikan mahasiswa memahami di mana letak kesalahannya."
    )
    hint_level, feedback_instruction = scaffold.get(req.wrong_count, default_scaffold)

    # RAG
    rag_chunks = retrieve_relevant_chunks(
        f"Kunci: {req.correct_answer}. Jawaban mahasiswa: {answer}", cognitive_code
    )
    context     = "\n\n".join(
        f"[{c['source']}]\n{c['text'][:RAG_CHUNK_MAX_CHARS]}" for c in rag_chunks
    ) or "Tidak ada konteks materi."
    history_txt = format_history_as_text(history)

    # ── EVALUASI KETAT ─────────────────────────────────────────
    is_correct, evaluation_reasoning = strict_evaluate(
        answer, req.correct_answer, req.active_question,
        context, history_txt, label, cognitive_code
    )

    # ── UMPAN BALIK ────────────────────────────────────────────
    if is_correct:
        # Jawaban benar → konfirmasi singkat, tanpa pujian berlebihan, tanpa LLM call tambahan
        if req.active_question and req.active_question.strip():
            feedback_clean = f"✅ Jawaban kamu benar."
        else:
            feedback_clean = f"✅ Jawaban kamu benar."
        followup = ""

    else:
        feedback_prompt = f"""Kamu adalah tutor universitas di Indonesia.
Tipe kognitif mahasiswa: {label}

Materi referensi (profil {cognitive_code}):
{context}

Riwayat percakapan:
{history_txt}

Mahasiswa menjawab dengan SALAH. Jawaban mereka:
"{answer}"

Kunci jawaban (JANGAN ungkapkan langsung):
"{req.correct_answer}"

Penilaian sistem:
{evaluation_reasoning}

Level bantuan saat ini: {hint_level}
Instruksi umpan balik: {feedback_instruction}

Gunakan \\(...\\) untuk matematika inline dan \\[...\\] untuk persamaan blok.
JANGAN tambahkan pertanyaan di akhir — pertanyaan lanjutan akan dibuat terpisah."""

        feedback_clean = query_gpt(feedback_prompt)

        # Pertanyaan lanjutan teknikal untuk iterasi berikutnya
        followup = generate_technical_followup(
            req.correct_answer, feedback_clean, context, label
        )

    history.add_user_message(f"[JAWABAN MAHASISWA] {answer}")
    history.add_ai_message(f"[UMPAN BALIK TUTOR] {feedback_clean}")

    return {
        "is_correct":        is_correct,
        "feedback":          feedback_clean,
        "hint_level":        hint_level,
        "followup_question": followup,
        "cognitive":         cognitive_code,
        "session_id":        session_id,
    }


# ============================================================
# /history
# ============================================================
@app.get("/history", summary="Ambil riwayat percakapan", tags=["Riwayat"])
def get_history(format: str = "json"):
    if not conversation_history:
        return {"history": []}
    if format == "json":
        return {"history": conversation_history}
    lines = []
    for i, conv in enumerate(conversation_history, 1):
        lines += [
            f"[Percakapan {i}]",
            f"Profil Kognitif : {conv.get('cognitive', '-')}",
            f"Pertanyaan      : {conv['user_message']}",
            f"Jawaban         :\n{conv['reply']}",
            "-" * 60,
        ]
    return {"data": "\n".join(lines)}


# ============================================================
# /download-history
# ============================================================
@app.get("/download-history", summary="Unduh riwayat percakapan sebagai JSON atau CSV", tags=["Riwayat"])
def download_history(format: str = "json"):
    if format.lower() == "csv":
        path, media, fname = HISTORY_FILE_PATH_CSV,  "text/csv",        f"{HISTORY_FILE_BASE}.csv"
    else:
        path, media, fname = HISTORY_FILE_PATH_JSON, "application/json", f"{HISTORY_FILE_BASE}.json"
    if not os.path.exists(path):
        return JSONResponse({"error": "Belum ada riwayat percakapan."}, status_code=404)
    return FileResponse(path, media_type=media, filename=fname)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    load_global_materials()
    print("\n🚀  Server  →  http://127.0.0.1:8000")
    print(f"🧠  Mode Testing: {'AKTIF (Lokal Only)' if TEST_MODE else 'NON-AKTIF (Hybrid)'}")
    print(f"🤖  Ollama Model: {OLLAMA_MODEL}")
    print(f"📚  Tipe kognitif: {len(VALID_COGNITIVE_TYPES)}")
    uvicorn.run("ollamaapi:app", host="127.0.0.1", port=8000, reload=True)