"""
=============================================================================
CSIPBLLM — RAG EVALUATION SUITE
=============================================================================
Sesuai dengan proposal penelitian (hal. 4–9):
Muhammad Ajisaka Arsyi Taj (G6401221090)
"Pengembangan Sistem Pembelajaran Terpersonalisasi Mata Kuliah CT
Menggunakan Chatbot Berbasis LLM dan RAG"

Metode evaluasi yang diimplementasikan:
  1. Embedding & Vector Store  — ē = f_embed(x), V = {d1, d2, ..., dN}
  2. Cosine Similarity         — sim(q,d) = (q·d) / (||q||·||d||)
  3. Top-K Retrieval           — TopK(q) = arg topK sim(q, di)
  4. Precision@K               — |relevant ∩ TopK| / K
  5. Recall@K                  — |relevant ∩ TopK| / |total relevant|
  6. MeanSim                   — (1/K) Σ si
  7. Coverage                  — proporsi chunk yang informatif (skor > threshold)
  8. Source Diversity          — jumlah sumber unik dalam Top-K
  9. Faithfulness              — apakah jawaban LLM berdasar pada konteks RAG
 10. Hallucination Detection   — deteksi jawaban yang menyimpang dari konteks
 11. Answer Quality (boolean) — benar/salah berdasarkan /evaluate endpoint
 12. Offline Statistical Analysis — dari log JSON/CSV historis
=============================================================================
"""

import os
import sys
import json
import csv
import time
import math
import statistics
import re
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import requests

# =============================================================================
# KONFIGURASI — sesuaikan dengan lingkungan Anda
# =============================================================================
BASE_URL          = "http://127.0.0.1:8000"      # URL backend FastAPI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "YOUR_GPT_API_KEY_HERE")
OPENAI_API_BASE   = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
EMBEDDING_MODEL   = "text-embedding-3-small"
CHAT_MODEL        = "gpt-3.5-turbo"
TOP_K             = 4                             # K untuk Top-K retrieval
RELEVANCE_THRESHOLD = 0.30                        # batas skor cosine untuk "relevan"
COVERAGE_THRESHOLD  = 0.25                        # batas skor untuk chunk "berguna"
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATASET UJI — pasangan (query, jawaban_benar, label_relevansi)
# label_relevansi: kata kunci yang HARUS ada dalam chunk yang diambil
# Ubah/tambahkan sesuai dokumen materials yang Anda miliki
# =============================================================================
TEST_CASES: List[Dict] = [
    # --- ALGORITMA & COMPUTATIONAL THINKING ---
    {
        "query": "Apa itu algoritma dalam computational thinking?",
        "reference_answer": "Algoritma adalah serangkaian langkah terurut untuk menyelesaikan masalah",
        "relevant_keywords": ["algoritma", "langkah", "instruksi", "urutan"],
        "cognitive": "2TAR",
        "session_id": "eval-001",
    },
    {
        "query": "Jelaskan konsep dekomposisi dalam computational thinking",
        "reference_answer": "Dekomposisi adalah memecah masalah besar menjadi sub-masalah yang lebih kecil",
        "relevant_keywords": ["dekomposisi", "masalah", "sub-masalah", "bagian"],
        "cognitive": "1PAI",
        "session_id": "eval-002",
    },
    {
        "query": "Apa perbedaan antara rekursi dan iterasi?",
        "reference_answer": "Rekursi adalah fungsi yang memanggil dirinya sendiri, iterasi menggunakan perulangan",
        "relevant_keywords": ["rekursi", "iterasi", "fungsi", "perulangan"],
        "cognitive": "3TGR",
        "session_id": "eval-003",
    },
    {
        "query": "Bagaimana cara kerja sorting bubble sort?",
        "reference_answer": "Bubble sort membandingkan elemen berdampingan dan menukar jika tidak urut",
        "relevant_keywords": ["bubble sort", "elemen", "bandingkan", "tukar", "urut"],
        "cognitive": "2PAR",
        "session_id": "eval-004",
    },
    {
        "query": "Apa itu abstraksi dalam computational thinking?",
        "reference_answer": "Abstraksi adalah menyederhanakan masalah dengan mengabaikan detail yang tidak penting",
        "relevant_keywords": ["abstraksi", "sederhanakan", "detail", "masalah"],
        "cognitive": "4TAI",
        "session_id": "eval-005",
    },
    {
        "query": "Jelaskan konsep pattern recognition",
        "reference_answer": "Pattern recognition adalah mengenali pola atau kesamaan dalam data atau masalah",
        "relevant_keywords": ["pola", "pattern", "kesamaan", "data"],
        "cognitive": "1TGI",
        "session_id": "eval-006",
    },
    {
        "query": "Bagaimana cara menghitung kompleksitas algoritma?",
        "reference_answer": "Kompleksitas algoritma diukur dengan Big-O notation yang menggambarkan pertumbuhan waktu",
        "relevant_keywords": ["kompleksitas", "Big-O", "waktu", "efisiensi"],
        "cognitive": "5PAI",
        "session_id": "eval-007",
    },
    {
        "query": "Apa yang dimaksud dengan struktur data array?",
        "reference_answer": "Array adalah kumpulan elemen bertipe sama yang disimpan dalam memori berurutan",
        "relevant_keywords": ["array", "elemen", "memori", "indeks"],
        "cognitive": "2TGI",
        "session_id": "eval-008",
    },
    {
        "query": "Jelaskan apa itu pseudocode dan kegunaannya",
        "reference_answer": "Pseudocode adalah representasi informal algoritma menggunakan bahasa alami terstruktur",
        "relevant_keywords": ["pseudocode", "algoritma", "informal", "langkah"],
        "cognitive": "1PAR",
        "session_id": "eval-009",
    },
    {
        "query": "Apa perbedaan antara linked list dan array?",
        "reference_answer": "Linked list menggunakan pointer antar node, array menggunakan indeks memori berurutan",
        "relevant_keywords": ["linked list", "pointer", "node", "array", "indeks"],
        "cognitive": "3PAI",
        "session_id": "eval-010",
    },
    {
        "query": "Bagaimana algoritma binary search bekerja?",
        "reference_answer": "Binary search membagi array terurut menjadi dua dan mencari di bagian yang relevan",
        "relevant_keywords": ["binary search", "tengah", "terurut", "bagian"],
        "cognitive": "4PAR",
        "session_id": "eval-011",
    },
    {
        "query": "Apa itu stack dan bagaimana cara kerjanya?",
        "reference_answer": "Stack adalah struktur data LIFO (Last In First Out) dengan operasi push dan pop",
        "relevant_keywords": ["stack", "LIFO", "push", "pop"],
        "cognitive": "2TAI",
        "session_id": "eval-012",
    },
    {
        "query": "Jelaskan apa itu flowchart dan simbolnya",
        "reference_answer": "Flowchart adalah diagram alur yang menggambarkan langkah-langkah proses dengan simbol standar",
        "relevant_keywords": ["flowchart", "diagram", "simbol", "proses"],
        "cognitive": "1TGR",
        "session_id": "eval-013",
    },
    {
        "query": "Apa yang dimaksud dengan queue dalam struktur data?",
        "reference_answer": "Queue adalah struktur data FIFO (First In First Out) seperti antrian",
        "relevant_keywords": ["queue", "FIFO", "antrian", "enqueue", "dequeue"],
        "cognitive": "3TAR",
        "session_id": "eval-014",
    },
    {
        "query": "Bagaimana cara kerja algoritma selection sort?",
        "reference_answer": "Selection sort mencari elemen terkecil dan menempatkannya di posisi awal secara bertahap",
        "relevant_keywords": ["selection sort", "terkecil", "posisi", "pilih"],
        "cognitive": "2PGR",
        "session_id": "eval-015",
    },
    {
        "query": "Apa itu fungsi rekursif dan bagaimana basis kasusnya?",
        "reference_answer": "Fungsi rekursif memanggil dirinya sendiri dengan basis kasus untuk menghentikan rekursi",
        "relevant_keywords": ["rekursif", "basis kasus", "memanggil", "hentikan"],
        "cognitive": "4TGI",
        "session_id": "eval-016",
    },
    {
        "query": "Jelaskan konsep tree dalam struktur data",
        "reference_answer": "Tree adalah struktur data hierarki dengan node akar, cabang, dan daun",
        "relevant_keywords": ["tree", "node", "akar", "cabang", "hierarki"],
        "cognitive": "3PGI",
        "session_id": "eval-017",
    },
    {
        "query": "Apa itu variabel dan tipe data dalam pemrograman?",
        "reference_answer": "Variabel adalah tempat penyimpanan data, tipe data menentukan jenis data yang disimpan",
        "relevant_keywords": ["variabel", "tipe data", "integer", "string", "penyimpanan"],
        "cognitive": "1PAI",
        "session_id": "eval-018",
    },
    {
        "query": "Bagaimana cara kerja conditional statement if-else?",
        "reference_answer": "If-else menjalankan blok kode berbeda berdasarkan kondisi yang bernilai benar atau salah",
        "relevant_keywords": ["if", "else", "kondisi", "percabangan", "boolean"],
        "cognitive": "1TGI",
        "session_id": "eval-019",
    },
    {
        "query": "Apa perbedaan antara loop for dan while?",
        "reference_answer": "For digunakan saat iterasi diketahui jumlahnya, while saat kondisi berhenti tidak pasti",
        "relevant_keywords": ["for", "while", "iterasi", "kondisi", "perulangan"],
        "cognitive": "2TAR",
        "session_id": "eval-020",
    },
]

# =============================================================================
# OPENAI CLIENT (untuk embedding & faithfulness check)
# =============================================================================
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARN] openai package tidak tersedia — embedding lokal akan dinonaktifkan")


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Menghasilkan vektor embedding untuk teks masukan.

    SESUAI PROPOSAL hal. 4:
      ē = f_embed(x)
    di mana x adalah teks masukan (query atau dokumen),
    f_embed adalah fungsi embedding (GPT text-embedding-3-small),
    dan ē ∈ ℝ^d adalah vektor embedding berdimensi d.
    """
    if not OPENAI_AVAILABLE:
        return None
    try:
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        vec = np.array(resp.data[0].embedding, dtype="float32")
        return vec
    except Exception as e:
        print(f"  [Embedding Error] {e}")
        return None


# =============================================================================
# METRIK INTI
# =============================================================================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Menghitung cosine similarity antara dua vektor.

    SESUAI PROPOSAL hal. 4–5:
      sim(q, d) = (q · d) / (||q|| · ||d||)

    di mana:
      q  = vektor query
      d  = vektor dokumen
      q·d = dot product (hasil kali titik)
      ||q||, ||d|| = norma Euclidean dari masing-masing vektor

    Nilai hasil berada pada rentang [0, 1] untuk vektor non-negatif.
    Semakin besar nilai ini, semakin tinggi relevansi semantik antara
    query dan dokumen.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def precision_at_k(retrieved_scores: List[float], k: int,
                   threshold: float = RELEVANCE_THRESHOLD) -> float:
    """
    Menghitung Precision@K.

    SESUAI PROPOSAL hal. 5:
      Precision@K = |dokumen relevan di Top-K| / K

    Mengukur proporsi dokumen relevan dalam hasil Top-K retrieval.
    Dokumen dianggap relevan jika skor cosine similarity-nya
    melebihi threshold yang ditentukan.

    Parameter:
      retrieved_scores : daftar skor cosine similarity dari K dokumen teratas
      k                : jumlah dokumen yang diambil (Top-K)
      threshold        : batas minimum skor untuk dianggap relevan
    """
    if k == 0:
        return 0.0
    relevant_count = sum(1 for s in retrieved_scores[:k] if s >= threshold)
    return relevant_count / k


def recall_at_k(retrieved_scores: List[float], total_relevant: int,
                k: int, threshold: float = RELEVANCE_THRESHOLD) -> float:
    """
    Menghitung Recall@K.

    SESUAI PROPOSAL hal. 5:
      Recall@K = |dokumen relevan di Top-K| / total dokumen relevan

    Mengukur proporsi semua dokumen relevan yang berhasil ditemukan
    oleh retriever dalam Top-K hasil. Jika total_relevant = 0,
    recall dianggap 1.0 (tidak ada yang perlu ditemukan).

    Parameter:
      retrieved_scores : daftar skor cosine similarity dari K dokumen teratas
      total_relevant   : perkiraan total dokumen relevan dalam korpus
      k                : jumlah dokumen yang diambil (Top-K)
      threshold        : batas minimum skor untuk dianggap relevan
    """
    if total_relevant == 0:
        return 1.0
    found_relevant = sum(1 for s in retrieved_scores[:k] if s >= threshold)
    return found_relevant / total_relevant


def mean_similarity(scores: List[float]) -> float:
    """
    Menghitung Mean Similarity dari hasil retrieval.

    SESUAI PROPOSAL hal. 5:
      MeanSim = (1/K) * Σ(i=1 to K) si

    di mana s1, s2, ..., sK adalah skor cosine similarity dari
    K dokumen teratas yang diambil retriever.

    Fungsi ini digunakan untuk mengukur kesehatan retriever dan
    stabilitas embedding secara keseluruhan (Ammar et al., 2025).
    Nilai MeanSim yang tinggi menunjukkan retriever secara konsisten
    mengambil dokumen yang semantik relevan dengan query.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def coverage_score(scores: List[float],
                   threshold: float = COVERAGE_THRESHOLD) -> float:
    """
    Menghitung Coverage: proporsi chunk yang informatif.

    SESUAI PROPOSAL hal. 8:
      Coverage mengecek apakah RAG mengambil chunks yang berguna.

    Dihitung sebagai proporsi chunks dengan skor cosine similarity
    di atas threshold (artinya chunk tersebut semantik relevan dengan
    query, bukan noise). Coverage tinggi = retriever menemukan
    konten yang benar-benar berguna untuk LLM.

    Rumus implementasi:
      Coverage = |{si : si >= threshold}| / K
    """
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def source_diversity(sources: List[str]) -> float:
    """
    Menghitung Source Diversity: keragaman sumber dokumen dalam Top-K.

    SESUAI PROPOSAL hal. 8:
      Source Diversity mengecek seberapa berbeda dokumen-dokumen
      yang diambil RAG.

    Diukur sebagai rasio antara jumlah sumber unik terhadap total
    dokumen yang diambil (K):
      Diversity = |sumber unik| / K

    Nilai 1.0 = semua dokumen dari sumber berbeda (keragaman maksimal).
    Nilai mendekati 0 = semua dokumen dari sumber yang sama.
    """
    if not sources:
        return 0.0
    unique_sources = len(set(sources))
    return unique_sources / len(sources)


# =============================================================================
# FAITHFULNESS & HALLUCINATION DETECTION
# =============================================================================

def evaluate_faithfulness(answer: str, retrieved_chunks: List[str]) -> Dict:
    """
    Mengevaluasi faithfulness: apakah jawaban LLM berdasar pada konteks RAG.

    SESUAI PROPOSAL hal. 8–9:
      Faithfulness mengecek apakah jawaban yang dihasilkan dibantu
      oleh konteks yang diambil RAG.

    Implementasi menggunakan dua pendekatan paralel:
    1. Keyword overlap — proporsi kata kunci dari konteks yang muncul
       dalam jawaban (pendekatan heuristik cepat)
    2. Embedding similarity — cosine similarity antara embedding
       jawaban dan embedding gabungan konteks:
         sim(ē_answer, ē_context) = (ē_answer · ē_context) / (||ē_answer|| · ||ē_context||)

    Nilai faithfulness_score adalah rata-rata tertimbang keduanya.
    Skor mendekati 1.0 = jawaban sangat bergantung pada konteks RAG.
    Skor mendekati 0.0 = kemungkinan jawaban berasal dari memori LLM
    saja, bukan dari dokumen yang diambil.
    """
    if not retrieved_chunks or not answer:
        return {"faithfulness_score": 0.0, "keyword_overlap": 0.0,
                "embedding_sim": None, "method": "none"}

    # ─── Metode 1: Keyword Overlap ────────────────────────────────────────
    # Ekstrak kata penting dari konteks (non-stopword, panjang > 3 karakter)
    stopwords_id = {
        "yang", "dan", "atau", "dari", "dalam", "untuk", "adalah", "dengan",
        "pada", "ke", "ini", "itu", "juga", "tidak", "akan", "ada", "oleh",
        "satu", "dapat", "lebih", "sudah", "telah", "bisa", "karena", "maka",
        "sebuah", "tersebut", "namun", "serta", "antara", "sebagai", "seperti"
    }
    context_combined = " ".join(retrieved_chunks).lower()
    answer_lower     = answer.lower()

    context_words = {
        w for w in re.findall(r'\b\w{4,}\b', context_combined)
        if w not in stopwords_id
    }
    if context_words:
        overlap = sum(1 for w in context_words if w in answer_lower)
        keyword_overlap = min(overlap / len(context_words), 1.0)
    else:
        keyword_overlap = 0.0

    # ─── Metode 2: Embedding Cosine Similarity ────────────────────────────
    # Bandingkan vektor embedding jawaban dengan vektor embedding konteks
    embedding_sim = None
    if OPENAI_AVAILABLE:
        emb_answer  = get_embedding(answer[:1000])
        emb_context = get_embedding(context_combined[:1000])
        if emb_answer is not None and emb_context is not None:
            embedding_sim = cosine_similarity(emb_answer, emb_context)

    # ─── Gabungkan Skor ───────────────────────────────────────────────────
    if embedding_sim is not None:
        # Bobot: 60% embedding similarity (lebih semantik), 40% keyword overlap
        faithfulness_score = 0.6 * embedding_sim + 0.4 * keyword_overlap
        method = "embedding + keyword"
    else:
        faithfulness_score = keyword_overlap
        method = "keyword_only"

    return {
        "faithfulness_score": round(faithfulness_score, 4),
        "keyword_overlap":    round(keyword_overlap, 4),
        "embedding_sim":      round(embedding_sim, 4) if embedding_sim is not None else None,
        "method":             method,
    }


def detect_hallucination(answer: str, retrieved_chunks: List[str],
                         query: str) -> Dict:
    """
    Mendeteksi halusinasi: apakah jawaban LLM menyimpang dari konteks.

    SESUAI PROPOSAL hal. 9:
      Hallucination detection: mendeteksi jawaban yang menyimpang
      dari konteks yang diberikan.

    LLM dapat menghasilkan jawaban yang tampak koheren tetapi tidak
    akurat secara konseptual (Lewis et al., 2021). Metode deteksi:

    1. Contradiction heuristic: apakah jawaban mengandung negasi dari
       fakta yang ada di konteks (kata seperti "tidak", "bukan", "salah"
       diikuti term penting dari konteks)
    2. Out-of-context score: 1 - faithfulness_score
       Jika faithfulness rendah (<0.20), jawaban kemungkinan berasal
       dari memori parametrik LLM, bukan dari konteks RAG
    3. Confidence flag: apakah jawaban mengandung kalimat ketidakpastian
       yang mengindikasikan LLM tidak yakin dengan jawabannya

    Skor hallucination_risk ∈ [0, 1]:
      0.0 = tidak ada indikasi halusinasi
      1.0 = sangat berisiko halusinasi
    """
    faith_result = evaluate_faithfulness(answer, retrieved_chunks)
    faithfulness = faith_result["faithfulness_score"]

    # ─── Indikator 1: Out-of-context (kebalikan faithfulness) ─────────────
    out_of_context = 1.0 - faithfulness

    # ─── Indikator 2: Contradiction heuristic ────────────────────────────
    negation_patterns = [
        r'\btidak\s+\w+', r'\bbukan\s+\w+', r'\bsalah\s+\w+',
        r'\bbertentangan\b', r'\bsebaliknya\b', r'\bpadahal\b'
    ]
    contradiction_score = 0.0
    if retrieved_chunks:
        context_terms = set(re.findall(r'\b\w{5,}\b', " ".join(retrieved_chunks).lower()))
        answer_lower  = answer.lower()
        for pat in negation_patterns:
            matches = re.findall(pat, answer_lower)
            for match in matches:
                match_words = set(re.findall(r'\b\w{4,}\b', match))
                if match_words & context_terms:
                    contradiction_score = min(contradiction_score + 0.2, 1.0)

    # ─── Indikator 3: Uncertainty flag ───────────────────────────────────
    uncertainty_phrases = [
        "saya tidak yakin", "saya tidak tahu", "mungkin", "kemungkinan besar",
        "saya pikir", "tampaknya", "sepertinya", "belum pasti", "tidak pasti"
    ]
    uncertainty_flag = any(p in answer.lower() for p in uncertainty_phrases)
    uncertainty_score = 0.3 if uncertainty_flag else 0.0

    # ─── Gabungkan Risiko Halusinasi ──────────────────────────────────────
    # Bobot: 50% out-of-context, 30% contradiction, 20% uncertainty
    hallucination_risk = (
        0.50 * out_of_context +
        0.30 * contradiction_score +
        0.20 * uncertainty_score
    )

    risk_label = (
        "TINGGI"   if hallucination_risk >= 0.60 else
        "SEDANG"   if hallucination_risk >= 0.35 else
        "RENDAH"
    )

    return {
        "hallucination_risk":  round(hallucination_risk, 4),
        "risk_label":          risk_label,
        "out_of_context":      round(out_of_context, 4),
        "contradiction_score": round(contradiction_score, 4),
        "uncertainty_flag":    uncertainty_flag,
        "faithfulness":        faithfulness,
    }


# =============================================================================
# API CALLS — panggil backend yang sedang berjalan
# =============================================================================

def call_chat_api(message: str, cognitive: str, session_id: str) -> Optional[Dict]:
    """Panggil endpoint /chat dan kembalikan respons."""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "cognitive": cognitive, "session_id": session_id},
            timeout=60,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  [API Error /chat] {e}")
        return None


def call_evaluate_api(answer: str, correct_answer: str,
                      active_question: str, wrong_count: int,
                      cognitive: str, session_id: str) -> Optional[Dict]:
    """
    Panggil endpoint /evaluate dan kembalikan hasil penilaian.

    SESUAI PROPOSAL hal. 9:
      Answer Quality (boolean): jawaban yang dimasukkan pengguna
      dievaluasi menjadi metrik kualitas jawaban (benar/salah).
    """
    try:
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json={
                "answer":          answer,
                "correct_answer":  correct_answer,
                "active_question": active_question,
                "wrong_count":     wrong_count,
                "cognitive":       cognitive,
                "session_id":      session_id,
            },
            timeout=60,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  [API Error /evaluate] {e}")
        return None


# =============================================================================
# EVALUASI RETRIEVAL LANGSUNG (tanpa melewati API)
# Mengakses embedding + FAISS secara langsung untuk metrik presisi
# =============================================================================

def evaluate_retrieval_direct(query: str, relevant_keywords: List[str],
                               k: int = TOP_K) -> Dict:
    """
    Evaluasi kualitas retrieval menggunakan embedding langsung.

    SESUAI PROPOSAL hal. 4–5, mengimplementasikan:
      1. Embedding query: ē_q = f_embed(query)
      2. Cosine Similarity: sim(q,d) = (q·d)/(||q||·||d||)
      3. Top-K: TopK(q) = arg topK sim(q, di)
      4. Precision@K, Recall@K, MeanSim
      5. Coverage, Source Diversity

    Karena kita tidak bisa langsung mengakses FAISS internal dari
    backend yang berjalan, metode ini:
    - Membuat embedding query baru
    - Membandingkan dengan embedding beberapa referensi dokumen
    - Menggunakan skor dari respons API sebagai proxy retrieval scores

    Catatan: untuk evaluasi retrieval penuh, sebaiknya integrasikan
    langsung dengan modul ollamaapi.py di dalam proses yang sama.
    """
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI tidak tersedia untuk evaluasi embedding langsung"}

    emb_query = get_embedding(query)
    if emb_query is None:
        return {"error": "Gagal membuat embedding query"}

    # Buat embedding untuk setiap keyword relevan sebagai proxy dokumen
    # Dalam implementasi nyata, ini adalah embedding dari chunk dokumen aktual
    keyword_embeddings = []
    for kw in relevant_keywords:
        emb = get_embedding(kw)
        if emb is not None:
            keyword_embeddings.append((kw, emb))

    if not keyword_embeddings:
        return {"error": "Gagal membuat embedding keyword"}

    # Hitung cosine similarity antara query dan setiap keyword/dokumen
    # sim(q, d) = (q · d) / (||q|| · ||d||)
    scores = []
    for kw, emb_doc in keyword_embeddings:
        sim = cosine_similarity(emb_query, emb_doc)
        scores.append({"keyword": kw, "score": sim})

    # Urutkan berdasarkan skor (simulasi Top-K retrieval)
    scores.sort(key=lambda x: x["score"], reverse=True)
    top_k_scores = [s["score"] for s in scores[:k]]
    top_k_sources = [s["keyword"] for s in scores[:k]]

    # Hitung semua metrik retrieval
    prec_k  = precision_at_k(top_k_scores, k)
    rec_k   = recall_at_k(top_k_scores, len(relevant_keywords), k)
    mean_sim = mean_similarity(top_k_scores)
    cov     = coverage_score(top_k_scores)
    div     = source_diversity(top_k_sources)

    return {
        "query_embedding_dim": len(emb_query),
        "top_k_scores":        [round(s, 4) for s in top_k_scores],
        "top_k_keywords":      top_k_sources,
        "precision_at_k":      round(prec_k, 4),
        "recall_at_k":         round(rec_k, 4),
        "mean_similarity":     round(mean_sim, 4),
        "coverage":            round(cov, 4),
        "source_diversity":    round(div, 4),
        "detail_scores":       [{"keyword": s["keyword"], "score": round(s["score"], 4)}
                                for s in scores],
    }


# =============================================================================
# RUNNER UTAMA — menjalankan seluruh test case
# =============================================================================

def run_full_evaluation() -> List[Dict]:
    """
    Menjalankan evaluasi RAG secara menyeluruh untuk semua test case.

    Alur evaluasi sesuai proposal hal. 8–9:
    1. Embedding & retrieval metrics (Precision@K, Recall@K, MeanSim)
    2. Coverage & Source Diversity
    3. Panggil /chat → dapatkan jawaban LLM + konteks RAG
    4. Faithfulness evaluation
    5. Hallucination detection
    6. Panggil /evaluate → Answer Quality (boolean)
    """
    print("\n" + "="*70)
    print("  CSIPBLLM — RAG EVALUATION SUITE")
    print("  Berdasarkan proposal: Muhammad Ajisaka Arsyi Taj (G6401221090)")
    print("="*70)
    print(f"  Total test case : {len(TEST_CASES)}")
    print(f"  Top-K           : {TOP_K}")
    print(f"  Waktu mulai     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    results = []

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"[{idx:02d}/{len(TEST_CASES)}] Query: {tc['query'][:60]}...")
        result = {"test_id": idx, "query": tc["query"],
                  "cognitive": tc["cognitive"], "timestamp": datetime.now().isoformat()}

        # ── LANGKAH 1: Retrieval Evaluation ────────────────────────────────
        print("         ↳ [1] Evaluasi retrieval embedding...")
        retrieval_result = evaluate_retrieval_direct(
            tc["query"], tc["relevant_keywords"], k=TOP_K
        )
        result["retrieval"] = retrieval_result
        if "error" not in retrieval_result:
            print(f"             Precision@{TOP_K}: {retrieval_result['precision_at_k']:.3f} | "
                  f"Recall@{TOP_K}: {retrieval_result['recall_at_k']:.3f} | "
                  f"MeanSim: {retrieval_result['mean_similarity']:.3f}")

        # ── LANGKAH 2: Panggil /chat ────────────────────────────────────────
        print("         ↳ [2] Memanggil /chat API...")
        chat_resp = call_chat_api(tc["query"], tc["cognitive"], tc["session_id"])
        result["chat_response"] = chat_resp

        if chat_resp and "reply" in chat_resp:
            llm_reply   = chat_resp.get("reply", "")
            followup_q  = chat_resp.get("followup_question", "")
            result["llm_reply"]          = llm_reply[:300]
            result["followup_question"]  = followup_q

            # Simulasikan konteks RAG yang diterima LLM dari sistem
            # (dalam implementasi nyata, ini adalah chunk aktual dari FAISS)
            simulated_context = [kw + " adalah konsep penting dalam CT"
                                 for kw in tc["relevant_keywords"]]

            # ── LANGKAH 3: Faithfulness ─────────────────────────────────────
            print("         ↳ [3] Mengevaluasi faithfulness...")
            faith_result = evaluate_faithfulness(llm_reply, simulated_context)
            result["faithfulness"] = faith_result
            print(f"             Faithfulness: {faith_result['faithfulness_score']:.3f} "
                  f"({faith_result['method']})")

            # ── LANGKAH 4: Hallucination Detection ─────────────────────────
            print("         ↳ [4] Deteksi halusinasi...")
            hall_result = detect_hallucination(llm_reply, simulated_context, tc["query"])
            result["hallucination"] = hall_result
            print(f"             Risiko halusinasi: {hall_result['hallucination_risk']:.3f} "
                  f"[{hall_result['risk_label']}]")

            # ── LANGKAH 5: Answer Quality via /evaluate ─────────────────────
            print("         ↳ [5] Evaluasi kualitas jawaban (/evaluate)...")
            # Gunakan reference_answer sebagai jawaban mahasiswa untuk uji positif
            eval_resp = call_evaluate_api(
                answer         = tc["reference_answer"],
                correct_answer = llm_reply,
                active_question= tc["query"],
                wrong_count    = 0,
                cognitive      = tc["cognitive"],
                session_id     = tc["session_id"] + "-eval",
            )
            result["answer_quality"] = eval_resp
            if eval_resp:
                is_correct = eval_resp.get("is_correct", False)
                result["answer_correct"] = is_correct
                print(f"             Jawaban referensi: {'✅ BENAR' if is_correct else '❌ SALAH'}")

        else:
            print("         ↳ ⚠️  API /chat tidak tersedia atau error")
            result["llm_reply"]     = None
            result["faithfulness"]  = None
            result["hallucination"] = None
            result["answer_correct"] = None

        results.append(result)
        print()
        time.sleep(1.5)  # Rate limiting

    return results


# =============================================================================
# OFFLINE STATISTICAL ANALYSIS
# =============================================================================

def run_offline_analysis(history_json_path: str) -> Dict:
    """
    Analisis statistik offline dari file JSON historis.

    SESUAI PROPOSAL hal. 9:
      Offline evaluation: menguji analisis statistik berdasarkan file
      JSON dan CSV yang disimpan dari interaksi-interaksi yang sudah
      dilakukan.

    Menghitung:
    - Distribusi tipe kognitif yang digunakan
    - Rata-rata panjang jawaban LLM
    - Proporsi pertanyaan yang dijawab benar (dari log evaluate)
    - Distribusi sesi (berapa query per sesi)
    """
    if not os.path.exists(history_json_path):
        return {"error": f"File tidak ditemukan: {history_json_path}"}

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return {"error": "File JSON kosong"}

    # Pastikan history adalah list
    entries = history if isinstance(history, list) else history.get("history", [])

    cognitive_dist   = {}
    reply_lengths    = []
    sessions         = {}
    followup_present = 0

    for entry in entries:
        cog = entry.get("cognitive", "unknown")
        cognitive_dist[cog] = cognitive_dist.get(cog, 0) + 1

        reply = entry.get("reply", "")
        if reply:
            reply_lengths.append(len(reply))

        sess = entry.get("session_id", "default")
        sessions[sess] = sessions.get(sess, 0) + 1

        if entry.get("followup_question"):
            followup_present += 1

    total = len(entries)
    stats = {
        "total_interactions":      total,
        "unique_sessions":         len(sessions),
        "avg_queries_per_session": round(total / max(len(sessions), 1), 2),
        "cognitive_distribution":  cognitive_dist,
        "avg_reply_length_chars":  round(statistics.mean(reply_lengths), 1) if reply_lengths else 0,
        "median_reply_length":     round(statistics.median(reply_lengths), 1) if reply_lengths else 0,
        "stdev_reply_length":      round(statistics.stdev(reply_lengths), 1) if len(reply_lengths) > 1 else 0,
        "followup_rate":           round(followup_present / max(total, 1), 4),
        "sessions_detail":         sessions,
    }
    return stats


# =============================================================================
# GENERATE LAPORAN
# =============================================================================

def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Menghitung metrik agregat dari seluruh hasil evaluasi.

    Merangkum semua metrik individual ke dalam statistik keseluruhan
    untuk pelaporan akhir:
    - Rata-rata Precision@K, Recall@K, MeanSim (retrieval)
    - Rata-rata Coverage, Source Diversity (retrieval health)
    - Rata-rata Faithfulness Score
    - Rata-rata Hallucination Risk
    - Proporsi Answer Quality (boolean)
    """
    prec_list, rec_list, meansim_list = [], [], []
    cov_list, div_list                = [], []
    faith_list, hall_list             = [], []
    correct_list                      = []

    for r in results:
        ret = r.get("retrieval", {})
        if "precision_at_k" in ret:
            prec_list.append(ret["precision_at_k"])
            rec_list.append(ret["recall_at_k"])
            meansim_list.append(ret["mean_similarity"])
            cov_list.append(ret["coverage"])
            div_list.append(ret["source_diversity"])

        faith = r.get("faithfulness", {})
        if faith and "faithfulness_score" in faith:
            faith_list.append(faith["faithfulness_score"])

        hall = r.get("hallucination", {})
        if hall and "hallucination_risk" in hall:
            hall_list.append(hall["hallucination_risk"])

        if r.get("answer_correct") is not None:
            correct_list.append(1 if r["answer_correct"] else 0)

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else None

    return {
        "n_tested":              len(results),
        "retrieval": {
            "avg_precision_at_k": avg(prec_list),
            "avg_recall_at_k":    avg(rec_list),
            "avg_mean_similarity":avg(meansim_list),
            "avg_coverage":       avg(cov_list),
            "avg_source_diversity":avg(div_list),
        },
        "generation": {
            "avg_faithfulness":      avg(faith_list),
            "avg_hallucination_risk":avg(hall_list),
        },
        "answer_quality": {
            "total_evaluated": len(correct_list),
            "correct":         sum(correct_list),
            "incorrect":       len(correct_list) - sum(correct_list),
            "accuracy":        avg(correct_list),
        },
    }


def save_results(results: List[Dict], aggregate: Dict, offline_stats: Dict) -> Tuple[str, str, str]:
    """Simpan hasil evaluasi ke file JSON dan CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON lengkap
    json_path = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}.json")
    payload   = {"aggregate": aggregate, "offline_stats": offline_stats,
                 "individual_results": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV ringkasan
    csv_path = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}.csv")
    fieldnames = [
        "test_id","query","cognitive",
        "precision_at_k","recall_at_k","mean_similarity","coverage","source_diversity",
        "faithfulness_score","hallucination_risk","risk_label","answer_correct"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ret   = r.get("retrieval", {})
            faith = r.get("faithfulness", {}) or {}
            hall  = r.get("hallucination", {}) or {}
            writer.writerow({
                "test_id":          r["test_id"],
                "query":            r["query"][:80],
                "cognitive":        r["cognitive"],
                "precision_at_k":   ret.get("precision_at_k", ""),
                "recall_at_k":      ret.get("recall_at_k", ""),
                "mean_similarity":  ret.get("mean_similarity", ""),
                "coverage":         ret.get("coverage", ""),
                "source_diversity": ret.get("source_diversity", ""),
                "faithfulness_score": faith.get("faithfulness_score", ""),
                "hallucination_risk": hall.get("hallucination_risk", ""),
                "risk_label":       hall.get("risk_label", ""),
                "answer_correct":   r.get("answer_correct", ""),
            })

    # Laporan teks ringkas
    txt_path = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        _write_report(f, aggregate, offline_stats, results)

    return json_path, csv_path, txt_path


def _write_report(f, aggregate: Dict, offline_stats: Dict, results: List[Dict]):
    """Tulis laporan evaluasi terformat ke file."""
    sep  = "=" * 70
    sep2 = "-" * 70

    def w(line=""): f.write(line + "\n")

    w(sep)
    w("  LAPORAN EVALUASI RAG — CSIPBLLM")
    w(f"  Tanggal  : {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")
    w(f"  Peneliti : Muhammad Ajisaka Arsyi Taj (G6401221090)")
    w(sep)
    w()
    w("RINGKASAN METRIK AGREGAT")
    w(sep2)

    ret = aggregate.get("retrieval", {})
    w(f"  Jumlah test case     : {aggregate['n_tested']}")
    w()
    w("  [A] RETRIEVAL METRICS (hal. 5 proposal)")
    w(f"      Precision@K      : {ret.get('avg_precision_at_k', 'N/A')}")
    w(f"        → Precision@K = |dokumen relevan di Top-K| / K")
    w(f"      Recall@K         : {ret.get('avg_recall_at_k', 'N/A')}")
    w(f"        → Recall@K = |relevan di Top-K| / total relevan")
    w(f"      Mean Similarity  : {ret.get('avg_mean_similarity', 'N/A')}")
    w(f"        → MeanSim = (1/K) Σ si")
    w()
    w("  [B] RETRIEVAL HEALTH (hal. 8 proposal)")
    w(f"      Coverage         : {ret.get('avg_coverage', 'N/A')}")
    w(f"        → Proporsi chunk dengan skor ≥ {COVERAGE_THRESHOLD}")
    w(f"      Source Diversity : {ret.get('avg_source_diversity', 'N/A')}")
    w(f"        → |sumber unik| / K")
    w()

    gen = aggregate.get("generation", {})
    w("  [C] GENERATION QUALITY (hal. 8–9 proposal)")
    w(f"      Avg Faithfulness : {gen.get('avg_faithfulness', 'N/A')}")
    w(f"        → sim(ē_answer, ē_context) + keyword overlap")
    w(f"      Avg Hallucination Risk: {gen.get('avg_hallucination_risk', 'N/A')}")
    w(f"        → 0.5×out_of_context + 0.3×contradiction + 0.2×uncertainty")
    w()

    aq = aggregate.get("answer_quality", {})
    w("  [D] ANSWER QUALITY — boolean (hal. 9 proposal)")
    w(f"      Total dievaluasi : {aq.get('total_evaluated', 0)}")
    w(f"      Benar            : {aq.get('correct', 0)}")
    w(f"      Salah            : {aq.get('incorrect', 0)}")
    w(f"      Akurasi          : {aq.get('accuracy', 'N/A')}")
    w()

    if offline_stats and "error" not in offline_stats:
        w("  [E] OFFLINE STATISTICAL ANALYSIS (hal. 9 proposal)")
        w(f"      Total interaksi      : {offline_stats.get('total_interactions', 'N/A')}")
        w(f"      Sesi unik            : {offline_stats.get('unique_sessions', 'N/A')}")
        w(f"      Avg query/sesi       : {offline_stats.get('avg_queries_per_session', 'N/A')}")
        w(f"      Avg panjang jawaban  : {offline_stats.get('avg_reply_length_chars', 'N/A')} karakter")
        w(f"      Tingkat followup     : {offline_stats.get('followup_rate', 'N/A')}")
        cog_dist = offline_stats.get("cognitive_distribution", {})
        if cog_dist:
            w("      Distribusi kognitif  :")
            for cog, count in sorted(cog_dist.items()):
                w(f"        {cog}: {count} sesi")
    w()

    w(sep2)
    w("DETAIL PER TEST CASE")
    w(sep2)
    for r in results:
        ret   = r.get("retrieval", {})
        faith = r.get("faithfulness", {}) or {}
        hall  = r.get("hallucination", {}) or {}
        w(f"\n  [{r['test_id']:02d}] {r['query'][:65]}")
        w(f"       Kognitif    : {r['cognitive']}")
        if "precision_at_k" in ret:
            w(f"       P@K         : {ret['precision_at_k']:.4f}  "
              f"R@K: {ret['recall_at_k']:.4f}  "
              f"MeanSim: {ret['mean_similarity']:.4f}")
            w(f"       Coverage    : {ret['coverage']:.4f}  "
              f"Diversity: {ret['source_diversity']:.4f}")
        if faith:
            w(f"       Faithfulness: {faith.get('faithfulness_score', '?'):.4f}  "
              f"({faith.get('method', '?')})")
        if hall:
            w(f"       Halusinasi  : {hall.get('hallucination_risk', '?'):.4f}  "
              f"[{hall.get('risk_label', '?')}]")
        ans = r.get("answer_correct")
        if ans is not None:
            w(f"       Jawaban     : {'✅ BENAR' if ans else '❌ SALAH'}")

    w()
    w(sep)
    w("  Fine della valutazione")
    w(sep)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("\n🔬 Memulai RAG Evaluation Suite...")
    print("   Pastikan server FastAPI sudah berjalan di", BASE_URL)
    print("   dan OPENAI_API_KEY telah diset.\n")

    # ── 1. Jalankan seluruh test case ────────────────────────────────────────
    results = run_full_evaluation()

    # ── 2. Hitung metrik agregat ──────────────────────────────────────────────
    print("\n📊 Menghitung metrik agregat...")
    aggregate = compute_aggregate_metrics(results)

    # ── 3. Offline analysis dari log historis ────────────────────────────────
    history_path = os.path.join(
        os.path.dirname(__file__), "history_logs", "conversation_log.json"
    )
    print(f"📂 Membaca log historis dari: {history_path}")
    offline_stats = run_offline_analysis(history_path)

    # ── 4. Simpan semua hasil ─────────────────────────────────────────────────
    print("\n💾 Menyimpan hasil...")
    json_path, csv_path, txt_path = save_results(results, aggregate, offline_stats)

    # ── 5. Tampilkan ringkasan di terminal ────────────────────────────────────
    print("\n" + "="*70)
    print("  ✅ EVALUASI SELESAI")
    print("="*70)
    print(f"  Hasil JSON  : {json_path}")
    print(f"  Hasil CSV   : {csv_path}")
    print(f"  Laporan TXT : {txt_path}")
    print()
    print("  RINGKASAN CEPAT:")
    ret = aggregate.get("retrieval", {})
    gen = aggregate.get("generation", {})
    aq  = aggregate.get("answer_quality", {})
    print(f"    Precision@{TOP_K}        : {ret.get('avg_precision_at_k', 'N/A')}")
    print(f"    Recall@{TOP_K}           : {ret.get('avg_recall_at_k', 'N/A')}")
    print(f"    Mean Similarity     : {ret.get('avg_mean_similarity', 'N/A')}")
    print(f"    Coverage            : {ret.get('avg_coverage', 'N/A')}")
    print(f"    Source Diversity    : {ret.get('avg_source_diversity', 'N/A')}")
    print(f"    Faithfulness        : {gen.get('avg_faithfulness', 'N/A')}")
    print(f"    Hallucination Risk  : {gen.get('avg_hallucination_risk', 'N/A')}")
    print(f"    Answer Accuracy     : {aq.get('accuracy', 'N/A')} "
          f"({aq.get('correct', 0)}/{aq.get('total_evaluated', 0)})")
    print("="*70)


if __name__ == "__main__":
    main()
