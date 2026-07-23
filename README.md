# LogiCT — Chatbot Tutor RAG-LLM untuk Computational Thinking

Sistem tutor adaptif berbasis **RAG (Retrieval-Augmented Generation)** dengan
**LLM lokal (Ollama: llama3 + nomic-embed-text)** untuk mata kuliah
Computational Thinking. Repositori ini adalah implementasi skripsi
*(Muhammad Ajisaka Arsyi Taj, Ilmu Komputer IPB)* dan sudah memuat
**seluruh revisi pasca-sidang**.

> **Dokumen ini juga berfungsi sebagai dokumentasi fungsi (revisi item 4):**
> setiap fungsi yang terlibat dalam skripsi dijelaskan pada bagian
> [Dokumentasi Fungsi per Modul](#dokumentasi-fungsi-per-modul), lengkap
> dengan pemetaan ke Persamaan/Bab skripsi.

---

## Daftar Isi

1. [Quickstart](#quickstart)
2. [Revisi Pasca-Sidang — Ringkasan Implementasi](#revisi-pasca-sidang--ringkasan-implementasi)
3. [Demo Kondisi A vs Kondisi B](#demo-kondisi-a-vs-kondisi-b)
4. [Transparansi Terminal (Verbose)](#transparansi-terminal-verbose)
5. [Menjalankan Evaluasi Skripsi](#menjalankan-evaluasi-skripsi)
6. [Konfigurasi (Environment Variables)](#konfigurasi-environment-variables)
7. [⚠️ Keamanan API Key](#️-keamanan-api-key)
8. [Arsitektur & Alur Data](#arsitektur--alur-data)
9. [Dokumentasi Fungsi per Modul](#dokumentasi-fungsi-per-modul)
10. [Struktur Repositori](#struktur-repositori)

---

## Quickstart

```bash
# 1. Instal dependensi
pip install -r requirements.txt

# 2. Jalankan Ollama & tarik model (sesuai Bab 3.3 skripsi)
ollama serve
ollama pull llama3            # LLM generatif
ollama pull nomic-embed-text  # model embedding 768 dimensi

# 3. Jalankan server
python main.py
# → buka http://localhost:8000  (UI demo dengan toggle Kondisi A/B)
# → http://localhost:8000/docs  (dokumentasi API interaktif)

# 4. (Opsional) Jalankan evaluasi skripsi 210 kasus uji
python scripts/run_evaluation.py            # lengkap
python scripts/run_evaluation.py --limit 3  # smoke test cepat
```

Semua pemrosesan berjalan **lokal** — tidak ada API eksternal yang dipakai
pada konfigurasi default (`CHAT_PROVIDER=ollama`).

---

## Revisi Pasca-Sidang — Ringkasan Implementasi

| # | Permintaan dosen | Implementasi |
|---|------------------|--------------|
| 1 | Demo memisahkan Kondisi A (RAG) dan Kondisi B (LLM murni) | Toggle radio di UI (`static/index.html`) + field `mode` pada `POST /chat` → `app/services/tutor.py::generate_reply(mode=...)` |
| 2 | Bukti kuat Kondisi B tanpa RAG/bantuan kognitif | **Tiga lapis bukti**: (a) `prompt_sent` — prompt persis yang dikirim, tampil di UI & terminal; (b) `app/core/rag_guard.py::NoRAGGuard` — memblokir teknis semua retrieval/embedding selama mode B (`RAGBlockedError`); (c) `no_rag_proof` — laporan jumlah upaya yang diblokir (0 = tidak ada jalur kode menyentuh RAG) |
| 3 | Terminal menampilkan semua aktivitas latar + metrik live saat demo | `app/core/verbose.py` (chunking, embedding, tabel chunk file+topik+skor, echo prompt) + `app/services/live_metrics.py` (P@K/Coverage/MeanSim/Diversity dengan rumus+substitusi, scan Uncertainty/Contradiction per jawaban) — aktif default (`LOGICT_VERBOSE=1`) |
| 4 | README mendokumentasikan setiap fungsi | Bagian [Dokumentasi Fungsi per Modul](#dokumentasi-fungsi-per-modul) di bawah |
| 5 | Justifikasi angka (K=6, model nomic, threshold) | Ditulis sebagai teks siap tempel skripsi (diberikan terpisah); parameter kini satu sumber di `evaluation/metrics.py` & `app/core/config.py` |
| 6 | Angka Coverage & Precision terbalik + output evaluasi detail | **Akar masalah**: `COVERAGE_THRESHOLD` lama = 0,20 < θ, padahal Pers. 8 mendefinisikan θc = 0,35 (lebih ketat) → Coverage selalu 1,00. **Diperbaiki** di `evaluation/metrics.py` (θ=0,25, θc=0,35) & `evaluation/runner.py`. Precision kini dihitung dari **skor chunk nyata** (sesuai contoh Pers. 2), bukan skor kata kunci. Evaluasi kini menghasilkan **`eval_TS.xlsx`** (8 sheet: setiap perhitungan, respons utuh A vs B, chunk+topik, lokasi frasa, verdict entailment, leksikon revisi — `evaluation/excel_report.py`) |

Perubahan ikutan yang penting:

* **Leksikon Uncertainty/Contradiction direvisi** (`evaluation/lexicon.py`) —
  frasa rawan false-positive dihapus, frasa baru ID+EN ditambah; deteksi
  berbasis word-boundary dengan pencatatan **posisi karakter & nomor
  kalimat**. `Contradiction` kini **proporsi kalimat** (sesuai Pers. 18),
  bukan akumulasi +0,2 per pola.
* **Entailment ganda dihapus** — `detect_hallucination` menerima
  `precomputed_faith` sehingga LLM-as-Judge hanya berjalan sekali per
  kondisi per kasus (hemat ±m panggilan Ollama, angka konsisten).
* **Retrieval sekali per kasus** pada evaluasi — chunk yang sama dipakai
  untuk metrik retrieval, konteks generate, dan referensi Faithfulness.
* **Dua API key yang sebelumnya hardcoded DIHAPUS** — lihat
  [Keamanan API Key](#️-keamanan-api-key).
* Parameter selaras skripsi: `rag_top_k=6`, `rag_embed_chunk_size=1200`,
  konteks per chunk 600 karakter, provider default Ollama.

---

## Demo Kondisi A vs Kondisi B

Buka `http://localhost:8000`. Di atas kolom pertanyaan ada **toggle kondisi**:

**Kondisi A — LLM + RAG (chatbot lengkap).**
Alur: RL Agent memilih Learning Type (atau dipilih manual) → `retrieve()`
mengambil Top-K=6 chunk materi GT → prompt berisi konteks + profil kognitif +
instruksi tutor → llama3 menjawab → pertanyaan lanjutan dibuat. Di bawah tiap
jawaban muncul panel **🔍 Detail Transparansi** berisi: tabel chunk
(peringkat, **nama file**, **topik**, skor cosine, lolos θ dan θc), metrik
live dengan rumus + substitusi angka, hasil scan frasa
uncertainty/contradiction (dengan posisi), dan **prompt persis** yang dikirim.

**Kondisi B — LLM murni (baseline skripsi).**
Prompt buta **persis Lampiran 4**:

```
Jawab pertanyaan berikut sebaik mungkin:

{pertanyaan}

Berikan jawaban dalam Bahasa Indonesia.
```

Tanpa konteks materi, tanpa profil kognitif, tanpa instruksi tutor, tanpa
riwayat sesi, tanpa pertanyaan lanjutan. Elemen UI alur tutor (dropdown
profil, badge RL, kartu evaluasi jawaban) otomatis disembunyikan. Panel
transparansi menampilkan **no_rag_proof** — bukti guard aktif dan jumlah
upaya retrieval/embedding yang diblokir — beserta prompt yang dikirim.

Melalui API langsung:

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"Apa itu rekursi?","session_id":"demo","mode":"B"}'
```

---

## Transparansi Terminal (Verbose)

Terminal server mencetak **semua aktivitas latar** (default aktif):

| Env | Default | Fungsi |
|-----|---------|--------|
| `LOGICT_VERBOSE` | `1` | Master switch. `0` = senyap. Chunking (file → n chunk), embedding (model, dimensi), tabel retrieval (peringkat, skor, ≥θ, ≥θc, file, **topik**), langkah pipeline, perhitungan metrik (rumus + substitusi), verdict entailment per klaim, `no_rag_proof`. |
| `LOGICT_PROMPT_ECHO` | `head` | Echo prompt yang dikirim ke LLM. `head` = 800 karakter pertama; `full` = utuh; `0` = mati. |
| `DEMO_LIVE_METRICS` | `1` | Hitung metrik live (Pers. 2/6/8/10 + scan Pers. 18) untuk setiap jawaban demo. |
| `DEMO_FULL_METRICS` | `0` | Tambahkan **Faithfulness live** (LLM-as-Judge) per jawaban — lambat (m panggilan Ollama ekstra), gunakan hanya saat demonstrasi khusus. |

Contoh keluaran terminal saat menjawab (Kondisi A):

```
──[ RETRIEVAL Top-6 untuk query (34 char) | kognitif 3TAR ]──────────
     #     skor  ≥θ=0.25  ≥θc=0.35  file sumber              topik
     1   0.6123        ✓         ✓  GT_CT_P04_rekursi.txt    Pertemuan 4: Rekursi …
     …
──[ METRIK LIVE — KONDISI A (dari skor chunk jawaban ini) ]──────────
    ∴ P@6 = |{skor ≥ θ=0.25}| / K = 6/6 = 1.0000
    ∴ Coverage = |{skor ≥ θc=0.35}| / K = 4/6 = 0.6667
    …
```

---

## Menjalankan Evaluasi Skripsi

```bash
python scripts/run_evaluation.py                 # 210 kasus (batch 1+2)
python scripts/run_evaluation.py --batch 1       # hanya batch 1
python scripts/run_evaluation.py --limit 3       # smoke test 3 kasus
python scripts/run_evaluation.py --quiet         # tanpa verbose
python scripts/run_evaluation.py --help          # semua opsi
```

Prasyarat: `ollama serve` berjalan, model `llama3` & `nomic-embed-text`
tersedia, server aplikasi hidup (`python main.py`).

**Keluaran** di `logs/eval_results/`:

| File | Isi |
|------|-----|
| `eval_TS.json` | Hasil mentah lengkap per kasus (semua field detail). |
| `eval_TS.csv` | Ringkas satu baris per kasus. |
| `eval_TS.txt` | Laporan teks agregat A vs B. |
| `responses_TS.csv` | Respons LLM utuh kedua kondisi (log justifikasi). |
| **`eval_TS.xlsx`** | **Laporan utama revisi item 6** — 8 sheet: **Ringkasan** (konfigurasi + agregat + status hipotesis), **Metrik Per Kasus**, **Perhitungan Detail** (rumus + substitusi + hasil untuk SETIAP metrik SETIAP kasus kedua kondisi), **Respons A vs B** (utuh, berdampingan), **Chunk & Sumber** (nama file + topik + skor + lolos θ/θc + cuplikan), **Uncertainty-Contradiction** (frasa, kalimat ke-, posisi karakter, kutipan), **Entailment Detail** (verdict YA/TIDAK per klaim), **Leksikon** (bahan pembaruan Lampiran 5). |

Karena ambang Coverage sudah diperbaiki (θc = 0,35) dan basis Precision
diganti ke skor chunk nyata, **angka hasil re-run akan berbeda dari Tabel 18
versi sidang** — khususnya Coverage tidak lagi 1,00. Ini yang diharapkan.

---

## Konfigurasi (Environment Variables)

Semua nilai di `app/core/config.py` dapat dioverride lewat env var:

| Env | Default | Arti |
|-----|---------|------|
| `CHAT_PROVIDER` | `ollama` | `ollama` (sesuai skripsi) / `openai` / `auto` |
| `EMBEDDING_PROVIDER` | `ollama` | idem untuk embedding |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | alamat Ollama |
| `OLLAMA_CHAT_MODEL` | `llama3` | LLM generatif |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | model embedding (768 dim) |
| `RAG_TOP_K` | `6` | K pada Top-K retrieval (Pers. 2) |
| `RAG_EMBED_CHUNK_SIZE` | `1200` | ukuran chunk karakter (Bab 3.4.2) |
| `RAG_CHUNK_MAX_CHARS` | `600` | potongan konteks per chunk di prompt |
| `THETA_RETRIEVAL` | `0.25` | θ (Pers. 2 & 4) |
| `THETA_COVERAGE` | `0.35` | θc (Pers. 8) |
| `OPENAI_API_KEY` | *(kosong)* | HANYA bila memakai provider OpenAI-compatible |
| `LOGICT_VERBOSE`, `LOGICT_PROMPT_ECHO`, `DEMO_LIVE_METRICS`, `DEMO_FULL_METRICS` | lihat tabel verbose | transparansi |
| `EVAL_LIMIT`, `TEST_BATCH` | `0`, `all` | kontrol evaluasi |

---

## ⚠️ Keamanan API Key

Versi sidang menyimpan **dua API key ChatAnywhere hardcoded** (di
`app/core/config.py` dan `app/services/llm.py`). Keduanya **sudah dihapus**
dari kode, tetapi karena pernah tercantum di repositori, **keduanya wajib
dianggap bocor — segera revoke/regenerasi di dashboard ChatAnywhere.**
Kini key hanya dibaca dari env `OPENAI_API_KEY`, dan konfigurasi default
tidak membutuhkan key sama sekali (semua lokal via Ollama).

---

## Arsitektur & Alur Data

```
                         POST /chat {message, mode}
                                   │
                 ┌─────────────────┴──────────────────┐
            mode="A"                             mode="B"
                 │                                    │
  rl_select_cognitive()  (RL memilih LT)       [RL DILEWATI]
                 │                                    │
  rag.retrieve(query, code)                    NoRAGGuard AKTIF
   ├ load_cognitive_materials(code)             (retrieval/embedding
   ├ load_global_materials()                     → RAGBlockedError)
   ├ get_embedding(query)  ── nomic-embed-text        │
   ├ FAISS/NumPy Top-K=6                       prompt buta Lampiran 4
   └ + score + topic per chunk                        │
                 │                             query_llm_ollama_raw()
  prompt = template(konteks, profil,            (tanpa system prompt,
           riwayat, instruksi tutor)             tanpa fallback)
                 │                                    │
  query_llm() ── llama3                               │
                 │                                    │
  _generate_followup()                                │
                 │                                    │
  compute_live_metrics_a()                    compute_live_metrics_b()
                 │                                    │
                 └────────────┬───────────────────────┘
                              ▼
        ChatResponse{reply, mode, rag_used, retrieved[],
                     live_metrics, no_rag_proof, prompt_sent, …}
```

Evaluasi batch (`scripts/run_evaluation.py` → `evaluation/runner.py`)
memakai jalur mandiri yang identik secara metodologis: retrieve sekali per
kasus → generate A (konteks + profil) & B (buta) → Faithfulness
(LLM-as-Judge entailment, Pers. 12-15) → Hallucination (Pers. 18) →
Answer Accuracy (Pers. 20) → agregasi → JSON/CSV/TXT/**XLSX**.

---

## Dokumentasi Fungsi per Modul

Konvensi: **`nama(param) → return`** — deskripsi. *Pemetaan skripsi* dicetak
miring. Fungsi berprefix `_` adalah privat modul.

### `topics.py` — resolusi topik materi

| Fungsi | Deskripsi |
|--------|-----------|
| **`topic_of(fname, materials_dir=None) → str`** | Membaca beberapa baris awal file materi dan mengembalikan **topik manusiawi** — mis. `"Pertemuan 4: Rekursi — Subtopik: Rekursi Dasar"` — dari pola header (`Pertemuan N: …`, `Subtopik: …`, `KOMBINASI KOGNITIF: …`); fallback ke nama file yang dirapikan. Hasil di-cache (`lru_cache`). Dipakai tabel chunk terminal, panel UI, sheet "Chunk & Sumber". *Mendukung revisi item 3 & 6 (chunk ditampilkan dengan topiknya).* |

### `app/core/config.py` — konfigurasi terpusat

| Fungsi/Objek | Deskripsi |
|--------------|-----------|
| **`Settings`** (dataclass) | Seluruh parameter sistem; setiap field dibaca dari env var seragam. Nilai default = metodologi skripsi (K=6, chunk 1200, θ 0,25, θc 0,35, provider ollama). Properti path: `materials_dir`, `history_dir`, `static_dir`, `rl_logs_dir`, `rl_plots_dir`, `eval_results_dir`, `data_dir`. |
| **`get_settings() → Settings`** | Singleton ber-cache (`lru_cache`) — semua modul membaca konfigurasi yang sama. |
| `_env_str/_env_int/_env_float/_env_bool(name, default)` | Pembaca env var dengan konversi tipe aman. |

### `app/core/verbose.py` — transparansi terminal *(revisi item 3)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`enabled() → bool`** | `True` bila `LOGICT_VERBOSE != "0"` (dibaca saat dipanggil — bisa diubah tanpa restart impor). |
| **`banner(title)` / `section(title)`** | Garis pemisah besar/kecil untuk fase pipeline. |
| **`step(msg)` / `kv(key, value)` / `note(msg)`** | Baris log langkah, pasangan kunci-nilai, dan catatan. |
| **`calc(label, formula=None, substitution=None, result=None)`** | Cetak perhitungan metrik. Bentuk 1-argumen menerima `calc_str` jadi (mis. `"P@6 = |{s ≥ 0,25}|/6 = 6/6 = 1,0000"`); bentuk 4-argumen mencetak label/rumus/substitusi/hasil terpisah. *Setiap angka skripsi bisa diaudit dari terminal.* |
| **`chunk_table(rows, theta, theta_c)`** | Tabel chunk: peringkat, skor, kolom ✓/✗ untuk ≥θ dan ≥θc, **nama file**, **topik** — memperlihatkan keputusan Precision (Pers. 2) & Coverage (Pers. 8) per chunk. |
| **`prompt_echo(title, prompt)`** | Echo prompt yang dikirim ke LLM (`LOGICT_PROMPT_ECHO=head/full/0`). *Bukti isi prompt kedua kondisi.* |

### `app/core/rag_guard.py` — bukti Kondisi B *(revisi item 2)*

| Fungsi/Objek | Deskripsi |
|--------------|-----------|
| **`RAGBlockedError`** | Exception saat ada upaya retrieval/embedding di dalam mode B. |
| **`NoRAGGuard`** (context manager) | `with NoRAGGuard() as g:` mengaktifkan blokade berbasis `threading.local` (aman multi-request) selama blok berjalan. |
| **`NoRAGGuard.report() → dict`** | `no_rag_proof`: `{guard_enforced, retrieval_calls_blocked, embedding_calls_blocked, blocked_operations, keterangan}` — dikirim ke UI & terminal. Nilai 0 = tidak ada jalur kode yang menyentuh RAG. |
| **`guard_active() → bool`** | Status guard pada thread berjalan. |
| **`assert_rag_allowed(operation)`** | Dipanggil di pintu masuk `rag.retrieve()` dan `llm.get_embedding()`; melempar `RAGBlockedError` + mencatat upaya bila guard aktif. |

### `app/core/cognitive.py` — profil kognitif

| Fungsi | Deskripsi |
|--------|-----------|
| **`cognitive_label(code) → str`** | Kode 4-karakter (mis. `3TAR`) → label deskriptif ("Level 3 … Teoritis, Analitis, Reflektif") untuk injeksi prompt Kondisi A. *48 profil = 6 level × [T/P] × [A/G] × [I/R] (Bab 3.2).* |
| **`is_valid(code) → bool`** | Validasi kode terhadap `VALID_COGNITIVE_TYPES`. |

### `app/core/prompts.py` — template prompt Kondisi A

Konstanta template (bukan fungsi): `SYSTEM_PROMPT`, `CHAT_PROMPT_TEMPLATE`
(pertanyaan biasa), `CHAT_CODE_PROMPT_TEMPLATE` (pertanyaan berkode),
`FOLLOWUP_PROMPT_TEMPLATE`, `EVALUATE_PROMPT_WITH/WITHOUT_QUESTION`,
`FEEDBACK_PROMPT_TEMPLATE`, `SCAFFOLD_LEVELS`/`SCAFFOLD_DEFAULT`
(tingkat bantuan bertahap), `CHECK_UNDERSTANDING_LEAD`. Kondisi B **tidak**
memakai satu pun template ini — prompt butanya didefinisikan sebagai
`BLIND_PROMPT_TEMPLATE` di `tutor.py` (identik Lampiran 4).

### `app/services/llm.py` — antarmuka LLM & embedding

| Fungsi | Deskripsi |
|--------|-----------|
| **`probe_and_set_chat_provider() → str`** | Dipanggil sekali saat startup: tentukan provider chat. `ollama` (paksa) / `openai` (paksa) / `auto` (tanpa key → langsung Ollama; ada key → probe 1 token). |
| **`get_active_provider() → str`** / **`force_ollama()`** | Baca / paksa provider aktif (thread-safe). |
| **`query_llm(prompt) → str`** | Jalur Kondisi A: kirim prompt ke provider aktif (jalur OpenAI menyisipkan `SYSTEM_PROMPT`; jalur Ollama mengirim prompt apa adanya), fallback otomatis ke Ollama saat kuota habis, normalisasi LaTeX. |
| **`query_llm_ollama_raw(prompt) → str`** | **Jalur Kondisi B**: prompt apa adanya → Ollama lokal. TANPA system prompt, TANPA fallback provider lain — identik jalur evaluasi. *Bukti kemurnian B (item 2).* |
| `_chat_openai(prompt)` / `_chat_ollama(prompt)` | Implementasi per provider; `_chat_ollama` mencoba `/api/generate` lalu `/api/chat`. |
| **`get_embedding(text) → List[float]`** | Vektor embedding ternormalisasi L2 (dot product = cosine). Ollama-first (nomic-embed-text, 768 dim, *Bab 3.4.2*), fallback OpenAI hanya bila Ollama mati **dan** key tersedia. Baris pertama: `assert_rag_allowed("embedding")` — diblokir total pada mode B. |
| `_embed_ollama(text)` / `_embed_openai(text)` | Implementasi per provider + normalisasi. |
| `_require_openai_client()` | Melempar error jelas bila provider OpenAI diminta tanpa `OPENAI_API_KEY` (pengganti key hardcoded yang dihapus). |
| `_is_quota_error(exc)` | Deteksi pola error kuota untuk fallback otomatis. |

### `app/services/rag.py` — indeks & retrieval *(inti Pers. 1-2)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`retrieve(query, cognitive_code, k=None) → List[dict]`** | **Fungsi inti RAG.** `assert_rag_allowed("retrieval")` → muat indeks per-kognitif & global → embed query (nomic) → cari Top-K (default K=6) di FAISS/NumPy → fallback indeks global bila slot belum penuh → dedup → tiap chunk diperkaya `score` (cosine 4 desimal) & `topic` (via `topic_of`) → tabel verbose. *Pers. 1 (cosine), K=6 (Pers. 2).* |
| **`chunks_to_context(chunks, max_chars=None) → str`** | Format chunk → string konteks prompt `[sumber]\nisi…` (potong `rag_chunk_max_chars`=600/chunk). |
| **`load_cognitive_materials(code)`** | Indeks file materi milik kode kognitif (mis. `3TAR.txt`) — sekali per kode (cache). |
| **`load_global_materials()`** | Indeks semua materi umum (GT_CT*.txt) — dipanggil saat startup. |
| `_embed_file(path, fname)` | Baca file → chunk 1.200 karakter (`rag_embed_chunk_size`, *Bab 3.4.2*) → embed tiap chunk → normalisasi L2; log verbose per file (jumlah chunk, topik). |
| `_build_faiss_index(chunks)` | Bangun `IndexFlatIP` FAISS; menyaring embedding gagal/dimensi tak konsisten. |
| `_numpy_search` / `_faiss_search` / `_search` | Pencarian Top-K (NumPy brute-force ↔ FAISS); hasil membawa `text`, `source`, `chunk_id`, `score`. |

### `app/services/tutor.py` — logika tutor & Kondisi A/B *(revisi item 1-2)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`generate_reply(message, cognitive_code, session_id, rl_selected=None, rl_phase=None, mode="A") → dict`** | Dispatcher kondisi eksperimen: `mode="A"` → `_generate_reply_kondisi_a`, `mode="B"` → `_generate_reply_kondisi_b`. |
| **`_generate_reply_kondisi_a(…) → dict`** | Alur lengkap: label profil → riwayat sesi → `retrieve()` → template (`CHAT_*_PROMPT_TEMPLATE`, deteksi kode via `is_code_like`) → `query_llm()` → `_generate_followup()` → metrik live → log. Return + `mode="A"`, `rag_used=True`, `retrieved[]` (rank/source/topic/score/preview), `live_metrics`, `prompt_sent`. |
| **`_generate_reply_kondisi_b(message, session_id) → dict`** | Prompt buta `BLIND_PROMPT_TEMPLATE` (persis Lampiran 4) → `with NoRAGGuard()` → `query_llm_ollama_raw()` → `no_rag_proof`. Sengaja TANPA riwayat sesi (mencegah kontaminasi konteks dari jawaban A sebelumnya), tanpa profil, tanpa follow-up. Return + `mode="B"`, `rag_used=False`, `retrieved=[]`, `no_rag_proof`, `prompt_sent`. |
| **`evaluate_student_answer(answer, correct_answer, active_question, wrong_count, cognitive_code, session_id) → dict`** | Evaluasi jawaban mahasiswa (alur tutor A): retrieve konteks → `_strict_evaluate` (HASIL: BENAR/SALAH) → bila salah, umpan balik ber-scaffold sesuai `wrong_count` + follow-up baru. |
| `_strict_evaluate(…) → (bool, str)` | Prompt evaluasi ketat + parsing `HASIL:`. |
| `_generate_followup(…) → str` | Satu pertanyaan lanjutan dari jawaban tutor. |
| `_generate_feedback(…) → str` | Umpan balik ber-scaffold (level petunjuk dari `SCAFFOLD_LEVELS`). |
| `BLIND_PROMPT_TEMPLATE` | Konstanta prompt buta — **satu-satunya sumber** teks prompt Kondisi B di aplikasi; identik dengan `evaluation/runner.py::_chat_without_rag` dan Lampiran 4. |

### `app/services/live_metrics.py` — metrik live demo *(revisi item 3)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`compute_live_metrics_a(query, chunks, reply) → dict`** | Dari skor chunk yang dipakai menjawab: `precision_at_k` (Pers. 2), `coverage` (Pers. 8, θc=0,35), `mean_similarity` (Pers. 6), `source_diversity` (Pers. 10) — masing-masing dengan `*_detail` (rumus + substitusi) — plus scan Uncertainty/Contradiction (Pers. 18) pada jawaban. `not_computed` menjelaskan **mengapa** Recall (butuh himpunan R kasus uji), Faithfulness (butuh LLM-as-Judge; aktifkan `DEMO_FULL_METRICS=1`), dan Accuracy tidak dihitung live. |
| **`compute_live_metrics_b(reply) → dict`** | Semua metrik retrieval `None` + alasan "tidak ada retrieval" (bagian bukti B); scan leksikal tetap dihitung. |
| `_lexical_block(reply)` | Pembungkus `uncertainty_scan` + `contradiction_scan` dengan keterangan. |
| `_entailment_live(reply, chunks)` | Faithfulness live opsional (hanya `DEMO_FULL_METRICS=1`). |

### `app/services/session.py` — riwayat & log

| Fungsi | Deskripsi |
|--------|-----------|
| **`get_session(session_id)`** | `ChatMessageHistory` per sesi (memori proses). |
| **`format_history(history, max_chars=None) → str`** | Riwayat → teks prompt (potong `max_history_chars`). |
| **`log_interaction(session_id, cognitive, user_message, reply, followup_question, rl_selected, rl_phase)`** | Catat interaksi ke `history_logs/conversation_log.json` (mode B dicatat dengan `cognitive="B-MURNI"`). |
| `get_all_logs()` / `_flush_logs()` | Baca semua log / tulis buffer ke disk. |

### `app/api/routes/tutor.py` — endpoint

| Fungsi | Deskripsi |
|--------|-----------|
| **`chat(req: ChatRequest) → ChatResponse`** | `POST /chat`. `mode="B"` → **lewati RL sepenuhnya** (tanpa pemilihan LT/profil), panggil `generate_reply(mode="B")`, kembalikan bukti (no_rag_proof, prompt_sent, live_metrics). `mode="A"` → RL memilih LT (cold start) → alur tutor lengkap + field transparansi. |
| **`evaluate(req: EvalRequest) → EvalResponse`** | `POST /evaluate` (alur A): nilai jawaban mahasiswa → update reward RL. |

`app/models/schemas.py`: `ChatRequest` (+`mode`), `ChatResponse`
(+`mode, rag_used, retrieved, live_metrics, no_rag_proof, prompt_sent`),
`EvalRequest`, `EvalResponse`, dsb.

### `evaluation/metrics.py` — metrik retrieval *(satu-satunya sumber ambang)*

| Fungsi/Konstanta | Pemetaan skripsi |
|------------------|------------------|
| **`THETA_RETRIEVAL = 0.25`** | θ — Pers. 2 & 4. |
| **`THETA_COVERAGE = 0.35`** | θc = θ + 0,10 — Pers. 8. **Perbaikan revisi item 6** (sebelumnya tertukar 0,20). |
| **`cosine_similarity(a, b)`** | Pers. 1. |
| **`precision_at_k(scores, k, threshold)`** / **`precision_at_k_detail(…)→dict`** | Pers. 2 — proporsi Top-K berskor ≥ θ, dihitung dari **skor chunk nyata**. Varian `_detail` mengembalikan `{value, formula, scores_used, substitution, calc_str}`. |
| **`recall_at_k(…)`** / **`recall_at_k_detail(…)`** | Pers. 4 — kata kunci R "ditemukan" bila sim(query, kw) ≥ θ. |
| **`mean_similarity(…)`** / **`mean_similarity_detail(…)`** | Pers. 6. |
| **`coverage_score(…)`** / **`coverage_detail(…)`** | Pers. 8 (θc). |
| **`source_diversity(…)`** / **`source_diversity_detail(…)`** | Pers. 10 — file unik / K (detail menyertakan daftar file unik). |
| `chunk_relevance_score(scores)` | Rerata skor chunk (metrik tambahan, bukan Persamaan skripsi). |

Semua varian `_detail` diverifikasi mereproduksi contoh perhitungan di
skripsi (mis. P@6 dari [0,61; 0,55; 0,47; 0,41; 0,33; 0,28] = 1,000;
Coverage = 5/6 = 0,8333).

### `evaluation/lexicon.py` — leksikon Uncertainty/Contradiction *(Lampiran 5 revisi)*

| Fungsi/Konstanta | Deskripsi |
|------------------|-----------|
| **`UNCERTAINTY_PHRASES` / `CONTRADICTION_PHRASES`** | Leksikon revisi pasca-sidang (ID+EN): frasa rawan false-positive dihapus ("perlu dicatat", "namun perlu", "sebaliknya" tunggal, "berbeda dengan", dll.), frasa baru ditambah ("bertolak belakang", "kontradiksi", "kelihatannya", "diduga", "barangkali", "it seems", …). |
| **`uncertainty_scan(text) → dict`** | `{value ∈ {0, 0.2} (biner, Pers. 18), flag, matches[{phrase, char_start, char_end, sentence_index, sentence}]}`. |
| **`contradiction_scan(text) → dict`** | `{value = n_flagged/n_sentences (proporsi kalimat, Pers. 18), n_sentences, n_flagged_sentences, matches[…]}`. |
| **`split_sentences_with_spans(text)`** | Pemecah kalimat dengan rentang posisi (untuk pelaporan lokasi). |
| **`lexicon_table()`** | Tabel `{jenis, frasa}` — sumber sheet "Leksikon" & pembaruan Lampiran 5. |
| `_compile` / `_scan` | Kompilasi regex word-boundary, longest-match-first, tanpa tumpang-tindih. |

### `evaluation/faithfulness.py` — Faithfulness & Hallucination *(Pers. 11-19)*

| Fungsi | Pemetaan skripsi |
|--------|------------------|
| **`evaluate_faithfulness(answer, contexts, embed_fn, use_entailment=True) → dict`** | Pers. 15: `F = 0,70×Entailment + 0,30×KWoverlap`. Return: `faithfulness_score`, `entailment_score`, `claims_supported/evaluated`, `entailment_detail[{claim, supported}]` (verdict per klaim), `keyword_overlap`, `kw_overlap_detail`, `calc_str`, `method`. Fallback embedding-similarity bila Ollama judge tak tersedia. |
| `_entailment_score(…)` | Pers. 12-13 — LLM-as-Judge: maksimal 6 kalimat terpanjang dinilai YA/TIDAK terhadap konteks (1.200 char). |
| `_judge_single_claim(claim, context)` | Satu panggilan judge → `True/False/None`. |
| `_keyword_overlap(…)` / `_keyword_overlap_detail(…)` | Pers. 14 — `(|Stems(ans)∩Stems(ctx)|/|Stems(ctx)|)^0,65` dengan stemming ID sederhana; detail berisi |∩|, |ctx|, rasio, hasil pangkat, `calc_str`. |
| `_split_sentences` / `_normalize_word` | Praproses klaim & stemming ringan. |
| `_embedding_faithfulness(…)` | Jalur fallback berbasis cosine embedding. |
| **`detect_hallucination(answer, contexts, query, embed_fn, precomputed_faith=None) → dict`** | Pers. 18: `Risk = 0,65×(1−F) + 0,20×Contradiction + 0,15×Uncertainty`. **`precomputed_faith` (revisi): pakai hasil `evaluate_faithfulness` yang sudah ada — entailment TIDAK dihitung dua kali.** Contradiction = proporsi kalimat (`contradiction_scan`); Uncertainty biner 0/0,2. Return: `hallucination_risk`, `risk_label` (RENDAH <0,32 / SEDANG / TINGGI >0,55 — Tabel 15), `contradiction_*`, `uncertainty_*`, `*_matches` (lokasi frasa), `calc_str`. Direproduksi terhadap contoh Pers. 19 (0,280). |
| `_ollama_generate_local(…)` | Panggilan Ollama mandiri untuk judge (terpisah dari app). |

### `evaluation/runner.py` — pipeline evaluasi A vs B *(Bab 3.4.6)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`run_evaluation() → List[dict]`** | Loop utama per kasus uji: **[A-0]** retrieve SEKALI (`_retrieve_local` — chunk membawa score+topic) → **[A-1]** `_eval_retrieval(retrieved=…)` → **[A-2]** `_chat_with_rag(chunks=…)` → **[A-3]** `evaluate_faithfulness` → **[A-4]** `detect_hallucination(precomputed_faith=faith_a)` → **[A-5]** `_evaluate_locally` → **[B-1]** `_chat_without_rag` (prompt buta) → **[B-2..B-4]** faithfulness/hallucination/accuracy B terhadap chunk GT dari A. Verbose penuh (calc_str per metrik, verdict per klaim). `EVAL_LIMIT` membatasi jumlah kasus. |
| **`_eval_retrieval(query, keywords, cognitive_code, k=6, retrieved=None) → dict`** | **Revisi item 6**: Precision/MeanSim/Coverage/Diversity dari **skor chunk nyata** (fallback skor kata kunci ditandai `precision_basis`); Recall dari kata kunci (Pers. 4). Mengembalikan kelima `*_detail` + `chunks_detail` (rank, source, **topic**, score, ge_theta, ge_theta_c, preview). |
| `_retrieve_local(query, code, k)` | Retrieval mandiri evaluasi (chunk 1.200, cache per file) + `score`/`topic` per chunk + tabel verbose. |
| `_load_and_embed_file` / `_ollama_embed` / `_ollama_embed_list` / `_ollama_generate` | I/O Ollama evaluasi (terpisah dari app agar bisa jalan tanpa server). |
| `_chat_with_rag(query, code, chunks=None)` | Prompt Kondisi A evaluasi (konteks + profil + instruksi); **tidak retrieve ulang** bila `chunks` diberikan. |
| `_chat_without_rag(query, code)` | Prompt buta **identik Lampiran 4** + echo verbose. |
| `_evaluate_locally(question, gt_reference, llm_reply)` | LLM-as-Judge Answer Accuracy (Pers. 20): parsing `HASIL: BENAR/SALAH`. |
| **`compute_aggregates(results) → dict`** | Rerata retrieval; A vs B: faithfulness, hallucination (+distribusi RENDAH/SEDANG/TINGGI per Tabel 15), accuracy; delta; Cohen's d; perbaikan relatif. |
| **`save_results(results, aggregates, offline, output_dir) → (json, csv, txt, responses_csv, xlsx)`** | Tulis kelima keluaran; xlsx via `build_excel` dengan blok konfigurasi (model, K, θ, θc, bobot). |
| `_save_response_log(…)` | `responses_TS.csv` — respons utuh A & B per kasus. |
| `run_offline_analysis(json_path)` | Statistik log percakapan (panjang jawaban, distribusi kognitif). |
| `_load_test_cases()` / `_eval_limit()` | Pemilihan batch (`TEST_BATCH`) & batas kasus (`EVAL_LIMIT`) saat runtime. |

### `evaluation/excel_report.py` — laporan Excel *(revisi item 6)*

| Fungsi | Deskripsi |
|--------|-----------|
| **`build_excel(results, aggregates, config, out_path) → str`** | Susun 8 sheet (lihat tabel keluaran evaluasi di atas) dengan styling (header, freeze pane, wrap). Setiap sheet dibangun `_sheet_*` privat: `_sheet_ringkasan`, `_sheet_metrik`, `_sheet_perhitungan`, `_sheet_respons`, `_sheet_chunks`, `_sheet_lexmatches`, `_sheet_entailment`, `_sheet_lexicon`. |

### `evaluation/test_cases.py` & `test_cases_batch2.py`

210 kasus uji (89 application, 34 confusion, 26 gap, 21 cross-topic,
21 out-of-scope, 13 scenario, 6 comparative) — masing-masing:
`query`, `cognitive`, `relevant_keywords` (himpunan R untuk Recall),
`query_type`, `context_note`.

### `scripts/run_evaluation.py` — CLI evaluasi

`main()` — pre-flight check Ollama (model tersedia?) & server aplikasi,
lalu `run_evaluation → compute_aggregates → run_offline_analysis →
save_results`. Flag: `--batch {1,2,all}`, `--limit N`, `--verbose`,
`--quiet`, `--ollama-url`, `--chat-model`, `--embed-model`,
`--pace-min/max`, `--skip-checks`.

### Modul pendamping (bukan bagian evaluasi skripsi ini)

* **`app/services/rl.py`, `app/api/routes/rl.py`, `rl_metrics.py`,
  `simulate_rl.py`, `param_sweep.py`, `pedagogy_selector.py`,
  `evaluation/evaluator.py`** — sistem **RL Contextual Bandit** pemilih
  Learning Type (cold-start ε-greedy). Ini kontribusi skripsi rekan
  (sistem pendamping); pada skripsi ini RL hanya berperan memilih profil di
  Kondisi A dan **dilewati sepenuhnya di Kondisi B**.
* **`rag_evaluator.py` (root)** — evaluator versi lama, **tidak dipakai**;
  dipertahankan sebagai arsip. Pipeline resmi = `evaluation/runner.py`.
* `app/api/routes/reference.py`, `history.py`, `question_bank.py` —
  endpoint pendukung (daftar tipe kognitif, unduh riwayat, bank soal).

---

## Struktur Repositori

```
logict/
├── main.py                     # entry point server (python main.py)
├── topics.py                   # resolusi topik dari header file materi
├── requirements.txt            # + openpyxl (laporan Excel)
├── app/
│   ├── main.py                 # FastAPI factory + startup echo konfigurasi
│   ├── core/
│   │   ├── config.py           # Settings env-driven (TANPA API key)
│   │   ├── verbose.py          # transparansi terminal (item 3)
│   │   ├── rag_guard.py        # NoRAGGuard — bukti Kondisi B (item 2)
│   │   ├── cognitive.py        # 48 profil kognitif
│   │   └── prompts.py          # template prompt Kondisi A
│   ├── services/
│   │   ├── llm.py              # query_llm / query_llm_ollama_raw / get_embedding
│   │   ├── rag.py              # retrieve() Top-K=6 + score + topic
│   │   ├── tutor.py            # generate_reply(mode="A"/"B")
│   │   ├── live_metrics.py     # metrik live demo (item 3)
│   │   ├── session.py          # riwayat & log
│   │   └── rl.py               # RL Contextual Bandit (sistem pendamping)
│   ├── api/routes/             # tutor(/chat,/evaluate), rl, reference, …
│   └── models/schemas.py       # ChatRequest.mode, ChatResponse.+transparansi
├── evaluation/
│   ├── metrics.py              # θ=0,25, θc=0,35 + *_detail (Pers. 1-10)
│   ├── lexicon.py              # leksikon revisi + scan berlokasi (Pers. 18)
│   ├── faithfulness.py         # entailment + hallucination (Pers. 11-19)
│   ├── runner.py               # pipeline A vs B (Bab 3.4.6)
│   ├── excel_report.py         # eval_TS.xlsx 8 sheet (item 6)
│   └── test_cases*.py          # 210 kasus uji
├── scripts/run_evaluation.py   # CLI evaluasi (--limit, --batch, --quiet)
├── materials/                  # dokumen GT materi CT (+ file profil kognitif)
├── static/                     # UI demo (toggle A/B + panel transparansi)
└── logs/eval_results/          # keluaran evaluasi (json/csv/txt/xlsx)
```

---

*README ini bagian dari revisi pasca-sidang. README versi sidang diarsipkan
sebagai `README_lama.md`.*
