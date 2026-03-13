"""
evaluation/test_cases.py
─────────────────────────
Test-case dataset for the RAG evaluation suite.

Design principles for these queries
────────────────────────────────────
1. HUMAN-LIKE FRAMING — questions read like a real student typing into a chat:
   hesitation markers, partial context ("I just learned…"), scenario setups,
   follow-up style ("wait, so…"), and occasional Indonesian mixing.

2. CT TECHNICALITY — every question maps to a concrete Computational Thinking
   concept (decomposition, abstraction, pattern recognition, algorithm design,
   complexity, data structures).

3. QUERY TYPES — five archetypes that cover the range of real student behaviour:
   • conceptual    — "what is / apa itu" — basic definitional
   • comparative   — "what's the diff / apa bedanya" — two concepts side-by-side
   • scenario      — grounded in a real mini-problem the student is solving
   • confusion     — student states a misconception and asks for clarification
   • application   — student wants to apply something they just learned

Each entry:
  query             — the raw question sent to /chat
  reference_answer  — ground-truth used by the Ollama local evaluator
  relevant_keywords — terms that SHOULD appear in retrieved material chunks
  cognitive         — 4-char cognitive type code
  session_id        — unique ID for this eval run
  query_type        — archetype label (for reporting)
  context_note      — short human-readable note on the scenario
"""

from typing import Dict, List

TEST_CASES: List[Dict] = [

    # ── 1. Conceptual — Algoritma ──────────────────────────────────────────
    {
        "query": (
            "Pak, saya baru mulai belajar CT nih dan masih bingung. "
            "Jadi kalau dibilang 'algoritma' itu sebenernya maksudnya apa sih? "
            "Apa bedanya sama langkah-langkah biasa yang kita tulis di kertas?"
        ),
        "reference_answer": (
            "Algoritma adalah urutan instruksi yang terdefinisi jelas, terbatas, dan "
            "pasti berhenti untuk menyelesaikan suatu masalah. Berbeda dari langkah biasa, "
            "algoritma harus deterministik, tidak ambigu, dan menghasilkan output yang benar."
        ),
        "relevant_keywords": ["algoritma", "instruksi", "urutan", "deterministik", "langkah"],
        "cognitive": "1PAR",
        "session_id": "eval-001",
        "query_type": "conceptual",
        "context_note": "Mahasiswa baru pertama kali dengar istilah 'algoritma' dalam konteks CT",
    },

    # ── 2. Scenario — Dekomposisi ──────────────────────────────────────────
    {
        "query": (
            "Oke jadi saya lagi ngerjain tugas bikin aplikasi absensi mahasiswa. "
            "Masalahnya gede banget, saya gak tau mulai dari mana. "
            "Kata teman saya pakai 'dekomposisi' bisa bantu — bisa dijelasin caranya "
            "kalau diterapin ke kasus absensi ini?"
        ),
        "reference_answer": (
            "Dekomposisi adalah teknik memecah masalah besar menjadi sub-masalah yang lebih kecil "
            "dan mudah dikelola. Untuk aplikasi absensi: pecah menjadi (1) input data mahasiswa, "
            "(2) pencatatan kehadiran, (3) penyimpanan data, (4) laporan. Setiap bagian bisa "
            "dikerjakan dan diuji secara terpisah."
        ),
        "relevant_keywords": ["dekomposisi", "sub-masalah", "bagian", "pecah", "modular"],
        "cognitive": "2PAR",
        "session_id": "eval-002",
        "query_type": "scenario",
        "context_note": "Mahasiswa menghadapi proyek nyata dan mencari cara memulai",
    },

    # ── 3. Confusion — Rekursi vs Iterasi ─────────────────────────────────
    {
        "query": (
            "Saya agak bingung nih. Katanya rekursi itu fungsi yang manggil dirinya sendiri, "
            "tapi kan itu keliatannya bakal jalan terus dong tanpa henti? "
            "Terus bedanya sama loop while apa, bukannya keduanya 'ngulang' sesuatu?"
        ),
        "reference_answer": (
            "Rekursi berhenti karena ada base case — kondisi yang menghentikan pemanggilan diri. "
            "Tanpa base case, memang terjadi stack overflow. Perbedaan dengan iterasi: rekursi "
            "menyimpan state di call stack (memori lebih besar), iterasi di loop variable. "
            "Rekursi lebih elegan untuk masalah hirarkis (tree, fractal), iterasi lebih efisien."
        ),
        "relevant_keywords": ["rekursi", "base case", "iterasi", "stack", "perulangan"],
        "cognitive": "3TGR",
        "session_id": "eval-003",
        "query_type": "confusion",
        "context_note": "Mahasiswa punya miskonsepsi bahwa rekursi = infinite loop",
    },

    # ── 4. Application — Bubble Sort ──────────────────────────────────────
    {
        "query": (
            "Saya nemu array [64, 25, 12, 22, 11] dan disuruh sort manual pakai bubble sort. "
            "Tapi saya gak yakin mekanismenya — setiap iterasi itu yang bergerak elemennya yang "
            "mana? Bisa tolong jelasin langkah pertamanya buat array itu?"
        ),
        "reference_answer": (
            "Bubble sort membandingkan dua elemen berdampingan dan menukar jika yang kiri lebih besar. "
            "Iterasi pertama pada [64,25,12,22,11]: bandingkan 64>25 → tukar → [25,64,12,22,11], "
            "64>12 → tukar → [25,12,64,22,11], 64>22 → tukar → [25,12,22,64,11], "
            "64>11 → tukar → [25,12,22,11,64]. Elemen terbesar 'menggelembung' ke akhir."
        ),
        "relevant_keywords": ["bubble sort", "bandingkan", "tukar", "elemen", "iterasi"],
        "cognitive": "2PAI",
        "session_id": "eval-004",
        "query_type": "application",
        "context_note": "Mahasiswa mengerjakan contoh konkret, butuh trace langkah per langkah",
    },

    # ── 5. Conceptual — Abstraksi ──────────────────────────────────────────
    {
        "query": (
            "Dalam materi CT pilar keempat itu abstraksi. Tapi saya masih blur — "
            "abstraksi itu berarti kita 'sembunyiin' detail yang gak penting kan? "
            "Tapi gimana kita tau detail mana yang penting dan mana yang bisa diabaikan?"
        ),
        "reference_answer": (
            "Abstraksi adalah proses mengidentifikasi dan menyimpan hanya informasi yang relevan "
            "untuk tujuan tertentu, mengabaikan detail yang tidak mempengaruhi solusi. "
            "Cara menentukan: tanyakan 'apakah detail ini mengubah output/solusi?' — jika tidak, "
            "abaikan. Contoh: peta jalan hanya tampilkan nama jalan, bukan warna aspal."
        ),
        "relevant_keywords": ["abstraksi", "relevan", "detail", "sederhanakan", "informasi"],
        "cognitive": "4TAI",
        "session_id": "eval-005",
        "query_type": "conceptual",
        "context_note": "Mahasiswa paham definisi tapi bingung cara praktis menerapkan abstraksi",
    },

    # ── 6. Scenario — Pattern Recognition ─────────────────────────────────
    {
        "query": (
            "Pak saya lagi analisis data nilai ujian 100 mahasiswa. "
            "Banyak yang dapat nilai rendah di soal nomor 3 dan 7 yang keduanya soal rekursi. "
            "Ini ada hubungannya sama pattern recognition di CT? Atau saya terlalu GR?"
        ),
        "reference_answer": (
            "Itu tepat — kamu sedang menerapkan pattern recognition. Kamu mengidentifikasi pola "
            "berulang (nilai rendah pada soal rekursi) dari data besar. Pattern recognition dalam CT "
            "adalah kemampuan mengenali kesamaan, tren, atau keterulangan yang bisa dimanfaatkan "
            "untuk solusi yang lebih umum dan efisien."
        ),
        "relevant_keywords": ["pattern recognition", "pola", "kesamaan", "tren", "identifikasi"],
        "cognitive": "3TGI",
        "session_id": "eval-006",
        "query_type": "scenario",
        "context_note": "Mahasiswa menemukan pola dalam data nyata dan ragu itu termasuk CT",
    },

    # ── 7. Comparative — Big-O ────────────────────────────────────────────
    {
        "query": (
            "Saya baca kalau binary search itu O(log n) dan linear search O(n). "
            "Tapi kalau arraynya cuma 10 elemen, bedanya kan gak kerasa dong? "
            "Kapan sih penting buat peduli sama Big-O notation ini?"
        ),
        "reference_answer": (
            "Untuk n=10 memang perbedaannya kecil. Big-O penting saat n besar: O(n) pada n=1.000.000 "
            "= 1 juta operasi vs O(log n) = 20 operasi. Big-O mengukur laju pertumbuhan, bukan "
            "kecepatan absolut. Jadi untuk data kecil, pertimbangkan juga overhead algoritma; "
            "untuk data besar, Big-O menjadi faktor penentu."
        ),
        "relevant_keywords": ["Big-O", "kompleksitas", "log n", "pertumbuhan", "efisiensi"],
        "cognitive": "5PAI",
        "session_id": "eval-007",
        "query_type": "comparative",
        "context_note": "Mahasiswa mempertanyakan relevansi Big-O pada skala kecil",
    },

    # ── 8. Confusion — Array vs List ──────────────────────────────────────
    {
        "query": (
            "Di Python saya selalu pakai list, tapi dosen bilang itu beda sama array. "
            "Saya lihat di internet ada yang bilang list Python itu 'dynamic array'. "
            "Jadi sebenernya mereka sama atau beda? Saya makin bingung."
        ),
        "reference_answer": (
            "Array klasik adalah blok memori berurutan bertipe sama dengan ukuran tetap. "
            "Python list adalah dynamic array — bisa berisi berbagai tipe, ukuran bisa berubah, "
            "tapi di balik layar tetap menggunakan array dengan resizing otomatis. "
            "Perbedaan praktis: array (numpy) lebih efisien untuk operasi numerik massal; "
            "Python list lebih fleksibel tapi ada overhead."
        ),
        "relevant_keywords": ["array", "list", "memori", "tipe", "dynamic", "indeks"],
        "cognitive": "2TGI",
        "session_id": "eval-008",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung karena Python list dan array tampak serupa",
    },

    # ── 9. Application — Pseudocode ───────────────────────────────────────
    {
        "query": (
            "Tugas saya harus bikin pseudocode sebelum coding. "
            "Masalahnya saya gak tau formatnya harus sedetail apa — "
            "apakah harus mirip Python, atau boleh pakai bahasa Indonesia semua? "
            "Ada aturan resmi gak untuk pseudocode?"
        ),
        "reference_answer": (
            "Pseudocode tidak punya standar baku — tujuannya adalah keterbacaan manusia, bukan "
            "eksekusi mesin. Boleh bahasa Indonesia, campuran, atau mirip Python. Yang penting: "
            "setiap langkah jelas dan tidak ambigu, struktur kontrol (if/loop) dinyatakan eksplisit, "
            "dan variabel dinamai bermakna. Hindari detail sintaks bahasa tertentu."
        ),
        "relevant_keywords": ["pseudocode", "algoritma", "langkah", "struktur", "keterbacaan"],
        "cognitive": "1PAR",
        "session_id": "eval-009",
        "query_type": "application",
        "context_note": "Mahasiswa kebingungan soal standar penulisan pseudocode",
    },

    # ── 10. Comparative — Linked List vs Array ────────────────────────────
    {
        "query": (
            "Kalau saya mau bikin daftar kontak di aplikasi yang sering insert/delete data, "
            "lebih baik pakai linked list atau array? "
            "Saya dengar linked list lebih bagus untuk insert tapi saya gak paham kenapa."
        ),
        "reference_answer": (
            "Linked list unggul untuk insert/delete karena hanya mengubah pointer — O(1) jika "
            "posisi diketahui. Array harus menggeser semua elemen — O(n). Tapi linked list lebih "
            "lambat untuk akses acak (O(n) vs O(1) array). Untuk daftar kontak dengan banyak "
            "insert/delete: linked list. Untuk akses cepat berdasarkan indeks: array."
        ),
        "relevant_keywords": ["linked list", "pointer", "node", "insert", "array", "akses"],
        "cognitive": "3PAI",
        "session_id": "eval-010",
        "query_type": "comparative",
        "context_note": "Mahasiswa sedang memilih struktur data untuk kasus konkret",
    },

    # ── 11. Application — Binary Search ───────────────────────────────────
    {
        "query": (
            "Saya coba implementasi binary search tapi hasilnya kadang salah. "
            "Array saya sudah diurutkan: [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]. "
            "Kalau saya cari angka 23, langkah-langkahnya gimana seharusnya?"
        ),
        "reference_answer": (
            "Binary search pada [2,5,8,12,16,23,38,56,72,91] mencari 23: "
            "mid=(0+9)/2=4 → arr[4]=16 < 23 → cari kanan. "
            "mid=(5+9)/2=7 → arr[7]=56 > 23 → cari kiri. "
            "mid=(5+6)/2=5 → arr[5]=23 = target → ditemukan di indeks 5. "
            "Total 3 perbandingan vs 6 untuk linear search."
        ),
        "relevant_keywords": ["binary search", "tengah", "terurut", "indeks", "perbandingan"],
        "cognitive": "4PAR",
        "session_id": "eval-011",
        "query_type": "application",
        "context_note": "Mahasiswa debugging implementasi binary search dengan trace manual",
    },

    # ── 12. Conceptual — Stack ────────────────────────────────────────────
    {
        "query": (
            "Saya dengar stack itu kayak 'tumpukan piring' — yang terakhir masuk, pertama keluar. "
            "Tapi di dunia nyata, di mana stack ini beneran dipake? "
            "Kayaknya abstrak banget kalau cuma diibaratkan piring."
        ),
        "reference_answer": (
            "Stack digunakan di: (1) function call stack — setiap pemanggilan fungsi push frame, "
            "return pop frame; (2) undo/redo di text editor; (3) evaluasi ekspresi matematika; "
            "(4) navigasi browser (back button). Prinsip LIFO memastikan konteks terakhir "
            "selalu yang pertama diselesaikan — kritikal untuk rekursi."
        ),
        "relevant_keywords": ["stack", "LIFO", "push", "pop", "call stack", "rekursi"],
        "cognitive": "2TAI",
        "session_id": "eval-012",
        "query_type": "conceptual",
        "context_note": "Mahasiswa minta contoh dunia nyata agar konsep stack lebih konkret",
    },

    # ── 13. Application — Flowchart ───────────────────────────────────────
    {
        "query": (
            "Saya mau gambar flowchart untuk sistem login — user input username+password, "
            "kalau salah 3 kali dikunci. Simbol apa yang harus saya pakai untuk "
            "bagian 'cek password' dan 'hitung percobaan gagal'-nya?"
        ),
        "reference_answer": (
            "Untuk sistem login: gunakan belah ketupat (diamond) untuk keputusan 'password benar?' "
            "dan 'percobaan >= 3?'. Gunakan persegi panjang (rectangle) untuk proses 'increment "
            "counter' dan 'kunci akun'. Oval untuk Start/End. Panah menghubungkan alur dengan "
            "label Ya/Tidak pada tiap diamond."
        ),
        "relevant_keywords": ["flowchart", "simbol", "diamond", "keputusan", "proses", "alur"],
        "cognitive": "1TGR",
        "session_id": "eval-013",
        "query_type": "application",
        "context_note": "Mahasiswa mengerjakan flowchart untuk sistem nyata",
    },

    # ── 14. Scenario — Queue ──────────────────────────────────────────────
    {
        "query": (
            "Saya lagi bikin simulasi antrian di klinik — pasien datang, didaftarkan, "
            "dipanggil dokter satu-satu. Teman saya bilang pakai queue. "
            "Itu FIFO kan? Jadi kalau pasien ke-3 datang, dia dipanggil ke-3 juga? "
            "Terus bagaimana kalau ada pasien prioritas?"
        ),
        "reference_answer": (
            "Benar, queue adalah FIFO — pasien ke-3 dipanggil ke-3. Untuk prioritas, "
            "gunakan priority queue: setiap elemen punya nilai prioritas, elemen prioritas "
            "tertinggi di-dequeue duluan terlepas dari urutan masuk. "
            "Implementasi dengan min-heap atau sorted linked list."
        ),
        "relevant_keywords": ["queue", "FIFO", "antrian", "enqueue", "dequeue", "prioritas"],
        "cognitive": "3TAR",
        "session_id": "eval-014",
        "query_type": "scenario",
        "context_note": "Mahasiswa membangun simulasi antrian dan menemukan kasus edge priority",
    },

    # ── 15. Comparative — Selection Sort vs Bubble Sort ───────────────────
    {
        "query": (
            "Jadi tadi kita bahas bubble sort. Sekarang saya baca tentang selection sort. "
            "Keduanya O(n²) kan? Jadi apa gunanya belajar keduanya kalau sama-sama lambat? "
            "Kapan saya pilih selection sort over bubble sort?"
        ),
        "reference_answer": (
            "Meskipun sama-sama O(n²), selection sort membuat lebih sedikit swap (selalu tepat n-1 swap) "
            "vs bubble sort yang bisa swap O(n²) kali. Ini penting jika operasi tukar itu mahal "
            "(misal menulis ke disk atau SSD). Bubble sort lebih bagus jika data hampir terurut "
            "(bisa O(n) dengan optimasi). Belajar keduanya melatih analisis trade-off algoritma."
        ),
        "relevant_keywords": ["selection sort", "bubble sort", "swap", "perbandingan", "trade-off"],
        "cognitive": "4PGR",
        "session_id": "eval-015",
        "query_type": "comparative",
        "context_note": "Mahasiswa mempertanyakan manfaat mempelajari dua algoritma O(n²)",
    },

    # ── 16. Confusion — Rekursi Base Case ─────────────────────────────────
    {
        "query": (
            "Saya coba bikin fungsi faktorial rekursif tapi dapat RecursionError. "
            "Kodenya: def faktorial(n): return n * faktorial(n-1). "
            "Apa yang salah? Saya pikir n pasti akan mencapai 1 sendiri."
        ),
        "reference_answer": (
            "Masalahnya: tidak ada base case. Ketika n=1, fungsi masih memanggil faktorial(0), "
            "lalu faktorial(-1), dan seterusnya tanpa henti sampai stack overflow. "
            "Perbaikan: tambahkan if n <= 1: return 1. Base case adalah kondisi penghenti "
            "yang HARUS ada di setiap fungsi rekursif."
        ),
        "relevant_keywords": ["rekursif", "base case", "stack overflow", "faktorial", "kondisi berhenti"],
        "cognitive": "3TGI",
        "session_id": "eval-016",
        "query_type": "confusion",
        "context_note": "Mahasiswa debugging RecursionError karena lupa base case",
    },

    # ── 17. Scenario — Tree ───────────────────────────────────────────────
    {
        "query": (
            "Saya disuruh representasikan struktur folder sistem file (folder bisa punya "
            "subfolder dan file di dalamnya) sebagai struktur data. "
            "Dosen bilang pakai tree. Kenapa tree cocok untuk ini? "
            "Node-nya itu apa dan edge-nya apa?"
        ),
        "reference_answer": (
            "Tree cocok karena sistem file bersifat hierarki — satu root, tiap node bisa punya "
            "nol atau lebih anak. Node = folder atau file; edge = relasi 'berisi'. "
            "Root = folder paling atas (/). Folder = internal node (punya anak); "
            "File = leaf node (tidak punya anak). Traversal DFS cocok untuk list semua file."
        ),
        "relevant_keywords": ["tree", "node", "edge", "hierarki", "akar", "leaf", "traversal"],
        "cognitive": "3PGI",
        "session_id": "eval-017",
        "query_type": "scenario",
        "context_note": "Mahasiswa memodelkan sistem file nyata menggunakan tree",
    },

    # ── 18. Conceptual — Variabel dan Tipe Data ───────────────────────────
    {
        "query": (
            "Pertanyaan basic tapi saya mau mastiin paham beneran — "
            "kalau saya tulis umur = 20 di Python, tipe datanya otomatis int. "
            "Tapi di bahasa lain katanya harus deklarasi dulu. "
            "Ini bedanya static typing vs dynamic typing? Mana yang lebih aman?"
        ),
        "reference_answer": (
            "Benar. Dynamic typing (Python): tipe ditentukan saat runtime — fleksibel tapi "
            "error tipe baru muncul saat dieksekusi. Static typing (Java, C): tipe harus "
            "dideklarasikan — lebih verbose tapi error terdeteksi saat kompilasi. "
            "'Lebih aman' bergantung konteks: static typing lebih aman untuk sistem besar; "
            "dynamic lebih produktif untuk prototipe cepat."
        ),
        "relevant_keywords": ["tipe data", "variabel", "static", "dynamic", "deklarasi", "runtime"],
        "cognitive": "2TGR",
        "session_id": "eval-018",
        "query_type": "conceptual",
        "context_note": "Mahasiswa mengobservasi perbedaan Python vs bahasa lain",
    },

    # ── 19. Application — If-Else ─────────────────────────────────────────
    {
        "query": (
            "Saya bikin program nilai — A kalau >= 85, B kalau >= 70, C kalau >= 55, D sisanya. "
            "Saya pakai 4 if terpisah tapi kadang nilai 90 muncul sebagai A dan B sekaligus. "
            "Salah di mana? Apa bedanya pakai if berulang vs if-elif-else?"
        ),
        "reference_answer": (
            "4 if terpisah dievaluasi semua secara independen — nilai 90 memenuhi >= 85 DAN >= 70 "
            "sehingga keduanya dieksekusi. Dengan if-elif-else: begitu satu kondisi true, blok "
            "lainnya dilewati. Gunakan: if nilai >= 85: A elif nilai >= 70: B elif nilai >= 55: C "
            "else: D. Ini mutual exclusive — hanya satu cabang yang jalan."
        ),
        "relevant_keywords": ["if", "elif", "else", "kondisi", "percabangan", "mutual exclusive"],
        "cognitive": "2TAR",
        "session_id": "eval-019",
        "query_type": "confusion",
        "context_note": "Mahasiswa debugging bug logika karena salah pakai if vs elif",
    },

    # ── 20. Comparative — For vs While ────────────────────────────────────
    {
        "query": (
            "Kapan saya harus pakai for dan kapan while? "
            "Guru saya bilang 'pakai for kalau tau jumlah iterasinya', tapi "
            "saya bisa aja hitung dulu terus pakai for. Jadi apa bedanya beneran secara CT?"
        ),
        "reference_answer": (
            "Secara CT: for loop mengekspresikan iterasi yang terbatas dan terdefinisi — "
            "jumlah langkah diketahui sebelum loop dimulai, membuat niat kode lebih jelas. "
            "While lebih tepat saat kondisi berhenti bergantung pada state yang berubah "
            "(baca file sampai EOF, validasi input user). Secara teknis keduanya bisa saling "
            "menggantikan, tapi keterbacaan dan kejelasan niat berbeda."
        ),
        "relevant_keywords": ["for", "while", "iterasi", "kondisi", "perulangan", "bounded"],
        "cognitive": "4TAR",
        "session_id": "eval-020",
        "query_type": "comparative",
        "context_note": "Mahasiswa mempertanyakan perbedaan semantik for vs while, bukan hanya sintaks",
    },

    # ── 21. Scenario — Insertion Sort ─────────────────────────────────────
    {
        "query": (
            "Saya lagi debugging program sorting buat nilai ujian. "
            "Ada teman yang saran pakai insertion sort karena 'data sudah hampir urut'. "
            "Kenapa 'hampir urut' itu jadi alasan milih insertion sort? "
            "Bukannya tetap O(n²)?"
        ),
        "reference_answer": (
            "Insertion sort punya best-case O(n) ketika data sudah hampir urut — "
            "setiap elemen hanya perlu digeser sedikit atau tidak sama sekali. "
            "Berbeda dari bubble/selection sort yang tetap O(n²) meski data sudah terurut. "
            "Jadi untuk data yang 'nearly sorted', insertion sort jauh lebih cepat dalam praktik "
            "meski secara worst-case tetap O(n²)."
        ),
        "relevant_keywords": ["insertion sort", "nearly sorted", "best case", "geser", "O(n)"],
        "cognitive": "4PAI",
        "session_id": "eval-021",
        "query_type": "scenario",
        "context_note": "Mahasiswa mendapat saran teman dan ingin memahami alasannya secara teknis",
    },

    # ── 22. Conceptual — Hash Table ───────────────────────────────────────
    {
        "query": (
            "Saya sering dengar 'hash table' atau 'dictionary' di Python. "
            "Katanya lookup-nya O(1) — itu beneran? Masa iya secepat itu? "
            "Gimana cara kerjanya di balik layar biar bisa O(1)?"
        ),
        "reference_answer": (
            "Hash table menyimpan pasangan key-value menggunakan fungsi hash yang mengkonversi "
            "key menjadi indeks array. Lookup O(1) rata-rata karena langsung menuju indeks tanpa "
            "iterasi. Kasus terburuk O(n) saat banyak collision (banyak key ke indeks sama). "
            "Python dict adalah implementasi hash table dengan open addressing dan resizing otomatis."
        ),
        "relevant_keywords": ["hash table", "hash function", "collision", "lookup", "O(1)", "dictionary"],
        "cognitive": "3TGR",
        "session_id": "eval-022",
        "query_type": "conceptual",
        "context_note": "Mahasiswa skeptis dengan klaim O(1) dan ingin memahami mekanismenya",
    },

    # ── 23. Confusion — Pass by Value vs Reference ────────────────────────
    {
        "query": (
            "Saya bikin fungsi yang harusnya ubah nilai variabel, "
            "tapi setelah fungsinya selesai, variabel di luar fungsi gak berubah. "
            "Padahal saya yakin sudah assign di dalam fungsi. "
            "Ini ada hubungannya sama pass by value vs reference?"
        ),
        "reference_answer": (
            "Ya, ini pass by value. Untuk tipe primitif (int, float, string), Python mengoper "
            "salinan nilai — perubahan di dalam fungsi tidak mempengaruhi variabel asal. "
            "Untuk object mutable (list, dict), Python mengoper referensi — perubahan isi "
            "object terlihat di luar. Solusi: return nilai dari fungsi, atau gunakan global/nonlocal."
        ),
        "relevant_keywords": ["pass by value", "referensi", "fungsi", "variabel", "mutable", "scope"],
        "cognitive": "3PAR",
        "session_id": "eval-023",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung kenapa perubahan dalam fungsi tidak terefleksi di luar",
    },

    # ── 24. Application — Merge Sort ──────────────────────────────────────
    {
        "query": (
            "Di kuliah tadi dijelasin merge sort pakai divide and conquer. "
            "Saya paham konsepnya: bagi dua, sort masing-masing, lalu gabung. "
            "Tapi bagian 'merge' itu konkretnya gimana? "
            "Kalau saya punya [1,3,5] dan [2,4,6] cara gabungnya step by step?"
        ),
        "reference_answer": (
            "Merge dua array terurut [1,3,5] dan [2,4,6]: bandingkan elemen terdepan masing-masing, "
            "ambil yang terkecil. 1<2 → ambil 1; 3>2 → ambil 2; 3<4 → ambil 3; 5>4 → ambil 4; "
            "5<6 → ambil 5; sisa [6] → tambahkan. Hasil: [1,2,3,4,5,6]. "
            "Proses merge selalu O(n) karena tiap elemen dibandingkan tepat sekali."
        ),
        "relevant_keywords": ["merge sort", "divide and conquer", "merge", "gabung", "terurut"],
        "cognitive": "2TAI",
        "session_id": "eval-024",
        "query_type": "application",
        "context_note": "Mahasiswa memahami konsep tapi butuh trace konkret langkah merge",
    },

    # ── 25. Comparative — DFS vs BFS ──────────────────────────────────────
    {
        "query": (
            "Saya baca tentang DFS dan BFS untuk graph traversal. "
            "Keduanya bisa mengunjungi semua node kan? "
            "Terus kapan saya pilih DFS dan kapan BFS? "
            "Ada kasus di mana satu jauh lebih baik dari yang lain?"
        ),
        "reference_answer": (
            "DFS (stack/rekursi) cocok untuk: menemukan path yang ada, topological sort, "
            "deteksi siklus, dan masalah yang butuh eksplorasi mendalam. "
            "BFS (queue) cocok untuk: jalur terpendek di unweighted graph, level-order traversal, "
            "dan menemukan node terdekat. BFS menjamin jalur terpendek; DFS tidak. "
            "DFS hemat memori untuk graph dalam; BFS hemat untuk graph lebar."
        ),
        "relevant_keywords": ["DFS", "BFS", "graph", "traversal", "jalur terpendek", "stack", "queue"],
        "cognitive": "5TGR",
        "session_id": "eval-025",
        "query_type": "comparative",
        "context_note": "Mahasiswa ingin kriteria pemilihan DFS vs BFS untuk kasus nyata",
    },

    # ── 26. Scenario — Kompleksitas Nested Loop ───────────────────────────
    {
        "query": (
            "Saya bikin program cari semua pasangan elemen yang jumlahnya sama dengan target. "
            "Kodenya pakai loop for di dalam for lagi. "
            "Teman bilang kode saya 'O(n kuadrat)' dan itu jelek. "
            "Emang sejelek apa? Dan bisa dioptimasi gak?"
        ),
        "reference_answer": (
            "Nested loop O(n²) berarti untuk n=1000 butuh ~1 juta operasi — lambat untuk data besar. "
            "Untuk masalah two-sum, bisa dioptimasi ke O(n) dengan hash set: iterasi sekali, "
            "untuk setiap elemen cek apakah (target - elemen) sudah ada di set. "
            "Ini trade-off: O(n) waktu tapi O(n) memori tambahan untuk hash set."
        ),
        "relevant_keywords": ["nested loop", "O(n²)", "optimasi", "hash set", "two-sum", "kompleksitas"],
        "cognitive": "4PAR",
        "session_id": "eval-026",
        "query_type": "scenario",
        "context_note": "Mahasiswa mendapat feedback kode lambat dan ingin tahu cara optimasinya",
    },

    # ── 27. Conceptual — Abstraksi Fungsi ─────────────────────────────────
    {
        "query": (
            "Dosen bilang 'fungsi itu bentuk abstraksi'. "
            "Saya paham fungsi bisa dipanggil berulang, tapi hubungannya sama abstraksi gimana? "
            "Apa yang diabstraksi oleh sebuah fungsi?"
        ),
        "reference_answer": (
            "Fungsi mengabstraksi detail implementasi — pemanggil tidak perlu tahu bagaimana "
            "tugas diselesaikan, cukup tahu input dan output yang diharapkan. "
            "Ini adalah abstraksi prosedural: nama fungsi merepresentasikan operasi tanpa "
            "mengekspos cara kerjanya. Memungkinkan decomposisi dan reuse tanpa duplikasi kode."
        ),
        "relevant_keywords": ["abstraksi", "fungsi", "implementasi", "prosedural", "encapsulation", "reuse"],
        "cognitive": "3TGI",
        "session_id": "eval-027",
        "query_type": "conceptual",
        "context_note": "Mahasiswa mencoba menghubungkan konsep fungsi dengan pilar abstraksi CT",
    },

    # ── 28. Application — Binary Tree Traversal ───────────────────────────
    {
        "query": (
            "Saya punya binary tree: root=1, kiri=2, kanan=3, kiri.kiri=4, kiri.kanan=5. "
            "Kalau saya traversal inorder, preorder, dan postorder, "
            "urutannya bakal beda-beda ya? Bisa tolong tunjukin hasilnya untuk tree itu?"
        ),
        "reference_answer": (
            "Untuk tree: 1(root), 2(kiri), 3(kanan), 4(kiri.kiri), 5(kiri.kanan). "
            "Inorder (kiri-root-kanan): 4, 2, 5, 1, 3. "
            "Preorder (root-kiri-kanan): 1, 2, 4, 5, 3. "
            "Postorder (kiri-kanan-root): 4, 5, 2, 3, 1. "
            "Inorder pada BST menghasilkan urutan sorted."
        ),
        "relevant_keywords": ["binary tree", "inorder", "preorder", "postorder", "traversal", "BST"],
        "cognitive": "3PAI",
        "session_id": "eval-028",
        "query_type": "application",
        "context_note": "Mahasiswa ingin verifikasi hasil traversal pada tree konkret",
    },

    # ── 29. Confusion — Pointer dan Referensi ─────────────────────────────
    {
        "query": (
            "Di C++ ada pointer, di Java ada referensi, di Python katanya juga referensi. "
            "Saya bingung — ini sama aja atau beda? "
            "Kenapa di Python saya gak pernah pakai tanda * atau & tapi katanya pakai referensi?"
        ),
        "reference_answer": (
            "Pointer (C/C++) adalah variabel yang menyimpan alamat memori secara eksplisit — "
            "programmer kontrol langsung dengan * dan &. Referensi (Java, Python) adalah alias "
            "ke object — bahasa yang mengelola detail memori. Python menyembunyikan pointer: "
            "semua variable adalah referensi ke object, tapi programmer tidak perlu dereference manual. "
            "Keduanya konsep yang sama di level hardware, berbeda di level abstraksi."
        ),
        "relevant_keywords": ["pointer", "referensi", "memori", "alamat", "dereference", "abstraksi"],
        "cognitive": "4TGR",
        "session_id": "eval-029",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung karena terminologi berbeda di tiap bahasa",
    },

    # ── 30. Scenario — Desain Sistem Antrian Print ────────────────────────
    {
        "query": (
            "Tugas UTS saya harus desain sistem antrian print di perpustakaan. "
            "Beberapa printer, banyak mahasiswa. Dokumen yang lebih pendek bisa diproses duluan. "
            "Struktur data apa yang cocok, dan bagaimana CT membantu saya mikirin solusinya?"
        ),
        "reference_answer": (
            "Gunakan priority queue (min-heap berdasarkan jumlah halaman) agar dokumen pendek "
            "diprioritaskan. Proses CT: Decomposisi — pisahkan: antrian per printer, dispatcher, "
            "monitor status. Abstraksi — sembunyikan detail hardware printer dari logika antrian. "
            "Pattern recognition — ini mirip job scheduling di OS. Algoritma — priority queue "
            "dengan heap memberikan enqueue O(log n) dan dequeue O(log n)."
        ),
        "relevant_keywords": ["priority queue", "heap", "antrian", "printer", "dekomposisi", "job scheduling"],
        "cognitive": "5PAR",
        "session_id": "eval-030",
        "query_type": "scenario",
        "context_note": "Mahasiswa mengerjakan desain sistem nyata dengan multiple constraint",
    },

    # ── 31. Conceptual — Rekursi Ekor (Tail Recursion) ────────────────────
    {
        "query": (
            "Dosen nyebut 'tail recursion' tadi dan bilang itu lebih efisien. "
            "Saya gak ngerti bedanya sama rekursi biasa. "
            "Apa itu tail recursion dan kenapa lebih bagus dari yang biasa?"
        ),
        "reference_answer": (
            "Tail recursion adalah rekursi di mana pemanggilan rekursif adalah operasi terakhir "
            "di fungsi — tidak ada komputasi setelah recursive call. Compiler/interpreter bisa "
            "mengoptimasi ini menjadi iterasi (tail call optimization/TCO) sehingga tidak "
            "menambah stack frame baru. Hasilnya: O(1) memori stack vs O(n) pada rekursi biasa, "
            "menghindari stack overflow untuk n besar."
        ),
        "relevant_keywords": ["tail recursion", "TCO", "stack frame", "optimasi", "rekursi", "iterasi"],
        "cognitive": "5TGI",
        "session_id": "eval-031",
        "query_type": "conceptual",
        "context_note": "Mahasiswa belum familiar dengan konsep tail call optimization",
    },

    # ── 32. Application — Quick Sort Pivot ────────────────────────────────
    {
        "query": (
            "Saya diminta implementasi quicksort dan bingung soal pilihan pivot. "
            "Kalau arraynya [3, 6, 8, 10, 1, 2, 1] dan saya pilih pivot = elemen pertama (3), "
            "partition pertamanya jadi seperti apa? Kenapa pilihan pivot penting?"
        ),
        "reference_answer": (
            "Dengan pivot=3 pada [3,6,8,10,1,2,1]: pindahkan elemen < pivot ke kiri, > pivot ke kanan. "
            "Hasil partisi: [1,2,1] | 3 | [6,8,10]. Pivot penting karena menentukan keseimbangan: "
            "pivot buruk (selalu terkecil/terbesar) → O(n²) seperti insertion sort. "
            "Pivot ideal = median → O(n log n). Solusi: random pivot atau median-of-three."
        ),
        "relevant_keywords": ["quicksort", "pivot", "partisi", "median", "O(n log n)", "balance"],
        "cognitive": "4PAI",
        "session_id": "eval-032",
        "query_type": "application",
        "context_note": "Mahasiswa tracing quicksort dan mempertanyakan pengaruh pemilihan pivot",
    },

    # ── 33. Comparative — Stack vs Queue ──────────────────────────────────
    {
        "query": (
            "Saya paham stack itu LIFO dan queue itu FIFO. "
            "Tapi kalau dikasih soal 'pilih stack atau queue untuk kasus X', "
            "gimana cara mikirin jawaban yang benar? "
            "Ada framework atau cara berpikir yang bisa saya pakai?"
        ),
        "reference_answer": (
            "Framework: tanya 'apakah urutan pemrosesan kebalikan dari urutan masuk?' → Stack. "
            "'Apakah urutan pemrosesan sama dengan urutan masuk?' → Queue. "
            "Stack: undo/redo, evaluasi ekspresi, DFS, function calls. "
            "Queue: BFS, job scheduling, buffer data stream, antrian pelayanan. "
            "Kunci: pikirkan urutan pemrosesan yang dibutuhkan, bukan struktur datanya dulu."
        ),
        "relevant_keywords": ["stack", "queue", "LIFO", "FIFO", "urutan", "pemrosesan"],
        "cognitive": "3TAR",
        "session_id": "eval-033",
        "query_type": "comparative",
        "context_note": "Mahasiswa butuh mental model untuk memilih antara stack dan queue",
    },

    # ── 34. Scenario — Deteksi Siklus di Graph ────────────────────────────
    {
        "query": (
            "Saya ngerjain project simulasi jaringan komputer. "
            "Ada kasus router A → B → C → A yang bikin infinite loop. "
            "Ini masalah 'cycle detection' ya? Cara deteksinya gimana?"
        ),
        "reference_answer": (
            "Benar, itu cycle detection pada directed graph. Metode dengan DFS: tandai node sebagai "
            "'sedang dikunjungi' (abu-abu) dan 'selesai' (hitam). Jika DFS menemukan edge ke node "
            "abu-abu, siklus terdeteksi. Untuk undirected graph, gunakan Union-Find (Disjoint Set). "
            "Untuk use case routing: gunakan Bellman-Ford yang bisa deteksi negative cycle."
        ),
        "relevant_keywords": ["cycle detection", "DFS", "graph", "directed", "Union-Find", "visited"],
        "cognitive": "5PAR",
        "session_id": "eval-034",
        "query_type": "scenario",
        "context_note": "Mahasiswa menemukan bug infinite loop yang ternyata adalah cycle di graph",
    },

    # ── 35. Conceptual — Greedy Algorithm ─────────────────────────────────
    {
        "query": (
            "Saya dengar istilah 'greedy algorithm' tapi gak ngerti maksudnya. "
            "Greedy itu artinya 'serakah' — apakah algoritmanya beneran serakah? "
            "Kapan greedy berhasil dan kapan gagal?"
        ),
        "reference_answer": (
            "Greedy algorithm mengambil keputusan terbaik lokal di setiap langkah tanpa "
            "mempertimbangkan konsekuensi ke depan. 'Serakah' = selalu ambil yang tampak terbaik sekarang. "
            "Berhasil ketika masalah punya sifat greedy choice property dan optimal substructure. "
            "Contoh berhasil: Huffman coding, Dijkstra, activity selection. "
            "Gagal: knapsack 0/1 (butuh dynamic programming karena keputusan lokal bisa salah)."
        ),
        "relevant_keywords": ["greedy", "optimal lokal", "greedy choice", "Dijkstra", "Huffman", "dynamic programming"],
        "cognitive": "4TGI",
        "session_id": "eval-035",
        "query_type": "conceptual",
        "context_note": "Mahasiswa baru pertama kali mendengar istilah greedy algorithm",
    },

    # ── 36. Confusion — Rekursi vs Dynamic Programming ────────────────────
    {
        "query": (
            "Saya bikin fungsi fibonacci rekursif dan sangat lambat untuk n=40. "
            "Teman bilang pakai 'memoization'. Itu sama aja sama dynamic programming? "
            "Saya gak ngerti hubungan ketiganya — rekursi, memoization, DP."
        ),
        "reference_answer": (
            "Rekursi murni fibonacci menghitung subproblem berulang kali — O(2^n). "
            "Memoization = top-down DP: simpan hasil subproblem di cache, jika sudah dihitung pakai cache. "
            "Menjadi O(n). DP bottom-up: hitung dari kasus kecil ke besar dengan tabel. "
            "Ketiganya saling terkait: memoization adalah rekursi + caching = top-down DP. "
            "Bottom-up DP lebih efisien memori karena tidak ada overhead call stack."
        ),
        "relevant_keywords": ["memoization", "dynamic programming", "rekursi", "cache", "subproblem", "fibonacci"],
        "cognitive": "5TGR",
        "session_id": "eval-036",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung dengan tiga konsep yang saling overlapping",
    },

    # ── 37. Scenario — Pola pada Data Log ─────────────────────────────────
    {
        "query": (
            "Saya analisis log server dan lihat error 'connection timeout' selalu muncul "
            "tiap hari antara jam 14.00–15.00. Teman bilang saya lagi pakai CT. "
            "Langkah CT apa yang sedang saya jalankan, dan apa langkah berikutnya?"
        ),
        "reference_answer": (
            "Kamu sedang menjalankan pattern recognition — menemukan pola waktu berulang. "
            "Langkah CT berikutnya: Abstraksi — tentukan faktor relevan (load server, cron job, backup). "
            "Decomposisi — pisahkan: analisis traffic jam 14-15, cek scheduled task, monitor resource. "
            "Algoritma — desain langkah diagnosis sistematis. Pattern ini mengarah ke hipotesis "
            "'ada proses terjadwal yang memakan resource berlebihan'."
        ),
        "relevant_keywords": ["pattern recognition", "abstraksi", "dekomposisi", "log", "analisis", "hipotesis"],
        "cognitive": "4TGR",
        "session_id": "eval-037",
        "query_type": "scenario",
        "context_note": "Mahasiswa mengidentifikasi bahwa kerjanya terkait CT dan butuh framework lanjutan",
    },

    # ── 38. Application — Big-O Space Complexity ──────────────────────────
    {
        "query": (
            "Saya selalu mikirin kompleksitas waktu, tapi dosen tadi minta analisis "
            "space complexity juga. Untuk fungsi rekursif faktorial(n), "
            "berapa space complexity-nya dan kenapa?"
        ),
        "reference_answer": (
            "Faktorial rekursif: space complexity O(n) karena setiap pemanggilan rekursif "
            "menambah satu frame ke call stack — ada n+1 frame aktif bersamaan saat mencapai base case. "
            "Faktorial iteratif: O(1) space karena hanya satu variabel accumulator. "
            "Space complexity mengukur memori ekstra yang dibutuhkan algoritma, "
            "termasuk stack frames untuk rekursi."
        ),
        "relevant_keywords": ["space complexity", "call stack", "rekursi", "O(n)", "O(1)", "memori"],
        "cognitive": "5PAI",
        "session_id": "eval-038",
        "query_type": "application",
        "context_note": "Mahasiswa baru sadar ada dimensi kompleksitas selain waktu",
    },

    # ── 39. Comparative — Array vs Hash Table ─────────────────────────────
    {
        "query": (
            "Kalau saya mau cek apakah suatu nama sudah ada dalam daftar 10 ribu mahasiswa, "
            "lebih efisien pakai array (terus linear search) atau hash table? "
            "Bedanya di mana secara angka?"
        ),
        "reference_answer": (
            "Array dengan linear search: O(n) per query — 10.000 operasi per pengecekan. "
            "Array terurut dengan binary search: O(log n) — ~13 operasi, tapi perlu sort dulu O(n log n). "
            "Hash table: O(1) rata-rata per query — hanya satu operasi hash + lookup. "
            "Untuk 1.000 query: array linear = 10 juta operasi, hash table = 1.000 operasi. "
            "Hash table unggul jelas untuk data statis besar dengan banyak lookup."
        ),
        "relevant_keywords": ["array", "hash table", "linear search", "O(n)", "O(1)", "lookup", "efisiensi"],
        "cognitive": "4PAR",
        "session_id": "eval-039",
        "query_type": "comparative",
        "context_note": "Mahasiswa ingin perbandingan konkret dengan angka nyata",
    },

    # ── 40. Conceptual — Dekomposisi Fungsional ───────────────────────────
    {
        "query": (
            "Waktu ngoding, saya sering buat fungsi kecil-kecil untuk setiap tugas spesifik. "
            "Ternyata itu ada namanya ya? Dosen bilang 'functional decomposition'. "
            "Apa manfaat konkretnya selain biar kode kelihatan rapi?"
        ),
        "reference_answer": (
            "Functional decomposition memecah program menjadi fungsi dengan tanggung jawab tunggal. "
            "Manfaat konkret: testability — tiap fungsi bisa diuji isolasi; reusability — fungsi "
            "dipanggil dari banyak tempat tanpa duplikasi; maintainability — bug terlokalisasi; "
            "readability — nama fungsi mendokumentasikan intent; parallel development — tim bisa "
            "kerjakan fungsi berbeda secara bersamaan."
        ),
        "relevant_keywords": ["dekomposisi", "fungsi", "single responsibility", "testability", "reusability", "modular"],
        "cognitive": "3PAR",
        "session_id": "eval-040",
        "query_type": "conceptual",
        "context_note": "Mahasiswa sudah praktik tapi belum tahu konsep formal dan manfaatnya",
    },

    # ── 41. Application — Heap Sort ───────────────────────────────────────
    {
        "query": (
            "Dosen kasih soal: 'urutkan [4, 10, 3, 5, 1] pakai heap sort'. "
            "Saya tau heap itu struktur tree tapi gak paham hubungannya sama sorting. "
            "Apa itu max-heap dan bagaimana heap sort menggunakannya?"
        ),
        "reference_answer": (
            "Max-heap adalah binary tree di mana setiap node lebih besar dari anak-anaknya — "
            "root selalu elemen terbesar. Heap sort: 1) Build max-heap dari array O(n), "
            "2) Berulang: swap root (maks) ke posisi akhir, heapify ulang O(log n). "
            "Untuk [4,10,3,5,1]: heap [10,5,3,4,1] → swap 10 ke akhir → heapify → ulangi. "
            "Total O(n log n), in-place, tapi tidak stable."
        ),
        "relevant_keywords": ["heap sort", "max-heap", "heapify", "in-place", "O(n log n)", "binary tree"],
        "cognitive": "4TAI",
        "session_id": "eval-041",
        "query_type": "application",
        "context_note": "Mahasiswa tahu heap sebagai struktur tapi belum tahu koneksinya ke sorting",
    },

    # ── 42. Scenario — Validasi Input User ────────────────────────────────
    {
        "query": (
            "Saya bikin form registrasi dan harus validasi: email harus ada '@', "
            "password minimal 8 karakter, dan username hanya boleh huruf dan angka. "
            "Dosen bilang ini contoh 'algoritma dengan kondisi'. "
            "Bagaimana CT membantu saya desain validasi ini secara sistematis?"
        ),
        "reference_answer": (
            "CT untuk validasi form: Decomposisi — pisahkan validasi email, password, username "
            "menjadi fungsi terpisah. Pattern recognition — semua validasi punya pola: cek kondisi → "
            "kembalikan pesan error. Abstraksi — buat fungsi validate(value, rules) yang umum. "
            "Algoritma: untuk email cek indexOf('@') > 0 AND ada domain; password len >= 8; "
            "username regex [a-zA-Z0-9]+. Setiap kondisi independen dan testable."
        ),
        "relevant_keywords": ["validasi", "kondisi", "dekomposisi", "abstraksi", "regex", "algoritma"],
        "cognitive": "2PAR",
        "session_id": "eval-042",
        "query_type": "scenario",
        "context_note": "Mahasiswa mengerjakan fitur nyata dan ingin pendekatan CT yang sistematis",
    },

    # ── 43. Confusion — Kompleksitas Amortized ────────────────────────────
    {
        "query": (
            "Saya baca bahwa append ke Python list itu O(1) amortized. "
            "Tapi kan kadang list harus resize yang butuh O(n)? "
            "Jadi O(1) atau O(n)? Amortized itu maksudnya apa?"
        ),
        "reference_answer": (
            "Amortized analysis mengukur rata-rata biaya per operasi dalam sequence panjang. "
            "Python list resize terjadi jarang (saat kapasitas penuh) dengan strategi double size. "
            "Biaya total n append: O(n) karena total copy 1+2+4+...+n ≈ 2n = O(n). "
            "Amortized per operasi: O(n)/n = O(1). Jadi meski sesekali O(n), rata-ratanya O(1). "
            "Amortized berbeda dari average-case: amortized adalah jaminan worst-case atas sequence."
        ),
        "relevant_keywords": ["amortized", "append", "resize", "O(1)", "Python list", "sequence"],
        "cognitive": "5TGI",
        "session_id": "eval-043",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung dengan kontradiksi O(1) vs O(n) pada append",
    },

    # ── 44. Conceptual — Tipe Data Abstrak (ADT) ──────────────────────────
    {
        "query": (
            "Dosen bilang stack, queue, dan list itu 'Abstract Data Type' bukan struktur data. "
            "Saya kira mereka sama aja. Apa bedanya ADT dengan struktur data konkret?"
        ),
        "reference_answer": (
            "ADT mendefinisikan PERILAKU (operasi yang tersedia dan kontraknya) tanpa menentukan "
            "implementasi. Stack ADT: push, pop, peek, isEmpty — tidak ditentukan apakah pakai array atau linked list. "
            "Struktur data konkret adalah implementasi spesifik. "
            "Contoh: Stack bisa diimplementasikan dengan array atau linked list — keduanya memenuhi Stack ADT. "
            "ADT adalah abstraksi: pengguna peduli 'apa yang bisa dilakukan', bukan 'bagaimana caranya'."
        ),
        "relevant_keywords": ["ADT", "abstraksi", "implementasi", "kontrak", "stack", "interface"],
        "cognitive": "3TGI",
        "session_id": "eval-044",
        "query_type": "conceptual",
        "context_note": "Mahasiswa tidak membedakan konsep ADT dengan implementasi konkret",
    },

    # ── 45. Comparative — Iteratif vs Rekursif Fibonacci ──────────────────
    {
        "query": (
            "Saya diminta implementasi fibonacci dua cara: iteratif dan rekursif. "
            "Hasil keduanya sama, tapi dosen bilang kompleksitasnya beda jauh. "
            "Seberapa beda tepatnya, dan yang mana harus saya pakai di production?"
        ),
        "reference_answer": (
            "Fibonacci rekursif murni: O(2^n) waktu, O(n) space — sangat lambat, "
            "fibonacci(40) butuh ~1 miliar operasi. "
            "Fibonacci iteratif: O(n) waktu, O(1) space — hanya dua variabel. "
            "Fibonacci dengan memoization: O(n) waktu, O(n) space. "
            "Untuk production: iteratif selalu lebih baik. Rekursif hanya untuk pembelajaran "
            "atau ketika kejelasan kode lebih penting dari performa."
        ),
        "relevant_keywords": ["fibonacci", "rekursif", "iteratif", "O(2^n)", "O(n)", "memoization", "production"],
        "cognitive": "4TAR",
        "session_id": "eval-045",
        "query_type": "comparative",
        "context_note": "Mahasiswa mengerjakan tugas dua implementasi dan butuh analisis komparatif",
    },

    # ── 46. Scenario — Algoritma Rekomendasi Sederhana ────────────────────
    {
        "query": (
            "Tugas akhir saya harus bikin sistem rekomendasi film sederhana. "
            "Data yang saya punya: rating film dari tiap user. "
            "Cara paling simpel tapi masuk akal secara algoritma itu gimana? "
            "Saya belum pernah belajar ML."
        ),
        "reference_answer": (
            "Tanpa ML, gunakan Collaborative Filtering sederhana: cari user dengan preferensi mirip "
            "(cosine similarity atau Pearson correlation pada vektor rating), lalu rekomendasikan "
            "film yang disukai user mirip tapi belum ditonton. "
            "Langkah CT: Decomposisi — pisahkan: hitung similarity, filter film belum ditonton, ranking. "
            "Abstraksi — representasikan tiap user sebagai vektor rating. "
            "Kompleksitas: O(n²) untuk hitung semua pasangan user."
        ),
        "relevant_keywords": ["rekomendasi", "collaborative filtering", "cosine similarity", "vektor", "dekomposisi"],
        "cognitive": "5PAR",
        "session_id": "eval-046",
        "query_type": "scenario",
        "context_note": "Mahasiswa mengerjakan proyek akhir tanpa background ML",
    },

    # ── 47. Application — Notasi Big-O pada Kode Nyata ────────────────────
    {
        "query": (
            "Dosen kasih potongan kode:\n"
            "for i in range(n):\n"
            "    for j in range(n):\n"
            "        if arr[i] + arr[j] == target:\n"
            "            result.append((i, j))\n"
            "Kompleksitasnya apa? Dan bagian mana yang paling menentukan?"
        ),
        "reference_answer": (
            "Kode tersebut O(n²) karena dua nested loop masing-masing n iterasi: n × n = n² operasi. "
            "Bagian penentu: nested loop — setiap kombinasi pasangan (i,j) diperiksa. "
            "Operasi dalam loop (perbandingan dan append) adalah O(1) sehingga tidak mengubah "
            "kompleksitas keseluruhan. Optimasi ke O(n): gunakan hash set, simpan complement, "
            "cek dalam satu pass."
        ),
        "relevant_keywords": ["nested loop", "O(n²)", "Big-O", "kompleksitas", "analisis kode", "optimasi"],
        "cognitive": "3PAI",
        "session_id": "eval-047",
        "query_type": "application",
        "context_note": "Mahasiswa diminta analisis kompleksitas kode konkret yang diberikan dosen",
    },

    # ── 48. Conceptual — Paradigma Pemrograman dan CT ─────────────────────
    {
        "query": (
            "Saya dengar ada OOP, functional programming, procedural. "
            "Ini beda paradigma ya? Terus hubungannya sama CT gimana — "
            "apakah CT hanya berlaku untuk salah satu paradigma?"
        ),
        "reference_answer": (
            "CT adalah cara berpikir dalam menyelesaikan masalah, bukan paradigma bahasa. "
            "CT berlaku di semua paradigma: OOP menggunakan abstraksi via class/object; "
            "Functional menekankan decomposisi via pure functions dan komposisi; "
            "Procedural menggunakan algoritma step-by-step. "
            "Semua paradigma mengimplementasikan pilar CT (dekomposisi, abstraksi, pattern, algoritma) "
            "dengan cara berbeda. CT adalah mindset, paradigma adalah toolset."
        ),
        "relevant_keywords": ["paradigma", "OOP", "functional", "CT", "abstraksi", "dekomposisi", "mindset"],
        "cognitive": "4TGR",
        "session_id": "eval-048",
        "query_type": "conceptual",
        "context_note": "Mahasiswa menghubungkan konsep CT dengan paradigma pemrograman yang dipelajari",
    },

    # ── 49. Confusion — Stable vs Unstable Sort ───────────────────────────
    {
        "query": (
            "Saya sortir data mahasiswa berdasarkan nilai, tapi urutan mahasiswa "
            "dengan nilai sama jadi acak setelah sorting. "
            "Teman bilang saya butuh 'stable sort'. Apa itu? "
            "Algoritma sort mana yang stable?"
        ),
        "reference_answer": (
            "Stable sort mempertahankan urutan relatif elemen dengan nilai sama. "
            "Jika dua mahasiswa punya nilai 85, urutan mereka sebelum sorting tetap sama setelahnya. "
            "Stable: Merge sort, Insertion sort, Bubble sort, Tim sort (Python default). "
            "Unstable: Quick sort, Heap sort, Selection sort. "
            "Python sorted() dan list.sort() menggunakan Tim sort — stable secara default. "
            "Penting saat sort multi-level: sort by nama dulu, lalu sort by nilai."
        ),
        "relevant_keywords": ["stable sort", "merge sort", "urutan relatif", "Tim sort", "unstable", "Python"],
        "cognitive": "3TAI",
        "session_id": "eval-049",
        "query_type": "confusion",
        "context_note": "Mahasiswa menemukan bug karena memakai unstable sort pada data multi-kriteria",
    },

    # ── 50. Scenario — Desain Algoritma dari Nol ──────────────────────────
    {
        "query": (
            "Saya harus bikin program yang baca teks panjang dan cari 3 kata "
            "yang paling sering muncul. Saya belum punya ide sama sekali. "
            "Bagaimana CT membantu saya mulai dari nol sampai punya algoritma yang jelas?"
        ),
        "reference_answer": (
            "Pendekatan CT step by step: "
            "1. Dekomposisi: baca teks → tokenisasi kata → hitung frekuensi → ambil top-3. "
            "2. Abstraksi: representasikan kata sebagai string, frekuensi sebagai integer — "
            "abaikan case dan tanda baca. "
            "3. Pattern recognition: masalah ini mirip histogram/frequency count. "
            "4. Algoritma: gunakan hash map {kata: count}, iterasi semua kata, "
            "tambah/inisiasi counter. Ambil top-3 dengan partial sort O(n log 3) atau heapq.nlargest(). "
            "Total kompleksitas: O(n) untuk counting + O(n log 3) untuk top-k."
        ),
        "relevant_keywords": ["frekuensi", "hash map", "tokenisasi", "dekomposisi", "top-k", "algoritma"],
        "cognitive": "2TAR",
        "session_id": "eval-050",
        "query_type": "scenario",
        "context_note": "Mahasiswa menghadapi masalah terbuka dari nol dan butuh scaffolding CT penuh",
    },
]
