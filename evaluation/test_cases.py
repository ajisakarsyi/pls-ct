"""
evaluation/test_cases.py
─────────────────────────
110 test case REALISTIS untuk RAG evaluation suite.
Dokumen ground truth: pls-ct-main/materials/

TIDAK ADA reference_answer — evaluasi step 5 menggunakan chunk GT yang
di-retrieve oleh RAG (lihat runner.py yang sudah di-patch).

DESAIN relevant_keywords:
  Setiap test case punya 5–7 keyword dengan campuran:
  - 2–3 keyword yang genuinely relevan ke pertanyaan (ada di GT)
  - 1–2 keyword "false trail" — istilah CT lain yang semantically nearby
    tapi bukan yang dicari (membuat P@K dan R@K tidak trivially 1.0)
  - Kadang 1 keyword yang sama sekali tidak ada di GT (out-of-scope)

DESAIN pertanyaan:
  - Ditulis dari perspektif mahasiswa yang BENAR-BENAR bingung
  - ~35% pertanyaan cross-topik: menyentuh lebih dari satu materi
  - ~25% out-of-scope parsial: sudut pandang atau implementasi tidak langsung ada di GT
  - ~40% in-scope tapi ambiguous: topik ada tapi pertanyaannya dari sisi yang tricky

Cara run:
  set MATERIALS_DIR=C:\\path\\to\\pls-ct-main\\materials
  python scripts/run_evaluation.py
"""

from typing import Dict, List

TEST_CASES: List[Dict] = [

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 1 — PENDAHULUAN & DEFINISI CT (GT_CT01, GT_DETAIL_PT01)
    # ══════════════════════════════════════════════════════════════════════

    # 001 | gap — mahasiswa tidak tahu bedanya CT dengan "berpikir logis" biasa
    {
        "query": (
            "Saya sudah baca definisi Computational Thinking dari berbagai sumber "
            "dan semuanya bilang CT itu tentang 'formulasi masalah dan solusi'. "
            "Tapi berpikir logis yang saya pelajari di matematika SMA juga tentang "
            "formulasi masalah. Jadi bedanya apa CT dengan berpikir logis biasa?"
        ),
        "relevant_keywords": ["computational thinking", "formulasi masalah", "berpikir logis", "algoritme", "abstraksi"],
        "cognitive": "2PAR",
        "session_id": "eval-001",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak bisa membedakan CT dengan berpikir logis SMA",
    },

    # 002 | confusion — salah kira AADP itu urutan langkah prosedural
    {
        "query": (
            "Di catatan saya tulis AADP = Abstraksi → Algoritme → Dekomposisi → Pattern Recognition "
            "dan saya kerjakan soal dengan urutan itu. Tapi nilai saya jelek. "
            "Kata teman, AADP bukan urutan. Lalu AADP itu apa sebenarnya?"
        ),
        "relevant_keywords": ["AADP", "pilar", "dekomposisi", "urutan langkah", "pseudocode"],
        "cognitive": "1TAR",
        "session_id": "eval-002",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah mengira AADP adalah urutan prosedural",
    },

    # 003 | out-of-scope — tanya tentang era VUCA dari sudut pandang karir, bukan CT
    {
        "query": (
            "Dosen bilang kita hidup di era VUCA dan CT itu penting untuk karir. "
            "Tapi saya mau jadi dokter, bukan programmer. "
            "Apakah CT relevan untuk dokter di era VUCA, "
            "atau ini hanya penting untuk orang teknik saja?"
        ),
        "relevant_keywords": ["VUCA", "computational thinking", "karir", "dokter", "dekomposisi"],
        "cognitive": "3TGI",
        "session_id": "eval-003",
        "query_type": "out_of_scope",
        "context_note": "Mahasiswa tidak berlatar teknik mempertanyakan relevansi CT untuk karir non-teknik",
    },

    # 004 | cross_topic — CT + ICT literacy dicampur jadi satu pertanyaan
    {
        "query": (
            "Di silabus ada materi CT dan materi ICT Literacy. "
            "Saya bingung — apakah ICT Literacy itu bagian dari CT "
            "atau CT bagian dari ICT Literacy? "
            "Atau keduanya hal yang sama sekali berbeda?"
        ),
        "relevant_keywords": ["ICT literacy", "computational thinking", "literasi digital", "AADP", "teknologi"],
        "cognitive": "2TAI",
        "session_id": "eval-004",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa bingung hubungan hierarki CT dan ICT Literacy",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 2 — ICT LITERACY & ETIKA (GT_CT02, GT_DETAIL_PT02)
    # ══════════════════════════════════════════════════════════════════════

    # 005 | confusion — salah paham digital footprint hanya dari upload
    {
        "query": (
            "Teman saya bilang saya punya digital footprint besar karena sering pakai Instagram. "
            "Tapi saya hampir tidak pernah posting — saya hanya scroll dan nonton video orang lain. "
            "Apa betul saya tetap punya digital footprint meski tidak pernah upload apapun?"
        ),
        "relevant_keywords": ["digital footprint", "jejak digital", "media sosial", "privasi", "algoritme"],
        "cognitive": "3PAR",
        "session_id": "eval-005",
        "query_type": "confusion",
        "context_note": "Mahasiswa kira digital footprint hanya dari konten yang diunggah",
    },

    # 006 | gap — mahasiswa tidak tahu bedanya LwICT dengan ICT Literacy
    {
        "query": (
            "Di slide ada dua istilah: 'ICT Literacy' dan 'Literacy with ICT'. "
            "Keduanya terlihat sama — intinya kan paham teknologi. "
            "Kenapa perlu dua istilah berbeda untuk hal yang sama?"
        ),
        "relevant_keywords": ["ICT literacy", "LwICT", "literasi", "digital", "computational thinking"],
        "cognitive": "2PAI",
        "session_id": "eval-006",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham perbedaan antara ICT Literacy dan LwICT",
    },

    # 007 | out-of-scope — tanya implementasi kebijakan privasi spesifik (tidak ada di GT)
    {
        "query": (
            "Saya buat aplikasi untuk tugas dan kumpulkan data nama dan email teman. "
            "Dari materi etika digital, apakah saya perlu minta izin dulu sebelum menyimpan data mereka? "
            "Dan kalau sudah terlanjur, apa yang harus saya lakukan?"
        ),
        "relevant_keywords": ["etika digital", "privasi", "data pribadi", "izin", "dekomposisi"],
        "cognitive": "4TAI",
        "session_id": "eval-007",
        "query_type": "out_of_scope",
        "context_note": "Mahasiswa tanya prosedur konkret perlindungan data — detail regulasi tidak ada di GT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 3 — DEKOMPOSISI (GT_CT03, GT_DETAIL_PT03, GT_SUBTOPIK_01)
    # ══════════════════════════════════════════════════════════════════════

    # 008 | confusion — mahasiswa kira dekomposisi = membagi tugas ke orang berbeda
    {
        "query": (
            "Waktu kerja kelompok, kami bagi tugas: saya bikin bagian A, teman bikin bagian B. "
            "Kata saya itu dekomposisi, tapi teman saya bilang itu bukan dekomposisi CT. "
            "Lalu apa bedanya 'bagi tugas' biasa dengan dekomposisi dalam CT?"
        ),
        "relevant_keywords": ["dekomposisi", "sub-masalah", "modular", "pembagian tugas", "abstraksi"],
        "cognitive": "1PAI",
        "session_id": "eval-008",
        "query_type": "confusion",
        "context_note": "Mahasiswa kira bagi tugas kelompok = dekomposisi CT",
    },

    # 009 | application — soal lift berang-berang (ada di GT_SUBTOPIK_01)
    {
        "query": (
            "Ada soal: 2 lift kapasitas masing-masing 30 kg. "
            "Ada 9 berang dengan berat: A=2, B=3, C=5, D=8, E=9, F=9, G=12, H=12, I=22 kg. "
            "Bagaimana dekomposisi masalah ini untuk memaksimalkan jumlah berang yang terangkut?"
        ),
        "relevant_keywords": ["dekomposisi", "optimasi", "lift", "sub-masalah", "rekursi", "greedy"],
        "cognitive": "2PAR",
        "session_id": "eval-009",
        "query_type": "application",
        "context_note": "Soal lift berang-berang ada di GT_SUBTOPIK_01 — apakah RAG retrieve dokumen yang tepat",
    },

    # 010 | cross_topic — dekomposisi + fungsi (mahasiswa nyambungkan keduanya)
    {
        "query": (
            "Waktu belajar fungsi, dosen bilang setiap fungsi seharusnya hanya melakukan satu hal. "
            "Itu berhubungan dengan dekomposisi kan? "
            "Tapi saya bingung — apakah fungsi dalam pemrograman itu implementasi dari dekomposisi CT?"
        ),
        "relevant_keywords": ["fungsi", "dekomposisi", "modularitas", "DRY", "abstraksi", "sub-masalah"],
        "cognitive": "3TAI",
        "session_id": "eval-010",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa menghubungkan konsep fungsi dengan dekomposisi CT",
    },

    # 011 | gap — mahasiswa tidak tahu kapan dekomposisi berhenti (level berapa)
    {
        "query": (
            "Kalau saya dekomposisi masalah, sampai seberapa dalam saya harus memecahnya? "
            "Misalnya 'bikin kue' — apakah saya pecah sampai level 'gerakkan jari untuk ngaduk' "
            "atau cukup sampai 'campurkan bahan'? "
            "Apa ada kriteria kapan dekomposisi sudah cukup?"
        ),
        "relevant_keywords": ["dekomposisi", "sub-masalah", "granularitas", "abstraksi", "goal"],
        "cognitive": "4PGR",
        "session_id": "eval-011",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak tahu stopping criterion untuk dekomposisi",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 4 — ABSTRAKSI (GT_CT04, GT_DETAIL_PT04, GT_SUBTOPIK_02)
    # ══════════════════════════════════════════════════════════════════════

    # 012 | application — soal abstraksi lift berang-berang (ada di GT_SUBTOPIK_02)
    {
        "query": (
            "Dari soal lift 2 buah kapasitas 30 kg dengan 9 berang (A:2, B:3, C:5, D:8, E:9, F:9, G:12, H:12, I:22), "
            "dosen minta saya lakukan abstraksi dulu sebelum cari solusi. "
            "Apa yang dimaksud abstraksi untuk soal ini — apa yang perlu difokuskan dan diabaikan?"
        ),
        "relevant_keywords": ["abstraksi", "goal", "batasan", "data", "dekomposisi", "formulasi"],
        "cognitive": "3PAI",
        "session_id": "eval-012",
        "query_type": "application",
        "context_note": "Soal abstraksi lift berang-berang ada di GT_SUBTOPIK_02",
    },

    # 013 | confusion — mahasiswa kira abstraksi = menyederhanakan = membuang semua detail
    {
        "query": (
            "Saya bikin model data untuk sistem nilai mahasiswa dan saya buang semua field "
            "kecuali nama dan NIM karena 'itu abstraksi'. Tapi dosen bilang abstraksi saya salah. "
            "Katanya nilai ujian, kehadiran itu juga penting. "
            "Apakah abstraksi berarti membuang detail sebanyak mungkin?"
        ),
        "relevant_keywords": ["abstraksi", "relevan", "detail", "model", "goal", "dekomposisi"],
        "cognitive": "4TAI",
        "session_id": "eval-013",
        "query_type": "confusion",
        "context_note": "Mahasiswa kira abstraksi = buang semua detail sebanyak mungkin",
    },

    # 014 | gap — helicopter view tidak dipahami
    {
        "query": (
            "Di slide abstraksi ada istilah 'helicopter view'. "
            "Saya tidak ngerti maksudnya — apa hubungannya helikopter dengan CT? "
            "Dan bagaimana cara saya menerapkan helicopter view waktu mengerjakan soal?"
        ),
        "relevant_keywords": ["helicopter view", "abstraksi", "gambaran besar", "detail", "formulasi"],
        "cognitive": "2TGI",
        "session_id": "eval-014",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham metafora helicopter view dalam konteks abstraksi CT",
    },

    # 015 | cross_topic — abstraksi + pattern recognition dicampur
    {
        "query": (
            "Waktu analisis data nilai ujian, saya buang semua outlier dulu baru cari pola. "
            "Apakah membuang outlier itu abstraksi atau pattern recognition? "
            "Atau keduanya sekaligus? Saya bingung mana yang mana."
        ),
        "relevant_keywords": ["abstraksi", "pattern recognition", "outlier", "data", "pola", "dekomposisi"],
        "cognitive": "5TGR",
        "session_id": "eval-015",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa bingung apakah membuang outlier itu abstraksi atau pattern recognition",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 5 — PATTERN RECOGNITION (GT_CT05, GT_DETAIL_PT05, GT_SUBTOPIK_03, GT_SUBTOPIK_04)
    # ══════════════════════════════════════════════════════════════════════

    # 016 | application — deret coklat bebek (ada di GT_SUBTOPIK_03)
    {
        "query": (
            "Soal: bebek ke-n mendapat n coklat. Total coklat 500 buah. "
            "Bebek nomor berapa yang pertama kali tidak mendapat coklat? "
            "Saya sudah coba hitung manual tapi lama sekali."
        ),
        "relevant_keywords": ["pola bilangan", "deret segitiga", "pattern recognition", "rumus", "modulo"],
        "cognitive": "3PGR",
        "session_id": "eval-016",
        "query_type": "application",
        "context_note": "Soal coklat bebek ada di GT_SUBTOPIK_03 — apakah RAG bisa retrieve",
    },

    # 017 | application — digit terakhir 2^2003 (ada di GT_SUBTOPIK_04)
    {
        "query": (
            "Soal ujian: tentukan digit terakhir dari 2 pangkat 2003. "
            "Saya tidak mungkin hitung 2^2003 secara langsung. "
            "Dosen bilang pakai pattern recognition. "
            "Bagaimana caranya?"
        ),
        "relevant_keywords": ["modulo", "siklus", "digit terakhir", "pola", "perpangkatan", "deret"],
        "cognitive": "4TGR",
        "session_id": "eval-017",
        "query_type": "application",
        "context_note": "Soal digit terakhir 2^n ada di GT_SUBTOPIK_04",
    },

    # 018 | confusion — mahasiswa kira pattern recognition hanya untuk angka
    {
        "query": (
            "Saya pikir pattern recognition di CT hanya untuk data angka atau deret matematika. "
            "Tapi dosen bilang bisa juga untuk teks atau gambar. "
            "Apakah pattern recognition di CT sama dengan machine learning pattern recognition? "
            "Atau beda?"
        ),
        "relevant_keywords": ["pattern recognition", "pola", "data", "abstraksi", "machine learning"],
        "cognitive": "5PAR",
        "session_id": "eval-018",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung apakah pattern recognition CT sama dengan ML",
    },

    # 019 | gap — tidak tahu cara memvalidasi pola yang ditemukan
    {
        "query": (
            "Saya lihat data absensi dan nemu pola: nilai turun setiap minggu ke-5. "
            "Tapi bagaimana saya tahu ini bukan kebetulan? "
            "Apa cara CT untuk memvalidasi bahwa pola yang saya temukan itu benar?"
        ),
        "relevant_keywords": ["pattern recognition", "validasi", "pola", "data", "abstraksi", "dekomposisi"],
        "cognitive": "4PGI",
        "session_id": "eval-019",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak tahu cara validasi pola dalam CT",
    },

    # 020 | application — deret aritmatika/geometri (GT_SUBTOPIK_03)
    {
        "query": (
            "Diberikan barisan: 3, 6, 12, 24, 48. "
            "Dosen minta saya identifikasi jenis deret, tuliskan rumus suku ke-n, "
            "dan hitung suku ke-10 tanpa menghitung satu per satu."
        ),
        "relevant_keywords": ["deret geometri", "rasio", "pola bilangan", "rumus", "deret aritmatika", "modulo"],
        "cognitive": "3TGR",
        "session_id": "eval-020",
        "query_type": "application",
        "context_note": "Identifikasi deret geometri dari barisan angka — ada di GT_SUBTOPIK_03",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 6 — ALGORITME & NOTASI (GT_CT06, GT_CT07, GT_DETAIL_PT07, GT_SUBTOPIK_05)
    # ══════════════════════════════════════════════════════════════════════

    # 021 | confusion — mahasiswa kira flowchart lebih formal dari pseudocode
    {
        "query": (
            "Di tugas saya bikin flowchart karena saya kira itu lebih formal dan benar dari pseudocode. "
            "Tapi dosen lebih prefer pseudocode. "
            "Apa bedanya pseudocode dengan flowchart, dan mana yang lebih tepat untuk menggambarkan algoritme?"
        ),
        "relevant_keywords": ["pseudocode", "flowchart", "algoritme", "notasi", "for loop", "percabangan"],
        "cognitive": "2PAI",
        "session_id": "eval-021",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah kira flowchart lebih formal dari pseudocode",
    },

    # 022 | application — pseudocode FPB Euclidean (ada di GT_SUBTOPIK_05)
    {
        "query": (
            "Saya diminta menulis pseudocode algoritme Euclidean untuk mencari FPB dua bilangan. "
            "Saya tahu konsep FPB tapi tidak tahu cara tulis pseudocode-nya dengan while loop. "
            "Bisa tolong tunjukkan?"
        ),
        "relevant_keywords": ["pseudocode", "FPB", "while", "Euclidean", "modulo", "rekursi"],
        "cognitive": "3PAR",
        "session_id": "eval-022",
        "query_type": "application",
        "context_note": "Pseudocode FPB Euclidean ada di GT_SUBTOPIK_05",
    },

    # 023 | gap — mahasiswa tidak tahu ciri algoritme yang baik selain "bisa jalan"
    {
        "query": (
            "Saya sudah buat algoritme untuk cari nilai terbesar dari array dan bisa jalan. "
            "Tapi dosen bilang algoritme saya kurang baik meski hasilnya benar. "
            "Apa saja ciri algoritme yang dianggap 'baik' selain menghasilkan output yang benar?"
        ),
        "relevant_keywords": ["algoritme", "ciri", "finiteness", "efisiensi", "ambigu", "feasibility"],
        "cognitive": "1TGR",
        "session_id": "eval-023",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak tahu ciri algoritme baik selain correctness",
    },

    # 024 | out-of-scope — tanya kompleksitas algoritme Euclidean (tidak eksplisit di GT)
    {
        "query": (
            "Saya buat algoritme FPB Euclidean dan ingin tahu kompleksitasnya Big-O. "
            "Apakah Euclidean itu O(log n) atau O(n)? "
            "Dan kenapa bukan O(1) walaupun sering selesai cepat untuk angka kecil?"
        ),
        "relevant_keywords": ["FPB", "Euclidean", "Big-O", "kompleksitas", "O(log n)", "algoritme"],
        "cognitive": "5TAI",
        "session_id": "eval-024",
        "query_type": "out_of_scope",
        "context_note": "Kompleksitas Big-O dari Euclidean tidak eksplisit ada di GT",
    },

    # 025 | confusion — mahasiswa kira pseudocode harus ada tipe data seperti C++
    {
        "query": (
            "Di pseudocode saya selalu tulis 'int x = 5' karena terbiasa dari C++. "
            "Tapi teman saya hanya tulis 'x = 5'. Kata dosen keduanya benar. "
            "Lalu apa konvensi pseudocode CT yang benar untuk deklarasi variabel?"
        ),
        "relevant_keywords": ["pseudocode", "variabel", "deklarasi", "tipe data", "konvensi", "assignment"],
        "cognitive": "2TGI",
        "session_id": "eval-025",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung konvensi deklarasi variabel di pseudocode CT vs bahasa pemrograman",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 7 — VARIABEL, TIPE DATA, OPERATOR (GT_CT09, GT_DETAIL_PT09, GT_SUBTOPIK_06)
    # ══════════════════════════════════════════════════════════════════════

    # 026 | application — evaluasi ekspresi dengan prioritas operator
    {
        "query": (
            "Soal: hitung nilai ekspresi berikut: (8 + 4) / (2 * 3) - 1. "
            "Dan juga: 15 % 4 * 2 + 7 % 3. "
            "Saya sudah hitung tapi tidak yakin — apakah modulo punya prioritas yang sama dengan perkalian?"
        ),
        "relevant_keywords": ["prioritas operator", "modulo", "ekspresi", "aritmatika", "kurung", "pembagian"],
        "cognitive": "2PAR",
        "session_id": "eval-026",
        "query_type": "application",
        "context_note": "Soal evaluasi ekspresi ada di GT_SUBTOPIK_06",
    },

    # 027 | confusion — De Morgan salah diterapkan
    {
        "query": (
            "Saya sederhanakan ekspresi: NOT (A OR B) menjadi NOT A AND NOT B. "
            "Tapi teman saya bilang itu salah — harusnya NOT A OR NOT B. "
            "Padahal saya rasa Hukum De Morgan yang kedua memang seperti itu. "
            "Mana yang benar dan kenapa?"
        ),
        "relevant_keywords": ["De Morgan", "NOT", "AND", "OR", "logika", "tabel kebenaran"],
        "cognitive": "4TGI",
        "session_id": "eval-027",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah menerapkan Hukum De Morgan kedua",
    },

    # 028 | gap — mahasiswa tidak tahu perbedaan integer division dengan float division
    {
        "query": (
            "Di pseudocode CT, kalau saya tulis 7 / 2, hasilnya 3 atau 3.5? "
            "Di Python hasilnya 3.5 tapi di C hasilnya 3 karena integer division. "
            "Pseudocode CT pakai yang mana? Dan gimana cara nulis integer division di pseudocode?"
        ),
        "relevant_keywords": ["pembagian", "integer", "float", "pseudocode", "tipe data", "modulo"],
        "cognitive": "3TAI",
        "session_id": "eval-028",
        "query_type": "gap",
        "context_note": "Mahasiswa bingung konvensi pembagian integer vs float di pseudocode CT",
    },

    # 029 | application — evaluasi tabel kebenaran AND, OR, NOT lengkap
    {
        "query": (
            "Tugas: buat tabel kebenaran untuk ekspresi (A AND NOT B) OR (NOT A AND B). "
            "Saya buat untuk semua kombinasi A dan B tapi hasilnya berbeda dengan teman. "
            "Saya dapat F,T,T,F tapi teman dapat T,T,T,F. Mana yang benar?"
        ),
        "relevant_keywords": ["tabel kebenaran", "AND", "OR", "NOT", "logika", "evaluasi ekspresi"],
        "cognitive": "3PGI",
        "session_id": "eval-029",
        "query_type": "application",
        "context_note": "Mahasiswa latihan tabel kebenaran XOR — ada di GT_SUBTOPIK_06",
    },

    # 030 | out-of-scope — tanya tentang tipe data struct/record (tidak ada di GT)
    {
        "query": (
            "Di materi variabel dan tipe data, kita belajar integer, float, boolean, string, dan list. "
            "Tapi bagaimana kalau saya mau simpan data mahasiswa yang punya nama, NIM, dan nilai sekaligus? "
            "Apakah ada tipe data gabungan seperti struct di CT?"
        ),
        "relevant_keywords": ["tipe data", "variabel", "list", "string", "struct", "record"],
        "cognitive": "4PAI",
        "session_id": "eval-030",
        "query_type": "out_of_scope",
        "context_note": "Tipe data struct/record tidak ada di GT CT — hanya ada primitif dan list",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 8 — PERCABANGAN (GT_CT10, GT_DETAIL_PT10, GT_SUBTOPIK_07)
    # ══════════════════════════════════════════════════════════════════════

    # 031 | application — trace percabangan kuadran koordinat
    {
        "query": (
            "Trace pseudocode ini untuk input x=-3, y=5:\n"
            "  if (x > 0) then\n    if (y > 0) then print('Q1') else print('Q4')\n"
            "  else\n    if (y > 0) then print('Q2') else print('Q3')\n"
            "Saya dapat Q2 tapi teman dapat Q3. Siapa yang benar?"
        ),
        "relevant_keywords": ["trace", "percabangan", "nested if", "kondisi", "output", "kuadran"],
        "cognitive": "3PGR",
        "session_id": "eval-031",
        "query_type": "application",
        "context_note": "Trace nested if percabangan kuadran ada di GT_SUBTOPIK_07",
    },

    # 032 | confusion — mahasiswa tidak tahu kapan else-if vs nested if
    {
        "query": (
            "Saya harus cek: apakah suhu > 35 DAN kelembaban > 80. "
            "Saya buat nested if — if suhu>35 then if kelembaban>80. "
            "Tapi teman pakai AND: if suhu>35 AND kelembaban>80. "
            "Kata dosen keduanya boleh tapi ada perbedaannya. Apa perbedaannya?"
        ),
        "relevant_keywords": ["nested if", "AND", "percabangan", "kondisi", "else-if", "logika"],
        "cognitive": "4PAR",
        "session_id": "eval-032",
        "query_type": "confusion",
        "context_note": "Mahasiswa tidak tahu kapan nested if lebih tepat dari AND gabungan",
    },

    # 033 | gap — mahasiswa tidak tahu apa yang terjadi kalau dua kondisi sama-sama true di else-if
    {
        "query": (
            "Di else-if untuk konversi nilai: A(>=85), B(>=70), C(>=55), D(lainnya). "
            "Nilai saya 90. Kondisi pertama (>=85) true, tapi kondisi kedua (>=70) juga true. "
            "Mana yang dieksekusi? Dan kenapa tidak keduanya sekaligus?"
        ),
        "relevant_keywords": ["else-if", "kondisi", "true", "percabangan", "sekuensial", "eksekusi"],
        "cognitive": "2TAR",
        "session_id": "eval-033",
        "query_type": "gap",
        "context_note": "Mahasiswa bingung kenapa hanya satu cabang yang dieksekusi di else-if",
    },

    # 034 | application — trace else-if konversi nilai
    {
        "query": (
            "Trace pseudocode konversi nilai untuk input nilai=68:\n"
            "  if (nilai>=85) then print('A')\n"
            "  else if (nilai>=70) then print('B')\n"
            "  else if (nilai>=55) then print('C')\n"
            "  else print('D')\n"
            "Saya dapat C tapi teman dapat D. Mana yang benar dan kenapa?"
        ),
        "relevant_keywords": ["trace", "else-if", "nilai", "percabangan", "kondisi", "output"],
        "cognitive": "2PGI",
        "session_id": "eval-034",
        "query_type": "application",
        "context_note": "Trace else-if konversi nilai — ada di GT_SUBTOPIK_07",
    },

    # 035 | cross_topic — percabangan + loop (mahasiswa mau gabungkan keduanya)
    {
        "query": (
            "Saya mau buat program yang minta input terus sampai user masukkan angka valid (1-100). "
            "Ini perlu loop (while) dan percabangan (if untuk cek valid) sekaligus. "
            "Mana yang jadi 'pembungkus' — apakah if di dalam while, atau while di dalam if?"
        ),
        "relevant_keywords": ["while", "percabangan", "validasi", "loop", "kondisi", "sentinel"],
        "cognitive": "3TAR",
        "session_id": "eval-035",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa menggabungkan while dan if untuk validasi input",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 9 — PERULANGAN (GT_CT11, GT_DETAIL_PT11, GT_SUBTOPIK_08)
    # ══════════════════════════════════════════════════════════════════════

    # 036 | confusion — mahasiswa tidak tahu kapan for vs while dalam kasus nyata
    {
        "query": (
            "Saya diminta baca semua baris file CSV sampai habis. "
            "Saya tidak tahu ada berapa baris di file itu. "
            "Apakah saya pakai for atau while? "
            "Dosen bilang ada kasus pakai for tapi ada yang pakai while untuk hal yang sama."
        ),
        "relevant_keywords": ["for", "while", "iterasi", "kondisi", "file", "sentinel"],
        "cognitive": "2PAI",
        "session_id": "eval-036",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung for vs while untuk membaca file dengan jumlah baris tidak diketahui",
    },

    # 037 | application — pseudocode cetak bilangan genap dengan dua versi
    {
        "query": (
            "Saya diminta tulis pseudocode untuk cetak 10 bilangan genap pertama (2, 4, ..., 20) "
            "dalam DUA versi berbeda: satu dengan step dan satu tanpa step (pakai akumulasi). "
            "Versi dengan step saya sudah bisa, tapi yang akumulasi saya tidak mengerti maksudnya."
        ),
        "relevant_keywords": ["for", "step", "genap", "akumulasi", "pseudocode", "perulangan", "while"],
        "cognitive": "2TGR",
        "session_id": "eval-037",
        "query_type": "application",
        "context_note": "Pseudocode dua versi cetak bilangan genap ada di GT_SUBTOPIK_08",
    },

    # 038 | gap — mahasiswa tidak tahu apa itu sentinel value
    {
        "query": (
            "Di slide perulangan ada istilah 'sentinel value'. "
            "Katanya ini alternatif dari while biasa untuk baca input. "
            "Saya tidak paham apa itu sentinel dan kapan saya pakai itu "
            "daripada while biasa dengan kondisi boolean?"
        ),
        "relevant_keywords": ["sentinel", "while", "input", "perulangan", "kondisi", "for"],
        "cognitive": "3PAI",
        "session_id": "eval-038",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham konsep sentinel value dalam perulangan",
    },

    # 039 | application — trace while loop hitung digit (ada di GT_SUBTOPIK_08)
    {
        "query": (
            "Trace pseudocode berikut untuk input n=4523:\n"
            "  jumlah = 0\n  while (n > 0) do\n    jumlah = jumlah + (n mod 10)\n    n = n div 10\n"
            "  print(jumlah)\n"
            "Saya dapat 14 tapi teman dapat 41. Mana yang benar?"
        ),
        "relevant_keywords": ["trace", "while", "modulo", "div", "perulangan", "digit"],
        "cognitive": "3TGR",
        "session_id": "eval-039",
        "query_type": "application",
        "context_note": "Trace while loop hitung jumlah digit — ada di GT_SUBTOPIK_08",
    },

    # 040 | out-of-scope — do-while loop (tidak ada di GT CT)
    {
        "query": (
            "Di C++ dan Java ada do-while loop yang eksekusi body setidaknya satu kali. "
            "Di pseudocode CT kita hanya belajar for dan while. "
            "Apakah do-while ada di pseudocode CT? "
            "Dan bagaimana simulasi do-while dengan while biasa?"
        ),
        "relevant_keywords": ["do-while", "while", "perulangan", "body", "kondisi", "for"],
        "cognitive": "4TGI",
        "session_id": "eval-040",
        "query_type": "out_of_scope",
        "context_note": "Do-while tidak ada di GT CT — hanya for dan while yang diajarkan",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 10 — NESTED LOOP (GT_SUBTOPIK_09)
    # ══════════════════════════════════════════════════════════════════════

    # 041 | application — cetak segitiga bintang terbalik
    {
        "query": (
            "Saya bisa buat segitiga bintang biasa (1 bintang di baris pertama, semakin banyak ke bawah). "
            "Tapi dosen minta buat yang TERBALIK: 5 bintang di baris pertama, semakin sedikit ke bawah. "
            "Saya tidak tahu harus ubah kondisi loop dalam-nya bagaimana."
        ),
        "relevant_keywords": ["nested loop", "segitiga", "for", "pola", "bintang", "baris"],
        "cognitive": "2PAR",
        "session_id": "eval-041",
        "query_type": "application",
        "context_note": "Pola segitiga terbalik — variasi dari soal di GT_SUBTOPIK_09",
    },

    # 042 | confusion — mahasiswa salah hitung total iterasi nested loop
    {
        "query": (
            "Saya punya nested loop: luar 5 iterasi, dalam 3 iterasi. "
            "Saya kira total iterasinya 5+3=8. Tapi kata teman 5×3=15. "
            "Mana yang benar dan kenapa? Saya bingung karena selalu tambah di logika saya."
        ),
        "relevant_keywords": ["nested loop", "iterasi", "total", "perkalian", "kompleksitas", "O(n^2)"],
        "cognitive": "3PAR",
        "session_id": "eval-042",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah mengira total iterasi nested loop adalah penjumlahan bukan perkalian",
    },

    # 043 | application — matriks identitas dengan nested loop
    {
        "query": (
            "Tugas: cetak matriks identitas 3x3 dengan pseudocode nested loop. "
            "Saya tahu diagonal utamanya berisi 1 dan sisanya 0, "
            "tapi saya tidak tahu cara menulis kondisi untuk diagonal di pseudocode."
        ),
        "relevant_keywords": ["nested loop", "matriks", "kondisi", "diagonal", "for", "print"],
        "cognitive": "4PAI",
        "session_id": "eval-043",
        "query_type": "application",
        "context_note": "Matriks identitas dengan nested loop — ada di GT_SUBTOPIK_09",
    },

    # 044 | gap — tidak tahu kenapa nested loop identik dengan O(n²)
    {
        "query": (
            "Dosen bilang nested loop selalu O(n²). "
            "Tapi saya punya nested loop: luar dari 1 to n, dalam dari 1 to 5 (konstan). "
            "Apakah itu masih O(n²)? Karena loop dalamnya tidak bergantung n."
        ),
        "relevant_keywords": ["nested loop", "O(n^2)", "kompleksitas", "konstan", "Big-O", "iterasi"],
        "cognitive": "5TGR",
        "session_id": "eval-044",
        "query_type": "gap",
        "context_note": "Mahasiswa salah kira semua nested loop adalah O(n²)",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 11 — FUNGSI & MODULARITAS (GT_CT12, GT_DETAIL_PT12, GT_SUBTOPIK_10)
    # ══════════════════════════════════════════════════════════════════════

    # 045 | confusion — mahasiswa kira fungsi dan prosedur sama karena di Python sama
    {
        "query": (
            "Di Python, def bisa return nilai atau tidak — keduanya sama-sama 'fungsi'. "
            "Tapi di pseudocode CT, dosen bedakan 'fungsi' dan 'prosedur'. "
            "Apakah perbedaan ini penting di CT, atau hanya formalitas?"
        ),
        "relevant_keywords": ["fungsi", "prosedur", "return", "Python", "pseudocode", "DRY"],
        "cognitive": "3TAR",
        "session_id": "eval-045",
        "query_type": "confusion",
        "context_note": "Mahasiswa Python tidak paham pembedaan fungsi vs prosedur di pseudocode CT",
    },

    # 046 | application — fungsi luas bangun datar
    {
        "query": (
            "Tugas: tulis tiga fungsi untuk luas persegi panjang, segitiga, dan lingkaran. "
            "Lalu buat program utama yang memanggil ketiganya. "
            "Saya bingung cara menulis parameter dengan tipe data di pseudocode CT."
        ),
        "relevant_keywords": ["fungsi", "parameter", "return", "luas", "tipe data", "prosedur"],
        "cognitive": "2PAI",
        "session_id": "eval-046",
        "query_type": "application",
        "context_note": "Fungsi luas bangun datar ada di GT_SUBTOPIK_10 dan GT_DETAIL_PT12",
    },

    # 047 | gap — mahasiswa tidak paham scope variabel lokal vs global
    {
        "query": (
            "Saya punya variabel x = 10 di program utama. "
            "Di dalam fungsi saya ubah x = 99. "
            "Setelah fungsi selesai, nilai x di program utama jadi 99 atau tetap 10? "
            "Saya bingung karena di Python bisa berbeda hasilnya."
        ),
        "relevant_keywords": ["variabel", "scope", "fungsi", "lokal", "global", "return"],
        "cognitive": "3TAI",
        "session_id": "eval-047",
        "query_type": "gap",
        "context_note": "Mahasiswa bingung scope variabel lokal vs global dalam fungsi pseudocode",
    },

    # 048 | cross_topic — fungsi + rekursi (mahasiswa bingung rekursi adalah fungsi)
    {
        "query": (
            "Rekursi itu fungsi yang memanggil dirinya sendiri. "
            "Tapi kalau rekursi adalah fungsi, kenapa diajarkan terpisah? "
            "Apa yang membuat rekursi 'spesial' dibanding fungsi biasa?"
        ),
        "relevant_keywords": ["rekursi", "fungsi", "self-call", "base case", "stack", "prosedur"],
        "cognitive": "4TGI",
        "session_id": "eval-048",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa tidak paham apa yang membuat rekursi berbeda dari fungsi biasa",
    },

    # 049 | out-of-scope — higher-order function (tidak ada di GT)
    {
        "query": (
            "Di Python ada konsep fungsi yang menerima fungsi lain sebagai parameter — "
            "disebut higher-order function. Apakah konsep ini diajarkan di CT? "
            "Atau CT hanya bahas fungsi yang menerima tipe data primitif?"
        ),
        "relevant_keywords": ["fungsi", "parameter", "higher-order", "prosedur", "abstraksi", "modularitas"],
        "cognitive": "5PAI",
        "session_id": "eval-049",
        "query_type": "out_of_scope",
        "context_note": "Higher-order function tidak ada di GT CT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 12 — REKURSI (GT_SUBTOPIK_11)
    # ══════════════════════════════════════════════════════════════════════

    # 050 | confusion — mahasiswa tidak tahu kenapa rekursi berhenti
    {
        "query": (
            "Saya tulis fungsi rekursif tapi program saya crash dengan error 'maximum recursion depth exceeded'. "
            "Saya sudah tambahkan base case tapi masih crash. "
            "Apa yang salah? Apakah base case saya yang bermasalah atau ada hal lain?"
        ),
        "relevant_keywords": ["rekursi", "base case", "stack overflow", "depth", "infinite", "fungsi"],
        "cognitive": "3PAI",
        "session_id": "eval-050",
        "query_type": "confusion",
        "context_note": "Mahasiswa punya rekursi yang crash — base case tidak memenuhi syarat",
    },

    # 051 | application — trace faktorial(4) dengan call stack
    {
        "query": (
            "Saya diminta trace faktorial(4) secara lengkap dan tunjukkan call stack "
            "di fase 'turun' (push) dan fase 'naik' (pop). "
            "Saya bingung apa bedanya fase turun dan fase naik."
        ),
        "relevant_keywords": ["faktorial", "rekursi", "call stack", "trace", "base case", "push"],
        "cognitive": "3TGR",
        "session_id": "eval-051",
        "query_type": "application",
        "context_note": "Trace faktorial rekursif dengan call stack ada di GT_SUBTOPIK_11",
    },

    # 052 | confusion — mahasiswa tidak tahu mengapa Fibonacci rekursif lambat
    {
        "query": (
            "Saya buat Fibonacci rekursif dan berjalan normal untuk Fib(10). "
            "Tapi waktu saya coba Fib(40) program sangat lambat — hampir 1 menit. "
            "Padahal logikanya sederhana. Kenapa bisa selambat itu?"
        ),
        "relevant_keywords": ["Fibonacci", "rekursi", "O(2^n)", "kompleksitas", "call stack", "memoization"],
        "cognitive": "5TAR",
        "session_id": "eval-052",
        "query_type": "confusion",
        "context_note": "Mahasiswa tidak tahu mengapa Fibonacci rekursif eksponensial lambatnya",
    },

    # 053 | application — trace Fib(6) dari rekursi
    {
        "query": (
            "Hitung Fib(6) menggunakan definisi rekursif: Fib(1)=1, Fib(2)=1, Fib(n)=Fib(n-1)+Fib(n-2). "
            "Saya dapat 11 tapi teman dapat 8. "
            "Tolong trace dari awal untuk membuktikan mana yang benar."
        ),
        "relevant_keywords": ["Fibonacci", "rekursi", "trace", "base case", "perhitungan", "deret"],
        "cognitive": "3TGI",
        "session_id": "eval-053",
        "query_type": "application",
        "context_note": "Trace Fibonacci rekursif — ada di GT_SUBTOPIK_11",
    },

    # 054 | cross_topic — rekursi + kompleksitas (mahasiswa hubungkan keduanya)
    {
        "query": (
            "Dosen bilang faktorial rekursif O(n) tapi Fibonacci rekursif O(2^n). "
            "Keduanya rekursi, kenapa kompleksitasnya bisa beda jauh? "
            "Apa yang menentukan kompleksitas dari sebuah fungsi rekursif?"
        ),
        "relevant_keywords": ["rekursi", "kompleksitas", "O(n)", "O(2^n)", "Fibonacci", "faktorial"],
        "cognitive": "5TGI",
        "session_id": "eval-054",
        "query_type": "cross_topic",
        "context_note": "Mahasiswa menghubungkan rekursi dengan analisis kompleksitas",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 13 — STRUKTUR DATA (GT_CT13, GT_DETAIL_PT13)
    # ══════════════════════════════════════════════════════════════════════

    # 055 | confusion — mahasiswa kira array dan list Python identik
    {
        "query": (
            "Di Python saya pakai list untuk segalanya dan tidak pernah ada masalah. "
            "Tapi di slide CT dibedakan antara array dan list. "
            "Apakah Python list itu array? Kalau iya, kenapa disebut beda?"
        ),
        "relevant_keywords": ["array", "list", "tipe data", "memori", "Python", "indeks"],
        "cognitive": "4TGR",
        "session_id": "eval-055",
        "query_type": "confusion",
        "context_note": "Mahasiswa tidak paham perbedaan array klasik dan Python list",
    },

    # 056 | gap — mahasiswa tidak tahu kapan array lebih baik dari list
    {
        "query": (
            "Kalau Python list sudah bisa melakukan semua yang array bisa — "
            "tambah elemen, akses indeks, loop — kenapa orang masih pakai array? "
            "Kapan tepatnya array lebih baik dari list?"
        ),
        "relevant_keywords": ["array", "list", "efisiensi", "memori", "operasi", "tipe data"],
        "cognitive": "5PAR",
        "session_id": "eval-056",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham keunggulan array dibanding dynamic list",
    },

    # 057 | out-of-scope — linked list (tidak ada di GT CT)
    {
        "query": (
            "Saya baca tentang linked list di internet dan katanya lebih efisien dari array "
            "untuk insert dan delete. Apakah linked list diajarkan di mata kuliah CT ini? "
            "Kalau tidak, apa yang membedakan dengan array?"
        ),
        "relevant_keywords": ["linked list", "array", "insert", "delete", "struktur data", "pointer"],
        "cognitive": "4TGI",
        "session_id": "eval-057",
        "query_type": "out_of_scope",
        "context_note": "Linked list tidak ada di GT CT — hanya array/list yang diajarkan",
    },

    # 058 | application — operasi list: cari rata-rata dan distribusi
    {
        "query": (
            "Saya punya list nilai [75, 82, 60, 91, 55, 88, 70]. "
            "Saya diminta hitung rata-rata, lalu tampilkan berapa mahasiswa di atas dan di bawah rata-rata. "
            "Apakah ini bisa diselesaikan dengan satu loop atau perlu dua loop?"
        ),
        "relevant_keywords": ["list", "rata-rata", "loop", "for", "akumulasi", "perbandingan"],
        "cognitive": "3PAI",
        "session_id": "eval-058",
        "query_type": "application",
        "context_note": "Operasi statistik pada list — ada di GT_DETAIL_PT09 dan GT_CT13",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 14 — STACK & QUEUE (GT_SUBTOPIK_12)
    # ══════════════════════════════════════════════════════════════════════

    # 059 | application — simulasi stack PUSH/POP/PEEK
    {
        "query": (
            "Simulasikan stack dengan operasi berurutan:\n"
            "PUSH(7), PUSH(2), PUSH(5), POP, PUSH(9), PEEK, POP, POP.\n"
            "Saya tidak yakin apa yang dikembalikan PEEK — apakah elemen terhapus atau tidak?"
        ),
        "relevant_keywords": ["stack", "PUSH", "POP", "PEEK", "LIFO", "top", "simulasi"],
        "cognitive": "2TGI",
        "session_id": "eval-059",
        "query_type": "application",
        "context_note": "Simulasi operasi stack — ada di GT_SUBTOPIK_12",
    },

    # 060 | application — simulasi queue ENQUEUE/DEQUEUE
    {
        "query": (
            "Simulasikan queue dengan operasi:\n"
            "ENQUEUE(P), ENQUEUE(Q), ENQUEUE(R), DEQUEUE, ENQUEUE(S), DEQUEUE, DEQUEUE.\n"
            "Tunjukkan isi queue setelah setiap operasi dan nilai yang dikembalikan DEQUEUE."
        ),
        "relevant_keywords": ["queue", "ENQUEUE", "DEQUEUE", "FIFO", "front", "rear", "simulasi"],
        "cognitive": "2PAR",
        "session_id": "eval-060",
        "query_type": "application",
        "context_note": "Simulasi operasi queue — ada di GT_SUBTOPIK_12",
    },

    # 061 | confusion — mahasiswa tidak tahu pilih stack atau queue
    {
        "query": (
            "Saya mau buat fitur undo di aplikasi teks editor saya. "
            "Setiap aksi disimpan dan bisa di-undo ke aksi sebelumnya. "
            "Apakah saya pakai stack atau queue? Dan mengapa?"
        ),
        "relevant_keywords": ["stack", "queue", "LIFO", "FIFO", "undo", "aplikasi"],
        "cognitive": "3TAI",
        "session_id": "eval-061",
        "query_type": "confusion",
        "context_note": "Mahasiswa memilih stack vs queue untuk fitur undo",
    },

    # 062 | application — cek keseimbangan kurung dengan stack
    {
        "query": (
            "Soal: periksa apakah ekspresi '((a+b)*(c-d))' seimbang kurungnya. "
            "Dosen minta pakai stack. Saya paham LIFO tapi tidak tahu cara terapkan ke masalah kurung ini."
        ),
        "relevant_keywords": ["stack", "kurung", "PUSH", "POP", "seimbang", "LIFO", "ekspresi"],
        "cognitive": "5TGI",
        "session_id": "eval-062",
        "query_type": "application",
        "context_note": "Aplikasi stack untuk cek keseimbangan kurung — ada di GT_SUBTOPIK_12",
    },

    # 063 | gap — mahasiswa tidak tahu implementasi stack dengan array
    {
        "query": (
            "Di slide stack ada pseudocode implementasi dengan array dan variabel 'top'. "
            "Saya tidak mengerti mengapa perlu variabel top — kenapa tidak langsung pakai panjang array? "
            "Apa fungsi variabel top dalam implementasi stack?"
        ),
        "relevant_keywords": ["stack", "implementasi", "array", "top", "PUSH", "POP"],
        "cognitive": "4TGR",
        "session_id": "eval-063",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham peran variabel top dalam implementasi stack berbasis array",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 15 — GRAPH & TREE (GT_SUBTOPIK_13)
    # ══════════════════════════════════════════════════════════════════════

    # 064 | confusion — mahasiswa tidak bisa bedakan tree dan graph biasa
    {
        "query": (
            "Dosen bilang tree itu 'graph khusus'. "
            "Tapi kalau tree adalah graph, kenapa diajarkan terpisah? "
            "Apa yang membuat graph dianggap tree dan bukan hanya graph biasa?"
        ),
        "relevant_keywords": ["tree", "graph", "cycle", "hierarkis", "node", "edge"],
        "cognitive": "4TGI",
        "session_id": "eval-064",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung hubungan hierarki antara tree dan graph",
    },

    # 065 | application — trace Dijkstra pada graph berbobot
    {
        "query": (
            "Graph: A-B(4), A-C(2), B-D(3), C-D(5), D-E(1). "
            "Cari jarak terpendek dari A ke E menggunakan Dijkstra. "
            "Saya sudah coba tapi selalu dapat jalur yang berbeda dari teman."
        ),
        "relevant_keywords": ["Dijkstra", "graph", "jarak terpendek", "bobot", "BFS", "traversal"],
        "cognitive": "3TGI",
        "session_id": "eval-065",
        "query_type": "application",
        "context_note": "Trace Dijkstra ada di GT_SUBTOPIK_13",
    },

    # 066 | comparative — BFS vs DFS urutan kunjungan
    {
        "query": (
            "Graph: A→B, A→C, B→D, B→E, C→F. "
            "Saya trace BFS dari A dan dapat: A, B, C, D, E, F. "
            "Teman trace DFS dari A dan dapat: A, B, D, E, C, F. "
            "Apakah keduanya benar? Dan kenapa urutannya bisa berbeda?"
        ),
        "relevant_keywords": ["BFS", "DFS", "traversal", "queue", "stack", "urutan", "graph"],
        "cognitive": "4TGI",
        "session_id": "eval-066",
        "query_type": "comparative",
        "context_note": "Perbandingan BFS vs DFS ada di GT_SUBTOPIK_13",
    },

    # 067 | gap — mahasiswa tidak tahu bedanya directed dan undirected graph di kasus nyata
    {
        "query": (
            "Di slide graph ada directed dan undirected. "
            "Dosen minta pilih jenis graph yang tepat untuk 'jaringan pertemanan di media sosial'. "
            "Saya tidak yakin — pertemanan kan simetris tapi ada yang following tanpa difollow balik. "
            "Mana yang benar?"
        ),
        "relevant_keywords": ["directed graph", "undirected graph", "edge", "simetris", "media sosial", "relasi"],
        "cognitive": "3PAR",
        "session_id": "eval-067",
        "query_type": "gap",
        "context_note": "Memilih directed vs undirected graph untuk kasus nyata — ada di GT_SUBTOPIK_13",
    },

    # 068 | out-of-scope — binary search tree operasi insert/delete (tidak ada di GT)
    {
        "query": (
            "Saya baca tentang Binary Search Tree (BST) di internet. "
            "Apakah BST diajarkan di CT? Kalau ya, bagaimana cara insert dan search di BST?"
        ),
        "relevant_keywords": ["binary search tree", "tree", "insert", "search", "node", "hierarkis"],
        "cognitive": "5PGR",
        "session_id": "eval-068",
        "query_type": "out_of_scope",
        "context_note": "BST operasi spesifik tidak ada di GT CT — hanya konsep tree dasar",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 16 — KOMPLEKSITAS & BIG-O (GT_SUBTOPIK_14)
    # ══════════════════════════════════════════════════════════════════════

    # 069 | application — identifikasi Big-O dari tiga kode
    {
        "query": (
            "Tentukan kompleksitas waktu dari:\n"
            "a) x = arr[5]\n"
            "b) for i=1 to n: print(arr[i])\n"
            "c) for i=1 to n: for j=1 to n: print(i+j)\n"
            "Saya dapat a=O(1), b=O(n), c=O(n²). Apakah benar?"
        ),
        "relevant_keywords": ["Big-O", "O(1)", "O(n)", "O(n^2)", "loop", "nested loop", "kompleksitas"],
        "cognitive": "3PAI",
        "session_id": "eval-069",
        "query_type": "application",
        "context_note": "Identifikasi Big-O dari kode ada di GT_SUBTOPIK_14",
    },

    # 070 | confusion — mahasiswa salah simplifikasi O(n² + n)
    {
        "query": (
            "Program saya punya satu nested loop O(n²) dan satu loop biasa O(n) yang berjalan setelahnya. "
            "Saya tulis kompleksitasnya O(n²+n). Tapi dosen coret itu dan tulis O(n²). "
            "Kenapa n-nya dibuang?"
        ),
        "relevant_keywords": ["Big-O", "penyederhanaan", "O(n^2)", "O(n)", "suku dominan", "aturan"],
        "cognitive": "4PAR",
        "session_id": "eval-070",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung aturan penyederhanaan Big-O — buang suku rendah",
    },

    # 071 | application — analisis Big-O loop dengan pengali dua
    {
        "query": (
            "Tentukan Big-O pseudocode ini:\n"
            "  for i=1 to n do\n    j=1\n    while (j < n) do\n      print(i,j)\n      j = j*2\n"
            "Saya kira jawabannya O(n²) karena ada nested loop. Tapi dosen bilang bukan."
        ),
        "relevant_keywords": ["Big-O", "O(n log n)", "while", "logaritmik", "pengali", "analisis"],
        "cognitive": "5TGI",
        "session_id": "eval-071",
        "query_type": "application",
        "context_note": "Big-O loop dengan j*=2 ada di GT_SUBTOPIK_14",
    },

    # 072 | comparative — bubble sort vs merge sort untuk data besar
    {
        "query": (
            "Saya punya 100.000 data yang perlu diurutkan. "
            "Bubble sort terasa lambat. Dosen menyarankan merge sort. "
            "Kalau keduanya bisa menghasilkan array terurut, kenapa merge sort lebih baik untuk data besar?"
        ),
        "relevant_keywords": ["bubble sort", "merge sort", "O(n^2)", "O(n log n)", "kompleksitas", "sorting"],
        "cognitive": "5PAR",
        "session_id": "eval-072",
        "query_type": "comparative",
        "context_note": "Perbandingan sorting berdasarkan kompleksitas ada di GT_SUBTOPIK_14",
    },

    # 073 | gap — space complexity tidak dipahami
    {
        "query": (
            "Saya hanya tahu time complexity. Tapi dosen juga tanya space complexity dari algoritme saya. "
            "Apa itu space complexity dan bagaimana cara menganalisisnya? "
            "Apakah O(1) space berarti tidak pakai memori sama sekali?"
        ),
        "relevant_keywords": ["space complexity", "memori", "O(1)", "O(n)", "algoritme", "rekursi"],
        "cognitive": "4TGR",
        "session_id": "eval-073",
        "query_type": "gap",
        "context_note": "Space complexity dibahas di GT_SUBTOPIK_14 dalam konteks time vs space",
    },

    # 074 | out-of-scope — amortized complexity (tidak ada di GT)
    {
        "query": (
            "Saya baca bahwa append ke Python list itu O(1) amortized, bukan O(1) biasa. "
            "Apa itu amortized complexity dan kenapa append tidak selalu O(1)?"
        ),
        "relevant_keywords": ["amortized", "append", "list", "O(1)", "kompleksitas", "array"],
        "cognitive": "6TGR",
        "session_id": "eval-074",
        "query_type": "out_of_scope",
        "context_note": "Amortized complexity tidak ada di GT CT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 17 — SEARCHING & SORTING
    # ══════════════════════════════════════════════════════════════════════

    # 075 | application — trace binary search
    {
        "query": (
            "Array terurut: [3, 7, 12, 18, 25, 31, 44, 56, 67, 89]. Cari angka 25. "
            "Saya trace binary search dan butuh 4 langkah. Teman hanya butuh 2 langkah. "
            "Tolong trace dari awal untuk verifikasi berapa langkah yang benar."
        ),
        "relevant_keywords": ["binary search", "trace", "low", "high", "mid", "langkah", "array terurut"],
        "cognitive": "3TGR",
        "session_id": "eval-075",
        "query_type": "application",
        "context_note": "Trace binary search step by step",
    },

    # 076 | confusion — mahasiswa salah kira binary search bisa untuk data tidak terurut
    {
        "query": (
            "Saya coba binary search untuk array [5, 3, 8, 1, 9] dan hasilnya salah — "
            "elemen yang ada tidak ketemu. Padahal algoritmanya sudah benar. "
            "Kenapa binary search saya gagal?"
        ),
        "relevant_keywords": ["binary search", "terurut", "prasyarat", "array", "linear search", "O(log n)"],
        "cognitive": "2PAI",
        "session_id": "eval-076",
        "query_type": "confusion",
        "context_note": "Mahasiswa tidak tahu prasyarat data harus terurut untuk binary search",
    },

    # 077 | application — trace bubble sort lengkap semua pass
    {
        "query": (
            "Trace bubble sort untuk array [5, 2, 8, 1, 9] — tunjukkan semua pertukaran di setiap pass "
            "sampai array terurut. Saya bingung kapan pass berhenti dan berapa total pass yang dibutuhkan."
        ),
        "relevant_keywords": ["bubble sort", "pass", "tukar", "trace", "array", "iterasi"],
        "cognitive": "2TAI",
        "session_id": "eval-077",
        "query_type": "application",
        "context_note": "Trace bubble sort semua pass",
    },

    # 078 | application — trace selection sort
    {
        "query": (
            "Trace selection sort untuk array [4, 7, 2, 9, 1, 5]. "
            "Saya tidak yakin cara kerja selection sort — apakah dia cari minimum atau maximum dulu? "
            "Dan apakah hasilnya ascending atau descending?"
        ),
        "relevant_keywords": ["selection sort", "minimum", "tukar", "trace", "ascending", "iterasi"],
        "cognitive": "3PGI",
        "session_id": "eval-078",
        "query_type": "application",
        "context_note": "Trace selection sort ascending",
    },

    # 079 | comparative — kapan linear search lebih baik dari binary search
    {
        "query": (
            "Teman saya selalu pakai binary search karena katanya lebih cepat. "
            "Tapi dosen bilang ada kasus di mana linear search lebih tepat. "
            "Kapan linear search lebih baik atau lebih tepat digunakan daripada binary search?"
        ),
        "relevant_keywords": ["binary search", "linear search", "terurut", "O(n)", "O(log n)", "prasyarat"],
        "cognitive": "4PAI",
        "session_id": "eval-079",
        "query_type": "comparative",
        "context_note": "Perbandingan kapan linear vs binary search lebih tepat",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 18 — CROSS-TOPIK INTEGRATIF
    # ══════════════════════════════════════════════════════════════════════

    # 080 | cross_topic — dekomposisi + Big-O (desain sistem dengan analisis kompleksitas)
    {
        "query": (
            "Saya mau buat sistem pencarian kontak dari 10.000 data. "
            "Dosen minta saya dekomposisi dulu, baru pilih algoritme berdasarkan Big-O. "
            "Dari perspektif CT, bagaimana cara menggabungkan dekomposisi dan analisis Big-O "
            "untuk memilih solusi yang tepat?"
        ),
        "relevant_keywords": ["dekomposisi", "Big-O", "algoritme", "pencarian", "binary search", "abstraksi"],
        "cognitive": "6PAR",
        "session_id": "eval-080",
        "query_type": "cross_topic",
        "context_note": "Integrasi dekomposisi + Big-O untuk desain solusi",
    },

    # 081 | cross_topic — stack + rekursi (hubungan antara call stack dan stack data structure)
    {
        "query": (
            "Waktu belajar rekursi, dosen bilang setiap pemanggilan rekursif disimpan di 'call stack'. "
            "Waktu belajar struktur data, kita belajar 'stack' sebagai ADT. "
            "Apakah call stack dalam rekursi sama dengan stack yang kita pelajari di struktur data?"
        ),
        "relevant_keywords": ["rekursi", "call stack", "stack", "LIFO", "memori", "fungsi"],
        "cognitive": "5TGR",
        "session_id": "eval-081",
        "query_type": "cross_topic",
        "context_note": "Hubungan call stack rekursi dengan stack sebagai struktur data",
    },

    # 082 | cross_topic — pattern recognition + Big-O (mengenali kompleksitas dari pola kode)
    {
        "query": (
            "Dosen bilang kita bisa gunakan pattern recognition untuk langsung tahu Big-O dari kode "
            "tanpa menghitung detail. Misalnya 'lihat nested loop langsung tahu O(n²)'. "
            "Apa saja pola-pola kode yang bisa langsung saya kenali Big-O-nya?"
        ),
        "relevant_keywords": ["pattern recognition", "Big-O", "nested loop", "O(n^2)", "pola", "O(log n)"],
        "cognitive": "5PAI",
        "session_id": "eval-082",
        "query_type": "cross_topic",
        "context_note": "Menggunakan pattern recognition untuk identifikasi cepat Big-O",
    },

    # 083 | cross_topic — abstraksi + fungsi (fungsi sebagai implementasi abstraksi)
    {
        "query": (
            "Kalau abstraksi adalah 'sembunyikan detail yang tidak perlu', "
            "apakah setiap fungsi yang saya buat otomatis mengimplementasikan abstraksi? "
            "Atau ada bedanya antara fungsi yang mengimplementasikan abstraksi dan yang tidak?"
        ),
        "relevant_keywords": ["abstraksi", "fungsi", "detail", "implementasi", "modularitas", "DRY"],
        "cognitive": "4TAI",
        "session_id": "eval-083",
        "query_type": "cross_topic",
        "context_note": "Hubungan antara abstraksi sebagai pilar CT dan fungsi sebagai implementasinya",
    },

    # 084 | cross_topic — rekursi + pohon (tree traversal rekursif)
    {
        "query": (
            "Dosen bilang DFS di graph biasanya diimplementasikan rekursif. "
            "Saya paham rekursi dan paham DFS secara konsep, "
            "tapi tidak paham kenapa DFS alami untuk diimplementasikan rekursif "
            "sementara BFS tidak?"
        ),
        "relevant_keywords": ["DFS", "rekursi", "BFS", "stack", "tree", "traversal"],
        "cognitive": "5TGI",
        "session_id": "eval-084",
        "query_type": "cross_topic",
        "context_note": "Hubungan antara rekursi dan DFS traversal",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 19 — SOAL KOMBINASI NYATA (situasi mahasiswa)
    # ══════════════════════════════════════════════════════════════════════

    # 085 | application — palindrome dengan dua pointer
    {
        "query": (
            "Saya diminta cek apakah kata 'katak' adalah palindrom tanpa pakai fungsi string bawaan. "
            "Saya coba dengan loop biasa tapi hasilnya salah untuk beberapa input. "
            "Bagaimana algoritme yang benar untuk cek palindrom?"
        ),
        "relevant_keywords": ["string", "palindrom", "loop", "indeks", "perbandingan", "while"],
        "cognitive": "4TAI",
        "session_id": "eval-085",
        "query_type": "application",
        "context_note": "Cek palindrom dengan loop — menggabungkan string/list dan perulangan",
    },

    # 086 | application — cek bilangan prima optimal
    {
        "query": (
            "Saya buat program cek bilangan prima dengan loop dari 2 sampai n. "
            "Untuk n=1.000.000 sangat lambat. Dosen bilang ada cara lebih cepat. "
            "Bagaimana cara mengoptimasi pengecekan bilangan prima?"
        ),
        "relevant_keywords": ["bilangan prima", "loop", "akar kuadrat", "O(sqrt(n))", "optimasi", "algoritme"],
        "cognitive": "5TGR",
        "session_id": "eval-086",
        "query_type": "application",
        "context_note": "Optimasi cek prima dengan akar kuadrat",
    },

    # 087 | application — hitung jumlah digit tanpa string
    {
        "query": (
            "Tugas: hitung berapa digit bilangan 98765 tanpa mengubahnya ke string. "
            "Hanya boleh pakai operasi aritmatika. "
            "Saya tahu pembagian berulang bisa dipakai tapi tidak tahu cara pseudocode-nya."
        ),
        "relevant_keywords": ["digit", "while", "div", "modulo", "aritmatika", "pseudocode"],
        "cognitive": "3PAR",
        "session_id": "eval-087",
        "query_type": "application",
        "context_note": "Hitung jumlah digit dengan pembagian berulang — kombinasi while dan div",
    },

    # 088 | scenario — desain sistem antrian prioritas
    {
        "query": (
            "Saya diminta desain sistem antrian loket bank: nasabah biasa antri normal, "
            "lansia dan ibu hamil diprioritaskan. "
            "Struktur data apa yang paling cocok untuk kasus ini?"
        ),
        "relevant_keywords": ["queue", "priority queue", "FIFO", "stack", "struktur data", "LIFO"],
        "cognitive": "5PAI",
        "session_id": "eval-088",
        "query_type": "scenario",
        "context_note": "Desain sistem antrian prioritas menggunakan struktur data yang tepat",
    },

    # 089 | scenario — rekursi untuk kombinasi
    {
        "query": (
            "Soal ujian: tulis fungsi rekursif untuk C(5,2) menggunakan sifat Pascal: "
            "C(n,k) = C(n-1,k-1) + C(n-1,k), dengan C(n,0)=C(n,n)=1. "
            "Saya tidak tahu cara trace-nya dan tidak yakin berapa hasilnya."
        ),
        "relevant_keywords": ["kombinasi", "rekursi", "Pascal", "base case", "C(n,k)", "trace"],
        "cognitive": "5PGR",
        "session_id": "eval-089",
        "query_type": "scenario",
        "context_note": "Kombinasi rekursif dengan segitiga Pascal",
    },

    # 090 | scenario — memilih sorting untuk data hampir terurut
    {
        "query": (
            "Saya punya data nilai ujian 30.000 mahasiswa yang hampir terurut — "
            "hanya beberapa data yang posisinya tidak tepat. "
            "Dari bubble sort, selection sort, dan merge sort, mana yang paling efisien?"
        ),
        "relevant_keywords": ["bubble sort", "selection sort", "merge sort", "nearly sorted", "O(n)", "efisiensi"],
        "cognitive": "5TAI",
        "session_id": "eval-090",
        "query_type": "scenario",
        "context_note": "Pilih sorting terbaik untuk data hampir terurut",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 20 — PERTANYAAN TRICKY / EDGE CASE
    # ══════════════════════════════════════════════════════════════════════

    # 091 | confusion — mahasiswa kira return bisa dipanggil lebih dari sekali
    {
        "query": (
            "Fungsi saya punya dua return: pertama di dalam if, kedua di akhir. "
            "Untuk input tertentu, kondisi if true. Apakah program akan mengeksekusi return pertama "
            "lalu lanjut ke return kedua? Atau stop di return pertama?"
        ),
        "relevant_keywords": ["return", "fungsi", "kondisi", "if", "eksekusi", "alur"],
        "cognitive": "2PAI",
        "session_id": "eval-091",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah kira dua return dieksekusi berurutan",
    },

    # 092 | confusion — base case faktorial n=0 vs n=1
    {
        "query": (
            "Fungsi faktorial saya hanya punya base case n==0. "
            "Teman bilang harus tambahkan n==1 juga. "
            "Tapi hasil faktorial(4) saya tetap 24 tanpa n==1. "
            "Apakah n==1 wajib sebagai base case kedua atau tidak?"
        ),
        "relevant_keywords": ["faktorial", "base case", "rekursi", "n=0", "n=1", "fungsi"],
        "cognitive": "4TGI",
        "session_id": "eval-092",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung apakah n==1 wajib menjadi base case faktorial",
    },

    # 093 | gap — mahasiswa tidak paham infinite loop vs loop yang selesai lama
    {
        "query": (
            "Program saya sudah jalan 5 menit untuk input besar dan belum selesai. "
            "Apakah itu infinite loop atau hanya lambat? "
            "Bagaimana cara membedakan antara algoritme yang sangat lambat dengan yang benar-benar infinite loop?"
        ),
        "relevant_keywords": ["infinite loop", "finiteness", "kompleksitas", "while", "kondisi", "algoritme"],
        "cognitive": "3TAI",
        "session_id": "eval-093",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak bisa bedakan infinite loop dengan algoritme lambat",
    },

    # 094 | application — Fibonacci(6) trace rekursif
    {
        "query": (
            "Trace rekursif untuk fungsi:\n"
            "  function f(n): if n<=2 return 1 else return f(n-1)+f(n-2)\n"
            "Hitung f(6). Saya dapat 11, teman dapat 8. "
            "Tolong trace lengkap dan tunjukkan mana yang benar."
        ),
        "relevant_keywords": ["Fibonacci", "rekursi", "trace", "f(6)", "base case", "deret"],
        "cognitive": "3TGR",
        "session_id": "eval-094",
        "query_type": "application",
        "context_note": "Trace Fibonacci rekursif f(6) — ada di GT_SUBTOPIK_11",
    },

    # 095 | application — pseudocode cari nilai maksimum dari 5 input
    {
        "query": (
            "Tulis pseudocode untuk membaca 5 bilangan dari input dan tampilkan nilai terbesarnya. "
            "Saya bingung apakah harus simpan semua dulu ke array atau bisa langsung bandingkan satu per satu "
            "tanpa menyimpan ke array."
        ),
        "relevant_keywords": ["pseudocode", "maksimum", "for", "read", "perbandingan", "array"],
        "cognitive": "2TGI",
        "session_id": "eval-095",
        "query_type": "application",
        "context_note": "Pseudocode cari nilai maksimum ada di GT_SUBTOPIK_05",
    },

    # 096 | gap — mahasiswa tidak tahu apa itu pass by value dalam pseudocode
    {
        "query": (
            "Di fungsi saya, saya ubah nilai parameter dan berharap nilai aslinya ikut berubah. "
            "Tapi setelah fungsi selesai, nilai aslinya tidak berubah. "
            "Kenapa perubahan di dalam fungsi tidak mempengaruhi variabel di luar?"
        ),
        "relevant_keywords": ["parameter", "fungsi", "pass by value", "variabel", "scope", "return"],
        "cognitive": "3TAI",
        "session_id": "eval-096",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak paham pass by value dalam fungsi pseudocode",
    },

    # 097 | out-of-scope — exception handling (tidak ada di GT)
    {
        "query": (
            "Di program saya, kalau user masukkan huruf padahal yang diharapkan angka, program crash. "
            "Dosen bilang perlu 'error handling'. Apakah error handling atau exception handling "
            "diajarkan di CT? Bagaimana cara menanganinya di pseudocode?"
        ),
        "relevant_keywords": ["error handling", "input", "validasi", "percabangan", "pseudocode", "while"],
        "cognitive": "4PAR",
        "session_id": "eval-097",
        "query_type": "out_of_scope",
        "context_note": "Exception handling tidak ada di GT CT — hanya validasi dengan percabangan/loop",
    },

    # 098 | cross_topic — modulo + pattern recognition (siklus digit)
    {
        "query": (
            "Soal: hari ini Senin. Hari apa 1000 hari dari sekarang? "
            "Dosen bilang ini tentang pattern recognition dan modulo. "
            "Saya paham modulo tapi tidak tahu bagaimana menghubungkannya dengan hari dalam seminggu."
        ),
        "relevant_keywords": ["modulo", "siklus", "pattern recognition", "hari", "periodik", "pola"],
        "cognitive": "2PGR",
        "session_id": "eval-098",
        "query_type": "cross_topic",
        "context_note": "Aplikasi modulo untuk siklus hari — ada di GT_SUBTOPIK_04",
    },

    # 099 | application — pseudocode login dengan maksimal 3 percobaan
    {
        "query": (
            "Saya diminta buat pseudocode sistem login dengan maksimal 3 percobaan. "
            "Lebih dari 3 kali salah, akun terkunci. "
            "Saya tidak tahu bagaimana menghitung percobaan dan kapan berhenti."
        ),
        "relevant_keywords": ["while", "percobaan", "boolean", "validasi", "percabangan", "for"],
        "cognitive": "3TAR",
        "session_id": "eval-099",
        "query_type": "application",
        "context_note": "Sistem login dengan batas percobaan — kombinasi while dan if",
    },

    # 100 | scenario — memilih representasi graph yang tepat
    {
        "query": (
            "Saya mau representasikan peta kota dengan ratusan persimpangan dan ribuan jalan. "
            "Setiap jalan punya jarak. Rata-rata setiap persimpangan hanya terhubung ke 4 jalan lain. "
            "Mana yang lebih efisien: adjacency matrix atau adjacency list? Dan kenapa?"
        ),
        "relevant_keywords": ["adjacency matrix", "adjacency list", "graph", "sparse", "dense", "memori"],
        "cognitive": "5PAR",
        "session_id": "eval-100",
        "query_type": "scenario",
        "context_note": "Pilih representasi graph untuk sparse graph peta kota — ada di GT_SUBTOPIK_13",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK 21 — 10 TAMBAHAN (total 110)
    # ══════════════════════════════════════════════════════════════════════

    # 101 | application — modulo siklus digit terakhir 3^n
    {
        "query": (
            "Soal: tentukan digit terakhir dari 3^100. "
            "Saya coba hitung 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243... "
            "Saya lihat ada pola tapi tidak tahu cara formalkan dan aplikasikan ke 3^100."
        ),
        "relevant_keywords": ["modulo", "digit terakhir", "pola", "siklus", "perpangkatan", "deret"],
        "cognitive": "4PGR",
        "session_id": "eval-101",
        "query_type": "application",
        "context_note": "Digit terakhir 3^n menggunakan siklus modulo — ada di GT_SUBTOPIK_04",
    },

    # 102 | confusion — mahasiswa salah mengira coverage 1.0 = sistem sempurna
    {
        "query": (
            "Dosen bilang algoritme binary search O(log n) sangat efisien. "
            "Tapi untuk array berisi 8 elemen, binary search butuh log₂(8)=3 langkah, "
            "sementara linear search rata-rata butuh 4 langkah. "
            "Bedanya tidak jauh. Apakah binary search benar-benar sepadan untuk ukuran data kecil?"
        ),
        "relevant_keywords": ["binary search", "linear search", "O(log n)", "O(n)", "n kecil", "efisiensi"],
        "cognitive": "5TAR",
        "session_id": "eval-102",
        "query_type": "confusion",
        "context_note": "Mahasiswa mempertanyakan keuntungan binary search untuk data kecil",
    },

    # 103 | gap — mahasiswa tidak paham bedanya rekursi langsung vs tidak langsung
    {
        "query": (
            "Rekursi itu fungsi yang memanggil dirinya sendiri. "
            "Tapi dosen menyebut ada 'rekursi tidak langsung' — fungsi A memanggil B, B memanggil A. "
            "Apakah CT mengajarkan rekursi tidak langsung? "
            "Dan apakah base case-nya sama dengan rekursi langsung?"
        ),
        "relevant_keywords": ["rekursi", "rekursi tidak langsung", "base case", "fungsi", "pemanggilan"],
        "cognitive": "5TGR",
        "session_id": "eval-103",
        "query_type": "gap",
        "context_note": "Rekursi tidak langsung tidak eksplisit di GT — hanya rekursi langsung yang diajarkan",
    },

    # 104 | application — deret selisih bertingkat (ada di GT_SUBTOPIK_03)
    {
        "query": (
            "Diberikan barisan: 1, 3, 7, 13, 21, 31. "
            "Saya hitung selisihnya: 2, 4, 6, 8, 10 — itu selisih bertingkat pertama. "
            "Lalu selisih dari selisih: 2, 2, 2, 2 — konstan. "
            "Apa artinya dan bagaimana saya temukan rumus suku ke-n dari barisan aslinya?"
        ),
        "relevant_keywords": ["pola bilangan", "selisih bertingkat", "deret", "rumus", "pattern recognition", "kuadrat"],
        "cognitive": "4TGR",
        "session_id": "eval-104",
        "query_type": "application",
        "context_note": "Pola selisih bertingkat ada di GT_SUBTOPIK_03",
    },

    # 105 | confusion — mahasiswa kira while true = bug
    {
        "query": (
            "Saya lihat di kode server web ada while(true) yang jalan terus tanpa henti. "
            "Kata saya itu infinite loop dan harus diperbaiki. "
            "Tapi kata teman itu sengaja. Apakah while(true) selalu berarti bug?"
        ),
        "relevant_keywords": ["while", "infinite loop", "finiteness", "algoritme", "kondisi", "server"],
        "cognitive": "3PAR",
        "session_id": "eval-105",
        "query_type": "confusion",
        "context_note": "Mahasiswa kira while(true) selalu merupakan bug — kaitkan dengan finiteness",
    },

    # 106 | cross_topic — abstraksi + pseudocode (kerangka abstraksi PGBD di soal)
    {
        "query": (
            "Di soal abstraksi, dosen minta saya identifikasi PROBLEM, GOAL, BATASAN, dan DATA. "
            "Saya bingung bedanya PROBLEM dengan GOAL — keduanya terasa seperti 'apa yang mau dicapai'. "
            "Bisa jelaskan perbedaannya dengan contoh?"
        ),
        "relevant_keywords": ["abstraksi", "problem", "goal", "batasan", "data", "formulasi"],
        "cognitive": "3PGR",
        "session_id": "eval-106",
        "query_type": "cross_topic",
        "context_note": "Kerangka PGBD ada di GT_SUBTOPIK_02",
    },

    # 107 | application — pseudocode cari semua faktor bilangan
    {
        "query": (
            "Tugas: buat pseudocode untuk menampilkan semua faktor bilangan n. "
            "Contoh: faktor dari 12 adalah 1, 2, 3, 4, 6, 12. "
            "Saya tidak tahu harus loop sampai n atau ada cara lebih efisien."
        ),
        "relevant_keywords": ["for", "modulo", "faktor", "loop", "bilangan", "pseudocode"],
        "cognitive": "3PAI",
        "session_id": "eval-107",
        "query_type": "application",
        "context_note": "Cari semua faktor bilangan dengan loop dan modulo",
    },

    # 108 | scenario — desain database sederhana untuk nilai mahasiswa
    {
        "query": (
            "Saya diminta desain penyimpanan data nilai 500 mahasiswa untuk 10 mata kuliah. "
            "Dosen minta saya pakai abstraksi dulu sebelum tentukan struktur data. "
            "Dari perspektif CT, bagaimana proses abstraksi dan apa struktur data yang tepat?"
        ),
        "relevant_keywords": ["abstraksi", "struktur data", "array", "list", "dekomposisi", "formulasi"],
        "cognitive": "5PAR",
        "session_id": "eval-108",
        "query_type": "scenario",
        "context_note": "Desain penyimpanan data dengan abstraksi CT lalu pilih struktur data",
    },

    # 109 | out-of-scope — hash table / dictionary (tidak ada di GT CT)
    {
        "query": (
            "Saya butuh cari nama mahasiswa dari NIM-nya dengan cepat O(1). "
            "Di Python saya pakai dictionary. Apakah dictionary diajarkan di CT? "
            "Kalau tidak, struktur data apa dari CT yang bisa mencapai pencarian O(1)?"
        ),
        "relevant_keywords": ["dictionary", "hash table", "O(1)", "pencarian", "array", "list"],
        "cognitive": "6PAI",
        "session_id": "eval-109",
        "query_type": "out_of_scope",
        "context_note": "Hash table/dictionary tidak ada di GT CT — mahasiswa akan dapatkan jawaban parsial",
    },

    # 110 | cross_topic — semua pilar CT dalam satu soal desain sistem
    {
        "query": (
            "Saya diminta rancang algoritme sederhana untuk mengurutkan dan mencari nama mahasiswa "
            "dari file yang berisi 1000 data. "
            "Dosen minta saya tunjukkan: (1) abstraksi masalah, (2) dekomposisi langkah, "
            "(3) pola yang dikenali, (4) algoritme yang dipilih beserta alasannya."
        ),
        "relevant_keywords": ["abstraksi", "dekomposisi", "pattern recognition", "algoritme", "sorting", "searching"],
        "cognitive": "6TGR",
        "session_id": "eval-110",
        "query_type": "cross_topic",
        "context_note": "Soal integratif semua 4 pilar CT sekaligus — menguji kemampuan RAG retrieve multi-dokumen",
    },

]


# ── Sanity check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert len(TEST_CASES) == 110, f"Expected 110, got {len(TEST_CASES)}"
    ids = [tc["session_id"] for tc in TEST_CASES]
    assert len(set(ids)) == 110, "Duplicate session_id!"
    # Pastikan tidak ada reference_answer
    for tc in TEST_CASES:
        assert "reference_answer" not in tc, f"reference_answer ditemukan di {tc['session_id']}"
    types = {}
    for tc in TEST_CASES:
        t = tc["query_type"]
        types[t] = types.get(t, 0) + 1
    print(f"✅ {len(TEST_CASES)} test cases")
    print(f"   Types: {types}")
    cogs = set(tc["cognitive"] for tc in TEST_CASES)
    print(f"   Cognitives: {sorted(cogs)}")
