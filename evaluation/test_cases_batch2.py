"""
evaluation/test_cases_batch2.py
────────────────────────────────
100 test case TAMBAHAN (batch 2) untuk RAG evaluation suite.
Session ID: eval-201 hingga eval-300.

Semua topik dan context_note BERBEDA dari batch 1 (eval-001 s/d eval-110).
Desain mengikuti prinsip yang sama:
  - relevant_keywords campuran relevan + false trail
  - Tidak ada reference_answer
  - Query dari perspektif mahasiswa yang genuinely bingung
  - ~30% out_of_scope / gap untuk membuat metrik tidak trivially 1.0

Cara pakai (gabung dengan batch 1):
    from evaluation.test_cases      import TEST_CASES as TC1
    from evaluation.test_cases_batch2 import TEST_CASES as TC2
    ALL_CASES = TC1 + TC2
"""

from typing import Dict, List

TEST_CASES: List[Dict] = [

    # ══════════════════════════════════════════════════════════════════════
    # BLOK A — CT & KURIKULUM (sudut pandang baru)
    # ══════════════════════════════════════════════════════════════════════

    # 201
    {
        "query": (
            "Di ujian akhir semester saya, ada soal yang meminta saya "
            "mengklasifikasikan suatu aktivitas ke dalam salah satu pilar AADP. "
            "Aktivitasnya: 'mengelompokkan soal-soal serupa sebelum mengerjakan'. "
            "Saya tidak yakin ini Abstraksi atau Pattern Recognition karena keduanya "
            "terasa tentang 'mengelompokkan'. Bagaimana cara membedakannya?"
        ),
        "relevant_keywords": ["abstraksi", "pattern recognition", "pengelompokan", "AADP", "pilar CT", "dekomposisi"],
        "cognitive": "3PAR",
        "session_id": "eval-201",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung klasifikasi aktivitas ke pilar AADP yang tepat antara abstraksi dan pattern recognition",
    },

    # 202
    {
        "query": (
            "Dosen saya bilang CT bisa diterapkan bahkan tanpa komputer. "
            "Saya coba pikirkan contohnya tapi selalu berujung ke hal yang "
            "melibatkan teknologi. Apa contoh konkret penerapan CT "
            "dalam kehidupan sehari-hari yang sama sekali tidak melibatkan komputer?"
        ),
        "relevant_keywords": ["computational thinking", "kehidupan sehari-hari", "non-komputer", "dekomposisi", "algoritme", "abstraksi"],
        "cognitive": "2TGI",
        "session_id": "eval-202",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak bisa membayangkan CT tanpa komputer",
    },

    # 203
    {
        "query": (
            "Saya baru tahu ada istilah 'unplugged activities' dalam pengajaran CT. "
            "Katanya CT bisa diajarkan tanpa komputer sama sekali. "
            "Saya penasaran — apakah IPB menggunakan pendekatan ini di mata kuliah CT? "
            "Dan apa contoh konkret unplugged activity untuk mengajarkan algoritme?"
        ),
        "relevant_keywords": ["CT unplugged", "algoritme", "pengajaran CT", "aktivitas", "dekomposisi", "flowchart"],
        "cognitive": "1TGR",
        "session_id": "eval-203",
        "query_type": "out_of_scope",
        "context_note": "Unplugged CT activities tidak eksplisit ada di GT — pertanyaan tentang pedagogi spesifik",
    },

    # 204
    {
        "query": (
            "Saya sedang mengerjakan soal latihan: 'rancang solusi untuk masalah "
            "penjadwalan kuliah di sebuah universitas kecil dengan 10 ruang, "
            "50 mata kuliah, dan 200 mahasiswa.' "
            "Saya tidak tahu harus mulai dari mana. Bagaimana cara menerapkan "
            "keempat pilar CT secara berurutan untuk masalah ini?"
        ),
        "relevant_keywords": ["dekomposisi", "abstraksi", "algoritme", "pattern recognition", "AADP", "penjadwalan"],
        "cognitive": "4PAR",
        "session_id": "eval-204",
        "query_type": "application",
        "context_note": "Soal integratif AADP untuk masalah penjadwalan — angle berbeda dari batch 1",
    },

    # 205
    {
        "query": (
            "Di slide CT ada konsep 'computational artifact' — sesuatu yang dibuat "
            "dengan bantuan komputer. Tapi saya tidak ngerti apa hubungannya dengan "
            "belajar CT. Apakah kita juga akan membuat computational artifact "
            "dalam mata kuliah ini? Dan apa contoh konkretnya?"
        ),
        "relevant_keywords": ["computational artifact", "CT", "produk", "algoritme", "block programming", "kode"],
        "cognitive": "1PAI",
        "session_id": "eval-205",
        "query_type": "out_of_scope",
        "context_note": "Computational artifact sebagai produk CT — konsep ini mungkin tidak eksplisit di GT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK B — DEKOMPOSISI (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 206
    {
        "query": (
            "Saya punya tugas mendekomposisi masalah 'buat aplikasi kalkulator sederhana'. "
            "Saya sudah pecah jadi: input → hitung → output. Tapi teman saya pecah "
            "jadi 15 sub-masalah yang sangat kecil-kecil. "
            "Mana yang lebih benar? Apakah ada standar seberapa detail dekomposisi yang baik?"
        ),
        "relevant_keywords": ["dekomposisi", "granularitas", "sub-masalah", "kalkulator", "modular", "abstraksi"],
        "cognitive": "3TAI",
        "session_id": "eval-206",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung seberapa dalam level dekomposisi yang benar — angle berbeda dari batch 1",
    },

    # 207
    {
        "query": (
            "Di soal dekomposisi saya diminta analisis masalah berikut: "
            "'Sebuah toko online menerima 1000 pesanan per hari. "
            "Setiap pesanan perlu diverifikasi pembayaran, dikemas, dan dikirim.' "
            "Dosen minta saya gambar hierarki dekomposisi. "
            "Saya bingung cara membuat hierarki yang benar — apa aturannya?"
        ),
        "relevant_keywords": ["dekomposisi", "hierarki", "sub-masalah", "toko online", "modular", "verifikasi"],
        "cognitive": "2PGR",
        "session_id": "eval-207",
        "query_type": "application",
        "context_note": "Membuat hierarki dekomposisi untuk sistem e-commerce",
    },

    # 208
    {
        "query": (
            "Dosen bilang dekomposisi yang baik menghasilkan sub-masalah yang "
            "'loosely coupled' dan 'highly cohesive'. "
            "Saya dengar istilah ini di mata kuliah software engineering. "
            "Apakah prinsip ini sama persis dengan dekomposisi di CT, "
            "atau ada perbedaannya?"
        ),
        "relevant_keywords": ["dekomposisi", "loosely coupled", "cohesive", "software engineering", "modular", "sub-masalah"],
        "cognitive": "5TAR",
        "session_id": "eval-208",
        "query_type": "cross_topic",
        "context_note": "Hubungan prinsip dekomposisi CT dengan coupling/cohesion di software engineering",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK C — ABSTRAKSI (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 209
    {
        "query": (
            "Saya punya soal: 'Buat abstraksi dari sistem peminjaman buku perpustakaan.' "
            "Langkah pertama kata dosen adalah identifikasi GOAL. "
            "Saya tulis goalnya: 'agar buku bisa dipinjam'. "
            "Tapi dosen bilang itu bukan goal yang baik untuk abstraksi CT. "
            "Apa yang salah dengan goal saya dan bagaimana goal yang benar?"
        ),
        "relevant_keywords": ["abstraksi", "goal", "formulasi", "PGBD", "batasan", "data", "perpustakaan"],
        "cognitive": "2PAR",
        "session_id": "eval-209",
        "query_type": "confusion",
        "context_note": "Mahasiswa salah mendefinisikan goal dalam kerangka abstraksi PGBD",
    },

    # 210
    {
        "query": (
            "Soal abstraksi dari GT: ada masalah 2 lift berkapasitas 30 kg dengan "
            "9 berang-berang. Saya sudah bisa identifikasi goalnya. "
            "Tapi saya bingung apa yang dimaksud 'batasan' dalam kerangka PGBD. "
            "Apa perbedaan antara DATA dan BATASAN dalam konteks soal lift ini?"
        ),
        "relevant_keywords": ["abstraksi", "PGBD", "batasan", "data", "goal", "lift", "formulasi"],
        "cognitive": "3PGI",
        "session_id": "eval-210",
        "query_type": "application",
        "context_note": "Membedakan DATA vs BATASAN dalam kerangka PGBD untuk soal lift berang-berang",
    },

    # 211
    {
        "query": (
            "Saya sedang belajar abstraksi dan bingung dengan konsep 'level abstraksi'. "
            "Katanya abstraksi bisa dilakukan di berbagai level. "
            "Misalnya level rendah (detail hardware) vs level tinggi (konsep bisnis). "
            "Dalam konteks CT yang kita pelajari, level abstraksi mana yang relevan "
            "dan bagaimana cara memilihnya?"
        ),
        "relevant_keywords": ["abstraksi", "level", "detail", "gambaran besar", "helicopter view", "formulasi", "goal"],
        "cognitive": "4TGR",
        "session_id": "eval-211",
        "query_type": "gap",
        "context_note": "Mahasiswa bingung tentang level abstraksi dalam CT — konsep multilevel",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK D — PATTERN RECOGNITION (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 212
    {
        "query": (
            "Soal CT: diberikan barisan 1, 1, 2, 3, 5, 8, 13, 21. "
            "Dosen minta identifikasi pola dan buat rumus rekursif serta "
            "tentukan suku ke-10 tanpa menghitung satu per satu. "
            "Saya tahu ini Fibonacci tapi tidak tahu cara formalkan rumus rekursifnya."
        ),
        "relevant_keywords": ["Fibonacci", "pola bilangan", "rekursi", "deret", "rumus", "pattern recognition"],
        "cognitive": "2PAI",
        "session_id": "eval-212",
        "query_type": "application",
        "context_note": "Identifikasi pola Fibonacci dan formalisasi rumus rekursif — angle GT_SUBTOPIK_03",
    },

    # 213
    {
        "query": (
            "Di soal pattern recognition saya diminta cari pola dari data: "
            "hari Senin 120 pengunjung, Selasa 85, Rabu 90, Kamis 88, Jumat 150, "
            "Sabtu 200, Minggu 180. "
            "Saya lihat ada lonjakan di akhir pekan tapi bingung bagaimana "
            "cara menyatakannya sebagai 'pola' secara formal dalam CT."
        ),
        "relevant_keywords": ["pattern recognition", "pola", "data", "tren", "temporal", "abstraksi", "validasi"],
        "cognitive": "3PGR",
        "session_id": "eval-213",
        "query_type": "application",
        "context_note": "Pattern recognition pada data pengunjung — formalisasi pola temporal",
    },

    # 214
    {
        "query": (
            "Di soal modulo saya diminta: sebuah lampu lalu lintas punya siklus "
            "merah 30 detik, kuning 5 detik, hijau 25 detik. "
            "Jika saat ini detik ke-0 adalah awal merah, "
            "pada detik ke-247 lampu berwarna apa? "
            "Saya tahu harus pakai modulo tapi tidak tahu cara hitungnya."
        ),
        "relevant_keywords": ["modulo", "siklus", "lampu lalu lintas", "periodik", "pola", "mod"],
        "cognitive": "3TAR",
        "session_id": "eval-214",
        "query_type": "application",
        "context_note": "Aplikasi modulo untuk siklus lampu lalu lintas — variasi dari GT_SUBTOPIK_04",
    },

    # 215
    {
        "query": (
            "Saya dapat soal: digit terakhir dari 7 pangkat 2025. "
            "Dosen bilang gunakan pattern recognition pada siklus digit. "
            "Saya coba: 7^1=7, 7^2=49, 7^3=343, 7^4=2401. "
            "Saya lihat digit terakhirnya: 7, 9, 3, 1. Lalu apa? "
            "Bagaimana saya tahu digit terakhir 7^2025?"
        ),
        "relevant_keywords": ["modulo", "digit terakhir", "siklus", "perpangkatan", "pola", "7^n"],
        "cognitive": "4PGR",
        "session_id": "eval-215",
        "query_type": "application",
        "context_note": "Digit terakhir 7^n menggunakan siklus modulo — variasi dari GT_SUBTOPIK_04",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK E — PSEUDOCODE & NOTASI (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 216
    {
        "query": (
            "Di soal pseudocode saya diminta tulis algoritme untuk "
            "mengecek apakah sebuah bilangan adalah bilangan sempurna. "
            "Bilangan sempurna = jumlah pembaginya (selain dirinya) = dirinya. "
            "Contoh: 6 = 1+2+3. "
            "Saya tidak tahu cara mulai — apa langkah pertama yang benar?"
        ),
        "relevant_keywords": ["pseudocode", "bilangan sempurna", "for", "modulo", "algoritme", "faktor"],
        "cognitive": "3PAR",
        "session_id": "eval-216",
        "query_type": "application",
        "context_note": "Pseudocode bilangan sempurna — kombinasi loop dan modulo",
    },

    # 217
    {
        "query": (
            "Saya latihan pseudocode untuk konversi suhu Celsius ke Fahrenheit "
            "dan Kelvin sekaligus. Rumusnya: F = C × 9/5 + 32, K = C + 273.15. "
            "Dosen minta menggunakan fungsi terpisah untuk setiap konversi "
            "dan program utama yang memanggil keduanya. "
            "Bagaimana pseudocode yang benar?"
        ),
        "relevant_keywords": ["pseudocode", "fungsi", "konversi", "parameter", "return", "program utama"],
        "cognitive": "2TGI",
        "session_id": "eval-217",
        "query_type": "application",
        "context_note": "Pseudocode fungsi konversi suhu — variasi fungsi dari GT_SUBTOPIK_10",
    },

    # 218
    {
        "query": (
            "Saya diminta tulis pseudocode untuk menghitung nilai rata-rata, "
            "nilai terbesar, dan nilai terkecil dari array yang diinput pengguna "
            "secara dinamis (pengguna bisa input berapa saja elemennya, "
            "berhenti kalau input -1). "
            "Saya bingung kapan berhenti dan bagaimana struktur loop-nya."
        ),
        "relevant_keywords": ["pseudocode", "while", "sentinel", "rata-rata", "maksimum", "minimum", "array"],
        "cognitive": "3TGR",
        "session_id": "eval-218",
        "query_type": "application",
        "context_note": "Pseudocode dengan sentinel -1 untuk statistik array dinamis",
    },

    # 219
    {
        "query": (
            "Saya dapat soal: buat pseudocode untuk menampilkan semua bilangan "
            "prima antara 1 sampai N. Bukan hanya cek satu bilangan, "
            "tapi semua prima dalam range. "
            "Saya sudah bisa cek satu prima, tapi bagaimana mengintegrasikannya "
            "ke dalam loop yang iterasi 1 sampai N?"
        ),
        "relevant_keywords": ["pseudocode", "bilangan prima", "for", "nested", "fungsi", "O(n*sqrt(n))"],
        "cognitive": "4PAI",
        "session_id": "eval-219",
        "query_type": "application",
        "context_note": "Pseudocode cetak semua prima dalam range — kombinasi loop dan fungsi cek prima",
    },

    # 220
    {
        "query": (
            "Dosen minta tulis pseudocode yang membaca string dan "
            "menghitung frekuensi setiap huruf yang muncul. "
            "Contoh: 'hello' → h=1, e=1, l=2, o=1. "
            "Saya bingung struktur data apa yang dipakai karena "
            "kita belum belajar dictionary di CT."
        ),
        "relevant_keywords": ["pseudocode", "string", "frekuensi", "array", "loop", "karakter"],
        "cognitive": "5TGI",
        "session_id": "eval-220",
        "query_type": "out_of_scope",
        "context_note": "Hitung frekuensi karakter — memerlukan struktur seperti dictionary yang tidak ada di GT CT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK F — OPERATOR & EKSPRESI (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 221
    {
        "query": (
            "Di soal evaluasi ekspresi saya dapat: 2 ** 3 ** 2. "
            "Teman saya hitung (2**3)**2 = 8**2 = 64. "
            "Saya hitung 2**(3**2) = 2**9 = 512. "
            "Kata dosen salah satu benar karena perpangkatan bersifat right-associative. "
            "Mana yang benar dan apa artinya right-associative?"
        ),
        "relevant_keywords": ["prioritas operator", "perpangkatan", "right-associative", "ekspresi", "evaluasi", "kurung"],
        "cognitive": "4TGR",
        "session_id": "eval-221",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung tentang right-associativity pada perpangkatan",
    },

    # 222
    {
        "query": (
            "Saya diminta sederhanakan ekspresi boolean: "
            "(A AND B) OR (A AND NOT B). "
            "Teman saya langsung bilang jawabannya A tapi tidak bisa jelaskan kenapa. "
            "Saya mau buktikan dengan tabel kebenaran dan aljabar boolean. "
            "Langkah apa yang harus saya lakukan?"
        ),
        "relevant_keywords": ["boolean", "AND", "OR", "NOT", "tabel kebenaran", "sederhanakan", "aljabar"],
        "cognitive": "4PAR",
        "session_id": "eval-222",
        "query_type": "application",
        "context_note": "Penyederhanaan ekspresi boolean dengan tabel kebenaran",
    },

    # 223
    {
        "query": (
            "Saya bingung dengan ekspresi: x = 5 > 3 AND 10 < 20. "
            "Saya tidak tahu nilai x berakhir integer, boolean, atau error. "
            "Di Python ini menghasilkan True. Tapi bagaimana di pseudocode CT — "
            "apakah operator perbandingan menghasilkan boolean?"
        ),
        "relevant_keywords": ["operator", "boolean", "perbandingan", "tipe data", "ekspresi", "pseudocode"],
        "cognitive": "2TAI",
        "session_id": "eval-223",
        "query_type": "gap",
        "context_note": "Mahasiswa tidak tahu tipe hasil operator perbandingan di pseudocode CT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK G — PERCABANGAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 224
    {
        "query": (
            "Soal trace: diberikan pseudocode berikut:\n"
            "  read(tahun)\n"
            "  if (tahun mod 4 == 0 AND tahun mod 100 != 0) OR (tahun mod 400 == 0)\n"
            "    then print('kabisat')\n"
            "  else print('bukan')\n"
            "Input: tahun=1900. Saya dapat 'kabisat' tapi teman dapat 'bukan'. "
            "Tolong trace dan tentukan mana yang benar."
        ),
        "relevant_keywords": ["trace", "percabangan", "modulo", "tahun kabisat", "AND", "OR", "kondisi"],
        "cognitive": "3TGI",
        "session_id": "eval-224",
        "query_type": "application",
        "context_note": "Trace percabangan tahun kabisat dengan kondisi kompleks AND/OR",
    },

    # 225
    {
        "query": (
            "Saya diminta tulis pseudocode untuk kalkulator sederhana: "
            "baca dua bilangan dan satu operator (+, -, *, /), "
            "lalu hitung hasilnya. "
            "Saya sudah bisa tulis tapi bingung cara handle pembagian dengan nol. "
            "Apakah saya perlu tambah percabangan khusus untuk itu?"
        ),
        "relevant_keywords": ["percabangan", "if", "kalkulator", "pembagian", "validasi", "pseudocode", "nested"],
        "cognitive": "3PAI",
        "session_id": "eval-225",
        "query_type": "application",
        "context_note": "Pseudocode kalkulator dengan validasi pembagian nol",
    },

    # 226
    {
        "query": (
            "Saya dapat soal trace percabangan yang kompleks:\n"
            "  read(a, b, c)\n"
            "  if (a > b) then\n"
            "    if (a > c) then print(a)\n"
            "    else print(c)\n"
            "  else\n"
            "    if (b > c) then print(b)\n"
            "    else print(c)\n"
            "Input: a=7, b=12, c=9. Apa output-nya? Saya dapat 12 tapi tidak yakin."
        ),
        "relevant_keywords": ["trace", "nested if", "percabangan", "maksimum", "kondisi", "output"],
        "cognitive": "2PGI",
        "session_id": "eval-226",
        "query_type": "application",
        "context_note": "Trace nested if untuk mencari nilai terbesar dari tiga bilangan",
    },

    # 227
    {
        "query": (
            "Dosen bilang ada perbedaan antara 'short-circuit evaluation' "
            "dalam ekspresi AND dan OR. Katanya kalau kondisi pertama False dalam AND, "
            "kondisi kedua tidak dievaluasi sama sekali. "
            "Apakah pseudocode CT juga berlaku aturan ini, "
            "atau semua kondisi selalu dievaluasi?"
        ),
        "relevant_keywords": ["AND", "OR", "kondisi", "short-circuit", "evaluasi", "percabangan", "boolean"],
        "cognitive": "5TAI",
        "session_id": "eval-227",
        "query_type": "out_of_scope",
        "context_note": "Short-circuit evaluation tidak eksplisit dibahas di GT CT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK H — PERULANGAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 228
    {
        "query": (
            "Soal CT: tulis pseudocode untuk menghitung nilai faktorial n "
            "menggunakan while loop (bukan rekursi). "
            "Saya bisa buat dengan for, tapi dosen minta versi while. "
            "Apa perbedaan struktur antara for dan while untuk kasus ini?"
        ),
        "relevant_keywords": ["while", "for", "faktorial", "perulangan", "pseudocode", "iterasi"],
        "cognitive": "2TGR",
        "session_id": "eval-228",
        "query_type": "application",
        "context_note": "Faktorial iteratif dengan while — perbandingan struktur for vs while",
    },

    # 229
    {
        "query": (
            "Saya latihan trace while loop:\n"
            "  x = 1\n  jumlah = 0\n"
            "  while (x <= 5) do\n"
            "    jumlah = jumlah + x * x\n"
            "    x = x + 1\n"
            "  print(jumlah)\n"
            "Saya hitung hasilnya 55 tapi teman dapat 15. Mana yang benar?"
        ),
        "relevant_keywords": ["trace", "while", "akumulasi", "kuadrat", "loop", "variabel"],
        "cognitive": "2PAR",
        "session_id": "eval-229",
        "query_type": "application",
        "context_note": "Trace while loop hitung jumlah kuadrat — perlu trace manual",
    },

    # 230
    {
        "query": (
            "Di soal perulangan saya diminta hitung berapa banyak bilangan "
            "antara 1 sampai 1000 yang habis dibagi 3 TAPI tidak habis dibagi 9. "
            "Saya tidak tahu cara tulis kondisi gabungan ini dalam loop. "
            "Bagaimana pseudocode-nya?"
        ),
        "relevant_keywords": ["for", "modulo", "kondisi", "perulangan", "AND", "NOT", "counter"],
        "cognitive": "3TAR",
        "session_id": "eval-230",
        "query_type": "application",
        "context_note": "Loop dengan kondisi modulo gabungan untuk menghitung bilangan dengan syarat tertentu",
    },

    # 231
    {
        "query": (
            "Saya buat pseudocode untuk cetak segitiga bintang "
            "tapi kali ini saya mau buat yang berbentuk berlian (diamond): "
            "baris pertama 1 bintang, semakin banyak ke tengah, "
            "lalu semakin sedikit ke bawah. Untuk tinggi 4, outputnya harus:\n"
            "*\n**\n***\n****\n***\n**\n*\n"
            "Berapa loop yang saya butuhkan dan bagaimana strukturnya?"
        ),
        "relevant_keywords": ["nested loop", "pola", "segitiga", "berlian", "for", "baris", "bintang"],
        "cognitive": "4PGI",
        "session_id": "eval-231",
        "query_type": "application",
        "context_note": "Pola berlian dengan nested loop — variasi pola dari GT_SUBTOPIK_09",
    },

    # 232
    {
        "query": (
            "Soal: berapa total iterasi yang dieksekusi dari pseudocode berikut?\n"
            "  for i = 1 to n do\n"
            "    for j = i to n do\n"
            "      print(i, j)\n"
            "Untuk n=4. Saya hitung 10 tapi teman hitung 16. "
            "Mana yang benar dan bagaimana cara menghitungnya sistematis?"
        ),
        "relevant_keywords": ["nested loop", "iterasi", "total", "n*(n+1)/2", "kompleksitas", "O(n^2)"],
        "cognitive": "4TGI",
        "session_id": "eval-232",
        "query_type": "application",
        "context_note": "Total iterasi nested loop dengan j mulai dari i — triangular number",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK I — FUNGSI LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 233
    {
        "query": (
            "Saya diminta buat fungsi pseudocode yang menerima array nilai mahasiswa "
            "dan mengembalikan: rata-rata, nilai tertinggi, dan nilai terendah "
            "sekaligus. Tapi fungsi kan hanya bisa return satu nilai. "
            "Bagaimana cara mengembalikan tiga nilai sekaligus dari satu fungsi?"
        ),
        "relevant_keywords": ["fungsi", "return", "multiple return", "array", "tuple", "pseudocode", "prosedur"],
        "cognitive": "4TAI",
        "session_id": "eval-233",
        "query_type": "gap",
        "context_note": "Multiple return values dari fungsi — konsep yang tidak eksplisit di GT",
    },

    # 234
    {
        "query": (
            "Saya latihan buat fungsi rekursif untuk KPK (Kelipatan Persekutuan Terkecil). "
            "Saya tahu KPK(a,b) = (a×b) / FPB(a,b). "
            "Jadi saya butuh fungsi FPB dulu. "
            "Bagaimana cara memanggil satu fungsi dari dalam fungsi lain di pseudocode CT?"
        ),
        "relevant_keywords": ["fungsi", "FPB", "KPK", "rekursi", "pemanggilan", "return", "Euclidean"],
        "cognitive": "3PGR",
        "session_id": "eval-234",
        "query_type": "application",
        "context_note": "Fungsi memanggil fungsi lain — KPK menggunakan FPB Euclidean",
    },

    # 235
    {
        "query": (
            "Di tugas saya ada soal: buat prosedur 'tukar' untuk menukar nilai "
            "dua variabel a dan b. Saya tulis:\n"
            "  procedure tukar(a, b)\n    temp = a\n    a = b\n    b = temp\n"
            "Tapi setelah dipanggil, nilai a dan b di program utama tidak berubah. "
            "Kenapa prosedur tukar saya tidak bekerja?"
        ),
        "relevant_keywords": ["prosedur", "tukar", "pass by value", "parameter", "fungsi", "scope"],
        "cognitive": "3TGR",
        "session_id": "eval-235",
        "query_type": "confusion",
        "context_note": "Prosedur tukar gagal karena pass by value — variasi dari topik scope di batch 1",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK J — REKURSI LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 236
    {
        "query": (
            "Soal rekursi: hitung jumlah semua bilangan bulat dari 1 sampai n. "
            "Saya diminta buat versi rekursif DAN versi iteratif, "
            "lalu bandingkan berapa pemanggilan fungsi yang terjadi untuk n=5. "
            "Rekursif berapa kali? Iteratif berapa iterasi?"
        ),
        "relevant_keywords": ["rekursi", "iterasi", "jumlah", "pemanggilan", "base case", "for", "n=5"],
        "cognitive": "3PAI",
        "session_id": "eval-236",
        "query_type": "application",
        "context_note": "Bandingkan rekursi vs iterasi untuk sum 1 to n secara eksplisit",
    },

    # 237
    {
        "query": (
            "Saya punya soal rekursi: fungsi myFunc(n) = myFunc(n-1) + myFunc(n-2) + 1, "
            "dengan myFunc(0) = 0 dan myFunc(1) = 1. "
            "Diminta hitung myFunc(5). "
            "Ini mirip Fibonacci tapi ada +1 di setiap langkah. "
            "Bisakah saya trace ini seperti Fibonacci biasa?"
        ),
        "relevant_keywords": ["rekursi", "trace", "Fibonacci", "base case", "+1", "call stack", "hitung"],
        "cognitive": "4TGR",
        "session_id": "eval-237",
        "query_type": "application",
        "context_note": "Trace rekursi Fibonacci-variant dengan konstanta tambahan",
    },

    # 238
    {
        "query": (
            "Saya baca bahwa ada teknik 'tail recursion' yang lebih efisien dari "
            "rekursi biasa karena tidak menumpuk call stack. "
            "Apakah pseudocode CT mengajarkan tail recursion? "
            "Kalau tidak, apa perbedaannya dengan rekursi biasa?"
        ),
        "relevant_keywords": ["rekursi", "tail recursion", "call stack", "stack overflow", "optimasi", "base case"],
        "cognitive": "6TGR",
        "session_id": "eval-238",
        "query_type": "out_of_scope",
        "context_note": "Tail recursion tidak ada di GT CT — hanya rekursi biasa yang diajarkan",
    },

    # 239
    {
        "query": (
            "Soal rekursi: menara Hanoi dengan 3 cakram. "
            "Dosen minta hitung berapa minimum perpindahan yang diperlukan. "
            "Saya tahu rumusnya 2^n - 1 tapi tidak bisa buktikan dari rekursi. "
            "Bagaimana pseudocode rekursif untuk menara Hanoi dan "
            "bagaimana buktikan rumus minimumnya?"
        ),
        "relevant_keywords": ["rekursi", "Hanoi", "2^n", "call stack", "base case", "kompleksitas", "trace"],
        "cognitive": "5PGR",
        "session_id": "eval-239",
        "query_type": "application",
        "context_note": "Menara Hanoi rekursif dan pembuktian formula minimum langkah",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK K — STRUKTUR DATA LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 240
    {
        "query": (
            "Soal: saya punya list [5, 2, 8, 1, 9, 3] dan diminta lakukan "
            "operasi berikut satu per satu: append(7), insert di indeks 2 nilai 4, "
            "hapus elemen pertama, lalu tampilkan elemen terakhir. "
            "Tunjukkan isi list setelah setiap operasi."
        ),
        "relevant_keywords": ["list", "append", "insert", "hapus", "indeks", "operasi", "array"],
        "cognitive": "2PAR",
        "session_id": "eval-240",
        "query_type": "application",
        "context_note": "Simulasi operasi list append/insert/delete step by step",
    },

    # 241
    {
        "query": (
            "Dosen menjelaskan bahwa list di CT bisa dipakai sebagai "
            "implementasi stack dengan append dan pop, "
            "atau sebagai queue dengan append dan pop(0). "
            "Saya bingung — kalau list bisa jadi stack dan queue, "
            "mengapa kita perlu belajar stack dan queue sebagai konsep terpisah?"
        ),
        "relevant_keywords": ["list", "stack", "queue", "ADT", "implementasi", "LIFO", "FIFO"],
        "cognitive": "4TAR",
        "session_id": "eval-241",
        "query_type": "confusion",
        "context_note": "Pertanyaan tentang relevansi ADT stack/queue vs implementasi list — angle berbeda dari batch 1",
    },

    # 242
    {
        "query": (
            "Saya diminta implementasikan operasi 'rotate' pada list: "
            "geser semua elemen k posisi ke kiri. "
            "Contoh: [1,2,3,4,5] rotasi 2 ke kiri → [3,4,5,1,2]. "
            "Bagaimana pseudocode untuk operasi ini tanpa menggunakan "
            "fungsi built-in slice atau rotate?"
        ),
        "relevant_keywords": ["list", "rotate", "array", "pseudocode", "loop", "indeks", "modulo"],
        "cognitive": "4PGI",
        "session_id": "eval-242",
        "query_type": "application",
        "context_note": "Rotasi list dengan pseudocode manual — operasi array yang tidak trivial",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK L — STACK & QUEUE LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 243
    {
        "query": (
            "Saya punya soal: gunakan DUA stack untuk mensimulasikan queue. "
            "Operasi: enqueue(1), enqueue(2), enqueue(3), dequeue, enqueue(4), dequeue. "
            "Dosen bilang ini bisa dilakukan dengan dua stack. "
            "Saya tidak paham idenya — bagaimana dua stack bisa jadi queue?"
        ),
        "relevant_keywords": ["stack", "queue", "LIFO", "FIFO", "dua stack", "simulasi", "PUSH", "POP"],
        "cognitive": "5TAI",
        "session_id": "eval-243",
        "query_type": "application",
        "context_note": "Implementasi queue menggunakan dua stack — masalah klasik struktur data",
    },

    # 244
    {
        "query": (
            "Soal simulasi stack: saya punya ekspresi postfix '3 4 + 2 * 7 -'. "
            "Dosen minta evaluasi menggunakan stack. "
            "Saya paham stack tapi tidak paham cara evaluasi postfix. "
            "Apa itu postfix dan bagaimana stack membantu mengevaluasinya?"
        ),
        "relevant_keywords": ["stack", "postfix", "ekspresi", "PUSH", "POP", "evaluasi", "operator"],
        "cognitive": "5TGR",
        "session_id": "eval-244",
        "query_type": "application",
        "context_note": "Evaluasi ekspresi postfix menggunakan stack",
    },

    # 245
    {
        "query": (
            "Saya diminta buat simulasi antrian pelanggan bank dengan queue. "
            "Ada 3 loket aktif. Setiap loket melayani satu pelanggan. "
            "Ketika loket kosong, pelanggan berikutnya dari queue dipanggil. "
            "Apakah ini cukup dengan satu queue atau butuh struktur lebih kompleks?"
        ),
        "relevant_keywords": ["queue", "FIFO", "antrian", "multi-server", "dequeue", "enqueue", "struktur data"],
        "cognitive": "4PAR",
        "session_id": "eval-245",
        "query_type": "scenario",
        "context_note": "Desain sistem antrian multi-loket dengan queue",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK M — GRAPH & TREE LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 246
    {
        "query": (
            "Diberikan tree berikut:\n"
            "       A\n      / \\\n     B   C\n    / \\   \\\n   D   E   F\n"
            "Dosen minta lakukan in-order traversal dan level-order traversal. "
            "Saya tahu pre-order tapi bingung bedanya in-order, post-order, dan level-order."
        ),
        "relevant_keywords": ["tree", "traversal", "in-order", "level-order", "BFS", "DFS", "pre-order"],
        "cognitive": "3TGI",
        "session_id": "eval-246",
        "query_type": "application",
        "context_note": "Traversal tree: in-order dan level-order traversal",
    },

    # 247
    {
        "query": (
            "Di soal graph saya diminta tentukan apakah graph berikut memiliki cycle: "
            "A→B, B→C, C→D, D→B. "
            "Saya tidak tahu cara sistematis untuk deteksi cycle. "
            "Apakah DFS bisa digunakan untuk deteksi cycle dan bagaimana caranya?"
        ),
        "relevant_keywords": ["graph", "cycle", "DFS", "deteksi", "directed", "traversal", "stack"],
        "cognitive": "4TGR",
        "session_id": "eval-247",
        "query_type": "application",
        "context_note": "Deteksi cycle di directed graph menggunakan DFS",
    },

    # 248
    {
        "query": (
            "Saya diminta tentukan apakah dua graph berikut merupakan "
            "'connected graph' (semua node terhubung) atau tidak:\n"
            "Graph 1: A-B, B-C, C-D, D-A\n"
            "Graph 2: A-B, C-D, E-F\n"
            "Bagaimana cara sistematis mengecek konektivitas graph?"
        ),
        "relevant_keywords": ["graph", "connected", "BFS", "DFS", "traversal", "terhubung", "komponen"],
        "cognitive": "3PAR",
        "session_id": "eval-248",
        "query_type": "application",
        "context_note": "Cek konektivitas graph menggunakan BFS/DFS",
    },

    # 249
    {
        "query": (
            "Saya belajar tentang weighted graph dan ingin tahu: "
            "kalau saya pakai BFS pada weighted graph untuk cari jalur terpendek, "
            "apakah hasilnya selalu benar? "
            "Atau BFS tidak bisa dipakai untuk weighted graph?"
        ),
        "relevant_keywords": ["BFS", "weighted graph", "jarak terpendek", "Dijkstra", "bobot", "traversal"],
        "cognitive": "5PAI",
        "session_id": "eval-249",
        "query_type": "confusion",
        "context_note": "Keterbatasan BFS untuk weighted graph vs Dijkstra",
    },

    # 250
    {
        "query": (
            "Di soal graph saya diminta representasikan graph kota-kota Indonesia "
            "beserta jarak jalannya menggunakan adjacency matrix. "
            "Ada 6 kota: Jakarta, Bandung, Surabaya, Yogyakarta, Semarang, Malang. "
            "Bagaimana format adjacency matrix dan bagaimana cara membacanya?"
        ),
        "relevant_keywords": ["adjacency matrix", "graph", "weighted", "representasi", "jarak", "matriks"],
        "cognitive": "2TGR",
        "session_id": "eval-250",
        "query_type": "application",
        "context_note": "Membuat dan membaca adjacency matrix weighted graph",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK N — BIG-O LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 251
    {
        "query": (
            "Saya punya dua algoritme untuk masalah yang sama:\n"
            "Algoritme A: O(n log n) waktu, O(n) space\n"
            "Algoritme B: O(n²) waktu, O(1) space\n"
            "Untuk n=100 dan memori terbatas, mana yang lebih baik? "
            "Bagaimana cara mempertimbangkan trade-off time vs space complexity?"
        ),
        "relevant_keywords": ["Big-O", "space complexity", "time complexity", "trade-off", "O(n log n)", "O(n^2)"],
        "cognitive": "5PAR",
        "session_id": "eval-251",
        "query_type": "scenario",
        "context_note": "Trade-off time complexity vs space complexity dalam pemilihan algoritme",
    },

    # 252
    {
        "query": (
            "Tentukan Big-O dari fungsi rekursif berikut:\n"
            "  function f(n):\n"
            "    if n <= 1: return 1\n"
            "    return f(n/2) + f(n/2) + n\n"
            "Saya tidak bisa langsung lihat Big-O-nya seperti untuk loop biasa. "
            "Ada cara sistematis untuk analisis rekursi?"
        ),
        "relevant_keywords": ["Big-O", "rekursi", "T(n)", "master theorem", "O(n log n)", "analisis"],
        "cognitive": "6PAI",
        "session_id": "eval-252",
        "query_type": "application",
        "context_note": "Analisis Big-O fungsi rekursif — pendekatan master theorem",
    },

    # 253
    {
        "query": (
            "Dosen bilang best case, average case, dan worst case complexity "
            "bisa berbeda untuk algoritme yang sama. "
            "Untuk bubble sort: saya tahu worst case O(n²). "
            "Tapi berapa best case-nya dan kapan best case itu terjadi?"
        ),
        "relevant_keywords": ["bubble sort", "best case", "worst case", "O(n)", "O(n^2)", "kompleksitas", "early termination"],
        "cognitive": "4TAI",
        "session_id": "eval-253",
        "query_type": "gap",
        "context_note": "Perbedaan best/worst/average case complexity untuk bubble sort",
    },

    # 254
    {
        "query": (
            "Saya bingung dengan notasi big-Omega (Ω) dan big-Theta (Θ). "
            "Di kuliah CT kita hanya belajar Big-O. "
            "Apa perbedaan O, Ω, dan Θ, dan apakah semuanya diperlukan untuk "
            "evaluasi algoritme di CT?"
        ),
        "relevant_keywords": ["Big-O", "Omega", "Theta", "kompleksitas", "upper bound", "lower bound", "analisis"],
        "cognitive": "6TGI",
        "session_id": "eval-254",
        "query_type": "out_of_scope",
        "context_note": "Ω dan Θ notation tidak ada di GT CT — hanya Big-O yang diajarkan",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK O — SEARCHING & SORTING LANJUTAN (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 255
    {
        "query": (
            "Soal: lakukan binary search pada array [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] "
            "untuk mencari nilai 23. "
            "Tunjukkan nilai low, high, dan mid di setiap langkah "
            "dan berapa total langkah yang dibutuhkan."
        ),
        "relevant_keywords": ["binary search", "trace", "low", "high", "mid", "langkah", "array terurut"],
        "cognitive": "2TGI",
        "session_id": "eval-255",
        "query_type": "application",
        "context_note": "Trace binary search mencari 23 — array berbeda dari batch 1",
    },

    # 256
    {
        "query": (
            "Soal: lakukan insertion sort pada array [8, 3, 6, 1, 9, 2]. "
            "Saya tahu bubble dan selection sort tapi belum pernah latihan insertion sort. "
            "Bagaimana cara kerja insertion sort dan tunjukkan setiap langkahnya."
        ),
        "relevant_keywords": ["insertion sort", "sorting", "array", "trace", "shift", "O(n^2)", "perbandingan"],
        "cognitive": "3TAR",
        "session_id": "eval-256",
        "query_type": "application",
        "context_note": "Trace insertion sort — algoritme sorting yang belum ada di batch 1",
    },

    # 257
    {
        "query": (
            "Di ujian ada soal: 'jelaskan cara kerja merge sort dan trace untuk "
            "array [6, 3, 8, 2, 9, 1, 5, 4]'. "
            "Saya paham prinsipnya (bagi dua, sort masing-masing, merge) "
            "tapi bingung bagaimana menampilkan proses merge yang benar."
        ),
        "relevant_keywords": ["merge sort", "trace", "divide", "merge", "O(n log n)", "array", "rekursi"],
        "cognitive": "3PGR",
        "session_id": "eval-257",
        "query_type": "application",
        "context_note": "Trace merge sort lengkap termasuk proses merge step by step",
    },

    # 258
    {
        "query": (
            "Saya diminta memilih algoritme sorting terbaik untuk kasus berikut: "
            "data 10.000 elemen yang sudah terurut TERBALIK (descending), "
            "ingin diurutkan ascending. "
            "Dari bubble sort, selection sort, insertion sort, merge sort — "
            "mana yang paling efisien dan mengapa?"
        ),
        "relevant_keywords": ["sorting", "reverse sorted", "bubble sort", "insertion sort", "merge sort", "O(n^2)", "O(n log n)"],
        "cognitive": "5TGR",
        "session_id": "eval-258",
        "query_type": "scenario",
        "context_note": "Pilih sorting terbaik untuk data reverse-sorted — angle berbeda dari batch 1",
    },

    # 259
    {
        "query": (
            "Saya punya array yang sangat besar (10 juta elemen) dan "
            "perlu cari suatu elemen. Data sudah terurut. "
            "Saya tahu binary search O(log n) lebih cepat. "
            "Tapi seberapa cepat tepatnya? "
            "Berapa langkah binary search untuk 10 juta elemen?"
        ),
        "relevant_keywords": ["binary search", "O(log n)", "langkah", "10 juta", "log2", "efisiensi", "linear search"],
        "cognitive": "4PAI",
        "session_id": "eval-259",
        "query_type": "application",
        "context_note": "Menghitung langkah aktual binary search untuk n sangat besar",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK P — INTEGRATIF LINTAS TOPIK (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 260
    {
        "query": (
            "Saya diminta buat program untuk cek apakah sebuah angka adalah "
            "'bilangan Armstrong' (jumlah pangkat digit = angkanya sendiri). "
            "Contoh: 153 = 1³ + 5³ + 3³. "
            "Bagaimana pseudocode yang benar dan struktur apa saja yang dibutuhkan?"
        ),
        "relevant_keywords": ["pseudocode", "while", "modulo", "div", "digit", "pangkat", "loop"],
        "cognitive": "4TGI",
        "session_id": "eval-260",
        "query_type": "application",
        "context_note": "Pseudocode bilangan Armstrong — kombinasi while, modulo, dan perpangkatan",
    },

    # 261
    {
        "query": (
            "Saya punya dua array terurut: A=[1,3,5,7,9] dan B=[2,4,6,8,10]. "
            "Saya diminta gabungkan keduanya menjadi satu array terurut "
            "tanpa menggunakan sorting — langsung merge. "
            "Bagaimana pseudocode untuk merge dua sorted array?"
        ),
        "relevant_keywords": ["array", "merge", "terurut", "pseudocode", "while", "dua pointer", "indeks"],
        "cognitive": "4PGR",
        "session_id": "eval-261",
        "query_type": "application",
        "context_note": "Merge dua sorted array tanpa sorting ulang — bagian dari merge sort",
    },

    # 262
    {
        "query": (
            "Soal CT kompleks: saya punya list nilai 100 mahasiswa. "
            "Saya perlu: (1) urutkan secara descending, (2) cari median, "
            "(3) hitung berapa persen yang di atas rata-rata. "
            "Dosen minta pakai dekomposisi — pecah jadi sub-fungsi. "
            "Bagaimana rancangan fungsi-fungsinya?"
        ),
        "relevant_keywords": ["dekomposisi", "fungsi", "sorting", "median", "rata-rata", "modular", "array"],
        "cognitive": "5PAR",
        "session_id": "eval-262",
        "query_type": "scenario",
        "context_note": "Soal integratif dekomposisi + sorting + statistik",
    },

    # 263
    {
        "query": (
            "Di tugas saya harus buat sistem sederhana untuk "
            "riwayat browser: bisa push URL baru, back (kembali), dan forward (maju). "
            "Dosen bilang gunakan struktur data yang tepat. "
            "Apakah ini pakai satu stack, dua stack, atau queue?"
        ),
        "relevant_keywords": ["stack", "queue", "riwayat", "browser", "LIFO", "dua stack", "back forward"],
        "cognitive": "4TAI",
        "session_id": "eval-263",
        "query_type": "scenario",
        "context_note": "Desain riwayat browser menggunakan stack — aplikasi nyata struktur data",
    },

    # 264
    {
        "query": (
            "Saya mendapat soal: diberikan graph kota dengan jalan searah. "
            "Tentukan apakah setiap kota bisa dicapai dari kota A. "
            "Ini beda dari soal jarak terpendek — ini tentang keterjangkauan. "
            "Algoritme apa yang tepat dan bagaimana implementasinya?"
        ),
        "relevant_keywords": ["BFS", "DFS", "reachability", "graph", "directed", "traversal", "visited"],
        "cognitive": "4PGI",
        "session_id": "eval-264",
        "query_type": "application",
        "context_note": "Graph reachability dengan BFS/DFS — berbeda dari shortest path",
    },

    # 265
    {
        "query": (
            "Soal CT akhir semester: diberi list kata-kata dan diminta cari "
            "kata terpanjang yang merupakan palindrom. "
            "Dosen minta pakai pendekatan CT: dekomposisi, fungsi terpisah, efisiensi. "
            "Bagaimana rancangan solusinya?"
        ),
        "relevant_keywords": ["dekomposisi", "fungsi", "palindrom", "string", "loop", "pencarian", "modular"],
        "cognitive": "5TGI",
        "session_id": "eval-265",
        "query_type": "scenario",
        "context_note": "Cari palindrom terpanjang dari list kata — integratif string + loop + dekomposisi",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK Q — SOAL KONFUSI SPESIFIK (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 266
    {
        "query": (
            "Saya bingung antara 'increment' dan 'accumulate' dalam loop. "
            "Di kode saya:\n"
            "  total = total + 1  (increment counter)\n"
            "  total = total + nilai  (accumulate sum)\n"
            "Dosen bilang keduanya valid tapi berbeda tujuan. "
            "Kapan saya pakai counter dan kapan accumulator?"
        ),
        "relevant_keywords": ["loop", "counter", "akumulasi", "increment", "variabel", "for", "while"],
        "cognitive": "1PAR",
        "session_id": "eval-266",
        "query_type": "confusion",
        "context_note": "Perbedaan counter vs accumulator dalam loop — konsep fundamental yang sering tertukar",
    },

    # 267
    {
        "query": (
            "Saya punya soal trace:\n"
            "  a = 3\n  b = a\n  a = a + 1\n  print(b)\n"
            "Saya dapat 3, teman dapat 4. "
            "Dia bilang b 'mengikuti' a karena b = a. "
            "Siapa yang benar? Apakah assignment menyalin nilai atau menyalin referensi?"
        ),
        "relevant_keywords": ["assignment", "variabel", "nilai", "referensi", "trace", "pseudocode", "tipe data"],
        "cognitive": "2TGR",
        "session_id": "eval-267",
        "query_type": "confusion",
        "context_note": "Mahasiswa bingung apakah assignment menyalin nilai atau referensi",
    },

    # 268
    {
        "query": (
            "Di soal saya ada operasi: x = x + 1 ditulis sebelum loop, "
            "di dalam loop, dan setelah loop. "
            "Dosen bilang posisinya sangat penting. "
            "Saya trace tapi masih bingung kenapa posisi assignment "
            "di dalam atau di luar loop menghasilkan hasil yang berbeda."
        ),
        "relevant_keywords": ["assignment", "loop", "posisi", "trace", "variabel", "iterasi", "inisialisasi"],
        "cognitive": "2PAI",
        "session_id": "eval-268",
        "query_type": "confusion",
        "context_note": "Posisi assignment relatif terhadap loop mempengaruhi hasil — konsep penting tapi sering salah",
    },

    # 269
    {
        "query": (
            "Saya latihan trace kode ini:\n"
            "  i = 10\n"
            "  while (i > 0) do\n"
            "    print(i)\n"
            "    i = i - 3\n"
            "Berapa baris yang tercetak dan apa nilai-nilainya? "
            "Saya bingung kapan tepatnya loop berhenti."
        ),
        "relevant_keywords": ["trace", "while", "kondisi", "i > 0", "decrement", "loop", "berhenti"],
        "cognitive": "2TAI",
        "session_id": "eval-269",
        "query_type": "application",
        "context_note": "Trace while loop dengan decrement tidak uniform — kapan tepat berhenti",
    },

    # 270
    {
        "query": (
            "Di pseudocode CT, saya lihat dua cara tulis pengulangan:\n"
            "  for i = 1 to 10 do ...\n"
            "  for i = 0 to 9 do ...\n"
            "Keduanya 10 iterasi. Mana yang lebih 'benar' di CT? "
            "Dan kalau saya akses elemen array di dalam loop, "
            "mana indeks yang lebih aman?"
        ),
        "relevant_keywords": ["for", "indeks", "0-indexed", "1-indexed", "array", "loop", "konvensi"],
        "cognitive": "3TGI",
        "session_id": "eval-270",
        "query_type": "confusion",
        "context_note": "Konfusi 0-indexed vs 1-indexed dalam for loop — berbeda dari batch 1 yang fokus di list",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK R — POLA BILANGAN & DERET LANJUTAN
    # ══════════════════════════════════════════════════════════════════════

    # 271
    {
        "query": (
            "Soal: diberikan deret 2, 6, 12, 20, 30, 42. "
            "Dosen minta identifikasi pola, tuliskan rumus eksplisit suku ke-n, "
            "dan hitung suku ke-15. "
            "Saya lihat selisihnya: 4, 6, 8, 10, 12 — bertambah 2 tiap kali. "
            "Tapi bagaimana cara temukan rumus eksplisit dari pola ini?"
        ),
        "relevant_keywords": ["pola bilangan", "deret", "selisih bertingkat", "rumus eksplisit", "n(n+1)", "pattern recognition"],
        "cognitive": "4PGR",
        "session_id": "eval-271",
        "query_type": "application",
        "context_note": "Deret n(n+1) — identifikasi pola dan rumus eksplisit dari selisih bertingkat",
    },

    # 272
    {
        "query": (
            "Soal: berapa banyak jabat tangan yang terjadi jika ada 20 orang "
            "di sebuah ruangan dan setiap orang berjabat tangan dengan semua orang lain? "
            "Dosen bilang ini berkaitan dengan pola deret segitiga. "
            "Bagaimana cara menghitungnya dengan pattern recognition?"
        ),
        "relevant_keywords": ["pola bilangan", "deret segitiga", "kombinasi", "jabat tangan", "n*(n-1)/2", "pattern recognition"],
        "cognitive": "3PAR",
        "session_id": "eval-272",
        "query_type": "application",
        "context_note": "Masalah jabat tangan sebagai aplikasi deret segitiga",
    },

    # 273
    {
        "query": (
            "Di soal pattern recognition saya diminta cari rumus untuk "
            "pola kotak-kotak dalam grid: baris pertama 1 kotak, "
            "baris kedua 3 kotak, baris ketiga 6 kotak, baris keempat 10 kotak. "
            "Saya lihat ini deret segitiga tapi tidak yakin rumus ke-n-nya."
        ),
        "relevant_keywords": ["pola bilangan", "deret segitiga", "rumus", "n*(n+1)/2", "pattern recognition", "grid"],
        "cognitive": "3TGR",
        "session_id": "eval-273",
        "query_type": "application",
        "context_note": "Deret segitiga dalam konteks pola grid — variasi dari GT_SUBTOPIK_03",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK S — SOAL OUT-OF-SCOPE YANG MENARIK
    # ══════════════════════════════════════════════════════════════════════

    # 274
    {
        "query": (
            "Saya dengar ada algoritme sorting yang lebih cepat dari O(n log n) "
            "untuk kasus tertentu — namanya Counting Sort atau Radix Sort. "
            "Apakah ini diajarkan di CT? Dan bagaimana bisa lebih cepat "
            "dari batas teoritis O(n log n) untuk comparison-based sorting?"
        ),
        "relevant_keywords": ["counting sort", "radix sort", "O(n)", "sorting", "non-comparison", "kompleksitas"],
        "cognitive": "6PAR",
        "session_id": "eval-274",
        "query_type": "out_of_scope",
        "context_note": "Counting sort dan radix sort tidak ada di GT CT — lebih lanjut dari kurikulum",
    },

    # 275
    {
        "query": (
            "Saya tertarik dengan algoritme greedy dan dynamic programming "
            "yang sering muncul di lomba programming. "
            "Apakah kedua teknik ini diajarkan di mata kuliah CT IPB? "
            "Kalau tidak, apa hubungannya dengan CT yang sudah kita pelajari?"
        ),
        "relevant_keywords": ["greedy", "dynamic programming", "algoritme", "optimasi", "dekomposisi", "CT"],
        "cognitive": "5TGI",
        "session_id": "eval-275",
        "query_type": "out_of_scope",
        "context_note": "Greedy dan DP tidak ada di GT CT — konsep lanjutan di luar kurikulum",
    },

    # 276
    {
        "query": (
            "Di internet saya baca tentang 'P vs NP problem' yang katanya "
            "salah satu masalah matematika terbesar yang belum terpecahkan. "
            "Apakah ini berkaitan dengan kompleksitas Big-O yang kita pelajari? "
            "Dan apa artinya dalam praktik?"
        ),
        "relevant_keywords": ["P vs NP", "kompleksitas", "Big-O", "NP-hard", "algoritme", "O(n^2)"],
        "cognitive": "6TGR",
        "session_id": "eval-276",
        "query_type": "out_of_scope",
        "context_note": "P vs NP tidak ada di GT CT — di luar kurikulum tapi berkaitan dengan kompleksitas",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK T — SOAL KOMPARATIF (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 277
    {
        "query": (
            "Dosen minta saya bandingkan tiga cara menghitung faktorial: "
            "rekursi biasa, iterasi for loop, dan dengan rumus langsung (Stirling approximation). "
            "Dari sisi Big-O, akurasi, dan kemudahan implementasi — mana yang terbaik?"
        ),
        "relevant_keywords": ["faktorial", "rekursi", "iterasi", "for", "kompleksitas", "O(n)", "perbandingan"],
        "cognitive": "5TAR",
        "session_id": "eval-277",
        "query_type": "comparative",
        "context_note": "Tiga cara hitung faktorial dibandingkan secara komprehensif",
    },

    # 278
    {
        "query": (
            "Saya bingung memilih antara nested if dan switch/case untuk "
            "menentukan hari kerja berdasarkan angka 1-7. "
            "Di pseudocode CT kita hanya belajar if. "
            "Apakah ada konstruk seperti switch/case di pseudocode CT? "
            "Kalau tidak, bagaimana cara paling elegan menggantinya dengan if?"
        ),
        "relevant_keywords": ["switch case", "if-else", "percabangan", "else-if", "pseudocode", "hari", "kondisi"],
        "cognitive": "3TAI",
        "session_id": "eval-278",
        "query_type": "comparative",
        "context_note": "Switch/case tidak ada di pseudocode CT — bagaimana menggantinya dengan else-if",
    },

    # 279
    {
        "query": (
            "Saya mau bandingkan: untuk mencari nilai terbesar dalam array n elemen, "
            "apakah lebih efisien linear scan satu pass, "
            "atau sorting dulu lalu ambil elemen terakhir? "
            "Dari Big-O, mana yang lebih baik dan mengapa?"
        ),
        "relevant_keywords": ["linear search", "sorting", "maksimum", "O(n)", "O(n log n)", "efisiensi", "perbandingan"],
        "cognitive": "4PAR",
        "session_id": "eval-279",
        "query_type": "comparative",
        "context_note": "Bandingkan linear scan vs sorting untuk mencari nilai maksimum",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK U — SOAL APLIKASI DUNIA NYATA
    # ══════════════════════════════════════════════════════════════════════

    # 280
    {
        "query": (
            "Saya diminta rancang algoritme untuk sistem rekomendasi film sederhana. "
            "Diketahui: user bisa kasih rating 1-5 untuk film. "
            "Rekomendasikan film yang belum ditonton dengan rating tertinggi rata-rata. "
            "Dari perspektif CT, bagaimana dekomposisi dan algoritme dasarnya?"
        ),
        "relevant_keywords": ["dekomposisi", "algoritme", "array", "rata-rata", "sorting", "pencarian", "rekomendasi"],
        "cognitive": "5PAR",
        "session_id": "eval-280",
        "query_type": "scenario",
        "context_note": "Sistem rekomendasi sederhana menggunakan CT — dekomposisi + algoritme",
    },

    # 281
    {
        "query": (
            "Saya sedang bikin aplikasi to-do list dan perlu fitur: "
            "tambah tugas, hapus tugas selesai, dan tampilkan tugas berdasarkan prioritas. "
            "Dari perspektif CT, struktur data apa yang paling cocok "
            "dan bagaimana pseudocode untuk masing-masing operasi?"
        ),
        "relevant_keywords": ["struktur data", "list", "queue", "prioritas", "pseudocode", "operasi", "dekomposisi"],
        "cognitive": "3TGR",
        "session_id": "eval-281",
        "query_type": "scenario",
        "context_note": "Desain aplikasi to-do list dengan pemilihan struktur data yang tepat",
    },

    # 282
    {
        "query": (
            "Saya diminta analisis kompleksitas program cek duplikat dalam array: "
            "apakah ada elemen yang muncul lebih dari sekali. "
            "Pendekatan naif: bandingkan setiap pasangan elemen. "
            "Dosen bilang ada cara lebih cepat. "
            "Apa saja pendekatan yang bisa digunakan dan berapa Big-O masing-masing?"
        ),
        "relevant_keywords": ["duplikat", "Big-O", "O(n^2)", "array", "nested loop", "pencarian", "efisiensi"],
        "cognitive": "5TGI",
        "session_id": "eval-282",
        "query_type": "application",
        "context_note": "Analisis kompleksitas cek duplikat — O(n²) naive vs pendekatan lebih efisien",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK V — SOAL TRACE KOMPLEKS (angle baru)
    # ══════════════════════════════════════════════════════════════════════

    # 283
    {
        "query": (
            "Trace fungsi rekursif ini untuk input n=4:\n"
            "  function mystery(n):\n"
            "    if n == 0: return 0\n"
            "    if n mod 2 == 0: return mystery(n/2)\n"
            "    else: return 1 + mystery(n-1)\n"
            "Saya tidak tahu apa yang dihitung fungsi ini — tapi dosen bilang "
            "hasilnya punya makna. Apa output-nya dan apa yang dihitung fungsi ini?"
        ),
        "relevant_keywords": ["rekursi", "trace", "modulo", "bit", "count", "base case", "mystery function"],
        "cognitive": "5TGR",
        "session_id": "eval-283",
        "query_type": "application",
        "context_note": "Trace rekursi mysterious function — menghitung jumlah bit 1 dalam representasi biner",
    },

    # 284
    {
        "query": (
            "Trace program berikut untuk input array [3,1,4,1,5,9,2,6]:\n"
            "  found = false\n  i = 0\n"
            "  while (i < panjang(arr) AND NOT found) do\n"
            "    if arr[i] > arr[i+1] then found = true\n"
            "    else i = i + 1\n"
            "  print(found)\n"
            "Apa output-nya dan apa yang dicari program ini?"
        ),
        "relevant_keywords": ["trace", "while", "array", "boolean", "found", "kondisi", "adjacent"],
        "cognitive": "3PAI",
        "session_id": "eval-284",
        "query_type": "application",
        "context_note": "Trace while dengan short-circuit untuk cek apakah array sudah terurut",
    },

    # 285
    {
        "query": (
            "Saya trace kode ini:\n"
            "  x = 5\n  y = 3\n"
            "  x = x + y\n  y = x - y\n  x = x - y\n"
            "  print(x, y)\n"
            "Saya dapat x=3, y=5. Tapi teman dapat x=5, y=3. "
            "Mana yang benar? Dan apa yang sebenarnya dilakukan kode ini?"
        ),
        "relevant_keywords": ["trace", "assignment", "tukar variabel", "tanpa temp", "variabel", "aritmatika"],
        "cognitive": "3TGI",
        "session_id": "eval-285",
        "query_type": "application",
        "context_note": "Trace tukar variabel tanpa variabel temp menggunakan aritmatika",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK W — KOMBINASI TOPIK BARU
    # ══════════════════════════════════════════════════════════════════════

    # 286
    {
        "query": (
            "Saya belajar bahwa pattern recognition dan abstraksi sering "
            "digunakan bersama-sama. Dosen beri contoh: "
            "'mengidentifikasi pola desain dalam kode yang berulang'. "
            "Tapi saya tidak paham: apakah identifikasi pola dulu baru abstraksi, "
            "atau sebaliknya? Mana yang mendahului?"
        ),
        "relevant_keywords": ["pattern recognition", "abstraksi", "AADP", "CT", "urutan", "pilar", "dekomposisi"],
        "cognitive": "4TGI",
        "session_id": "eval-286",
        "query_type": "cross_topic",
        "context_note": "Urutan aplikasi pattern recognition dan abstraksi dalam pemecahan masalah",
    },

    # 287
    {
        "query": (
            "Dosen bilang kompleksitas binary search O(log n) 'berasal dari' "
            "strategi divide and conquer yang merupakan implementasi dekomposisi CT. "
            "Saya tidak melihat hubungannya secara langsung. "
            "Bagaimana dekomposisi CT menjelaskan mengapa binary search O(log n)?"
        ),
        "relevant_keywords": ["binary search", "dekomposisi", "divide and conquer", "O(log n)", "CT", "Big-O"],
        "cognitive": "5PAR",
        "session_id": "eval-287",
        "query_type": "cross_topic",
        "context_note": "Hubungan antara dekomposisi CT dan kompleksitas O(log n) binary search",
    },

    # 288
    {
        "query": (
            "Saya sedang belajar graph dan structure data sekaligus. "
            "Ternyata tree bisa ditraversal rekursif karena setiap subtree "
            "adalah tree juga. Ini membuat saya bertanya: "
            "apakah semua struktur rekursif (tree, dll.) bisa ditangani "
            "dengan fungsi rekursif secara alami?"
        ),
        "relevant_keywords": ["rekursi", "tree", "struktur data", "traversal", "DFS", "self-similar", "fungsi"],
        "cognitive": "5TGR",
        "session_id": "eval-288",
        "query_type": "cross_topic",
        "context_note": "Hubungan antara rekursi dan struktur data rekursif seperti tree",
    },

    # 289
    {
        "query": (
            "Di kelas abstraksi saya belajar bahwa kita 'menyembunyikan detail'. "
            "Di kelas fungsi saya belajar bahwa fungsi 'menyembunyikan implementasi'. "
            "Di kelas struktur data saya belajar ADT (Abstract Data Type) "
            "yang juga 'menyembunyikan implementasi'. "
            "Apakah ketiganya adalah hal yang sama?"
        ),
        "relevant_keywords": ["abstraksi", "fungsi", "ADT", "stack", "queue", "implementasi", "CT"],
        "cognitive": "5PAI",
        "session_id": "eval-289",
        "query_type": "cross_topic",
        "context_note": "Unifikasi konsep abstraksi di berbagai konteks: CT, fungsi, dan ADT",
    },

    # 290
    {
        "query": (
            "Dosen pernah bilang bahwa DFS menggunakan stack secara implisit "
            "melalui rekursi, sementara BFS menggunakan queue secara eksplisit. "
            "Saya tidak paham maksud 'implisit' di sini. "
            "Bagaimana rekursi DFS menggunakan stack tanpa kita mendeklarasikan stack?"
        ),
        "relevant_keywords": ["DFS", "rekursi", "call stack", "stack", "BFS", "queue", "implisit"],
        "cognitive": "5TGI",
        "session_id": "eval-290",
        "query_type": "cross_topic",
        "context_note": "Hubungan rekursi DFS dengan call stack yang implisit — angle berbeda dari batch 1",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK X — SOAL APLIKASI LANJUTAN
    # ══════════════════════════════════════════════════════════════════════

    # 291
    {
        "query": (
            "Soal CT: buat pseudocode untuk mengecek apakah sebuah string "
            "merupakan anagram dari string lain. "
            "Contoh: 'listen' dan 'silent' adalah anagram. "
            "Saya tidak boleh pakai sorting — harus pakai counting atau cara lain. "
            "Bagaimana algoritmanya?"
        ),
        "relevant_keywords": ["string", "anagram", "pseudocode", "loop", "array", "frekuensi", "perbandingan"],
        "cognitive": "4TAR",
        "session_id": "eval-291",
        "query_type": "application",
        "context_note": "Cek anagram dengan counting — string manipulation tanpa sorting",
    },

    # 292
    {
        "query": (
            "Saya diminta buat pseudocode untuk binary to decimal conversion. "
            "Input: string binary seperti '1011'. Output: nilai decimal (11). "
            "Saya tahu prinsipnya (nilai posisi: 2^3 + 2^1 + 2^0) "
            "tapi bingung cara implementasi pseudocode-nya dengan loop."
        ),
        "relevant_keywords": ["pseudocode", "binary", "decimal", "konversi", "for", "perpangkatan", "loop"],
        "cognitive": "3PGI",
        "session_id": "eval-292",
        "query_type": "application",
        "context_note": "Konversi binary ke decimal dengan pseudocode — operasi string dan aritmatika",
    },

    # 293
    {
        "query": (
            "Soal: buat pseudocode untuk mencari semua pasangan bilangan "
            "dalam array yang jumlahnya sama dengan nilai target K. "
            "Contoh: array [1,4,3,2,6,8] dan K=7, pasangannya: (1,6), (3,4). "
            "Pendekatan naif O(n²) sudah saya tahu — ada yang lebih efisien?"
        ),
        "relevant_keywords": ["array", "pasangan", "O(n^2)", "nested loop", "efisiensi", "target", "pencarian"],
        "cognitive": "5TAI",
        "session_id": "eval-293",
        "query_type": "application",
        "context_note": "Cari pasangan dengan sum K — O(n²) naive dan kemungkinan optimasi",
    },

    # 294
    {
        "query": (
            "Tugas CT saya: buat pseudocode untuk konversi Roman numeral ke integer. "
            "Contoh: 'XIV' = 14, 'IX' = 9. "
            "Saya tahu aturannya (kalau simbol lebih kecil ada sebelum yang lebih besar, dikurangi). "
            "Tapi saya bingung cara implementasi logika 'pengurangan' ini dalam pseudocode."
        ),
        "relevant_keywords": ["pseudocode", "string", "loop", "percabangan", "Roman numeral", "konversi", "kondisi"],
        "cognitive": "4PGR",
        "session_id": "eval-294",
        "query_type": "application",
        "context_note": "Konversi Roman numeral ke integer — kombinasi string traversal dan percabangan",
    },

    # 295
    {
        "query": (
            "Saya perlu buat fungsi untuk menghitung nilai Pi (π) menggunakan "
            "deret Leibniz: π/4 = 1 - 1/3 + 1/5 - 1/7 + ... "
            "Dosen minta pseudocode yang berhenti ketika selisih suku berikutnya "
            "sudah cukup kecil (misal < 0.0001). "
            "Bagaimana struktur while loop-nya?"
        ),
        "relevant_keywords": ["while", "deret", "konvergensi", "pseudocode", "fungsi", "epsilon", "while loop"],
        "cognitive": "5PGI",
        "session_id": "eval-295",
        "query_type": "application",
        "context_note": "Approksimasi Pi dengan deret Leibniz dan while dengan epsilon convergence",
    },

    # ══════════════════════════════════════════════════════════════════════
    # BLOK Y — SOAL GAP SPESIFIK
    # ══════════════════════════════════════════════════════════════════════

    # 296
    {
        "query": (
            "Di pseudocode CT saya selalu menulis komentar seperti ini: "
            "# ini adalah variabel counter. "
            "Dosen bilang komentar yang baik itu menjelaskan 'mengapa', bukan 'apa'. "
            "Tapi saya tidak mengerti bedanya. "
            "Apa yang dimaksud komentar yang baik dalam konteks penulisan pseudocode?"
        ),
        "relevant_keywords": ["pseudocode", "komentar", "dokumentasi", "algoritme", "penulisan", "konvensi"],
        "cognitive": "2TAR",
        "session_id": "eval-296",
        "query_type": "gap",
        "context_note": "Penulisan komentar yang baik dalam pseudocode — bukan tentang sintaks tapi praktik",
    },

    # 297
    {
        "query": (
            "Saya dengar ada prinsip 'KISS' (Keep It Simple, Stupid) dalam programming. "
            "Apakah prinsip ini berkaitan dengan pilar CT? "
            "Misalnya apakah KISS itu implementasi dari abstraksi, "
            "atau ada pilar lain yang lebih relevan?"
        ),
        "relevant_keywords": ["KISS", "abstraksi", "dekomposisi", "CT", "prinsip", "sederhana", "modular"],
        "cognitive": "3TAI",
        "session_id": "eval-297",
        "query_type": "out_of_scope",
        "context_note": "Prinsip KISS dikaitkan dengan pilar CT — tidak eksplisit di GT",
    },

    # 298
    {
        "query": (
            "Dosen bilang salah satu manfaat CT adalah kemampuan 'transfer learning' — "
            "solusi untuk satu masalah bisa ditransfer ke masalah serupa. "
            "Ini berkaitan dengan pilar apa dalam AADP? "
            "Dan bagaimana cara melatih kemampuan ini secara sadar?"
        ),
        "relevant_keywords": ["pattern recognition", "transfer", "CT", "AADP", "generalisasi", "algoritme", "abstraksi"],
        "cognitive": "4TGR",
        "session_id": "eval-298",
        "query_type": "gap",
        "context_note": "Transfer learning sebagai kemampuan CT — dikaitkan dengan pattern recognition",
    },

    # 299
    {
        "query": (
            "Dosen saya bilang debugging kode itu sebenarnya latihan CT — "
            "khususnya dekomposisi dan pattern recognition. "
            "Saya tidak melihat hubungannya. "
            "Bagaimana cara kerja debugging yang dikerjakan dengan pendekatan CT?"
        ),
        "relevant_keywords": ["dekomposisi", "pattern recognition", "debugging", "CT", "AADP", "algoritme", "masalah"],
        "cognitive": "3PGI",
        "session_id": "eval-299",
        "query_type": "cross_topic",
        "context_note": "Debugging sebagai aplikasi CT — dekomposisi dan pattern recognition dalam praktik",
    },

    # 300
    {
        "query": (
            "Saya hampir selesai belajar CT semester ini dan ingin tahu: "
            "dari semua yang dipelajari — dekomposisi, abstraksi, pattern recognition, "
            "algoritme, struktur data, kompleksitas — "
            "menurut materi CT di IPB, pilar atau konsep mana yang paling fundamental "
            "dan menjadi landasan dari semua konsep lainnya?"
        ),
        "relevant_keywords": ["CT", "AADP", "dekomposisi", "abstraksi", "algoritme", "fondasi", "pilar"],
        "cognitive": "6PAR",
        "session_id": "eval-300",
        "query_type": "gap",
        "context_note": "Mahasiswa mencari konsep paling fundamental dalam CT — pertanyaan reflektif akhir semester",
    },

]


# ── Sanity check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert len(TEST_CASES) == 100, f"Expected 100, got {len(TEST_CASES)}"
    ids = [tc["session_id"] for tc in TEST_CASES]
    assert len(set(ids)) == 100, "Duplicate session_id detected!"
    for tc in TEST_CASES:
        assert "reference_answer" not in tc, \
            f"reference_answer found in {tc['session_id']}"
    from collections import Counter
    types  = Counter(tc["query_type"] for tc in TEST_CASES)
    levels = Counter(tc["cognitive"][0] for tc in TEST_CASES)
    print(f"✅ {len(TEST_CASES)} test cases — eval-201 s/d eval-300")
    print(f"   Query types : {dict(types)}")
    print(f"   Cog levels  : {dict(sorted(levels.items()))}")
