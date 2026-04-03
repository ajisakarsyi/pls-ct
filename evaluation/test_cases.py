"""
evaluation/test_cases.py
─────────────────────────
Static test-case dataset used by the evaluation suite.

Each entry contains:
  query             — the student question
  reference_answer  — ground-truth (used as correct answer for /evaluate)
  relevant_keywords — keywords that should appear in retrieved chunks
  cognitive         — cognitive type code to test with
  session_id        — unique session ID for this test run
"""

from typing import Dict, List

TEST_CASES: List[Dict] = [
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
