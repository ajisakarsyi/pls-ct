# Struktur Data Fundamental dan Aplikasi

Struktur data adalah cara terorganisir untuk menyimpan dan mengelola data dalam memori komputer, memungkinkan akses dan modifikasi data yang efisien.

## 1. Array vs Linked List

| Fitur | Array | Linked List (Singly) |
| :--- | :--- | :--- |
| **Penyimpanan** | Berdekatan di memori (contiguous) | Tersebar (non-contiguous), dihubungkan oleh pointer |
| **Akses** | Akses acak (O(1)) | Akses sekuensial (O(n)) |
| **Insersi/Hapus** | Lambat (O(n)) karena butuh shifting | Cepat (O(1)) jika posisi pointer diketahui |
| **Ukuran** | Statis/Semi-Dinamis (membutuhkan realokasi) | Dinamis |

**Konsep Kunci (TAR):** Kinerja Linked List dalam *traversal* (O(n)) vs kinerja Array dalam *lookup* (O(1)) terkait langsung dengan **locality of reference** di memori.

## 2. Stack dan Queue

Kedua struktur data ini adalah bentuk abstraksi dari *Linked List* atau *Array* dengan batasan operasi:

* **Stack (Tumpukan):** Menggunakan prinsip **LIFO (Last-In, First-Out)**.
    * **Operasi:** PUSH (menambah), POP (menghapus), PEEK (melihat).
    * **Aplikasi (PAR):** Undo/Redo di perangkat lunak, *function call stack* (rekursi).

* **Queue (Antrian):** Menggunakan prinsip **FIFO (First-In, First-Out)**.
    * **Operasi:** ENQUEUE (menambah), DEQUEUE (menghapus).
    * **Aplikasi (PAR):** Manajemen tugas (task scheduling), antrian pesan (*message queues*).

## 3. Hash Table (Map/Dictionary)

**Definisi:** Struktur data yang mengimplementasikan *array asosiatif* (key-value pair) menggunakan **fungsi hash** untuk menghitung indeks di mana elemen disimpan.

**Manfaat (PAR):** Memberikan kinerja *lookup*, *insertion*, dan *deletion* rata-rata $O(1)$ yang luar biasa cepat.

**Isu Kritis (TAR):** **Collision** (dua kunci menghasilkan indeks hash yang sama). Solusi umum melibatkan *chaining* (menggunakan linked list di setiap *bucket*) atau *open addressing* (mencari slot kosong berikutnya). Memilih fungsi hash yang baik sangat penting.