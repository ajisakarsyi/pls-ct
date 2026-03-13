# Rekursi vs Iterasi: Dua Pendekatan Pemecahan Masalah

Sebagian besar masalah komputasi yang memerlukan pengulangan dapat diselesaikan baik dengan Rekursi maupun Iterasi.

## 1. Iterasi (Iterative Approach)

**Definisi:** Menyelesaikan masalah dengan mengulang blok kode menggunakan struktur kontrol seperti *for loop* atau *while loop*, memodifikasi variabel status secara eksplisit.

**Kelebihan (PAR):**
1.  **Efisiensi Ruang:** Umumnya lebih hemat memori karena tidak melibatkan *function call stack* yang dalam.
2.  **Kejelasan:** Lebih mudah dipahami oleh programmer yang baru belajar dan lebih mudah di-*debug* karena alur kontrol yang linier.

**Contoh:** Menghitung Faktorial secara Iteratif.
$$\text{Faktorial}(n) = n \times (n-1) \times \dots \times 1$$

## 2. Rekursi (Recursive Approach)

**Definisi:** Menyelesaikan masalah dengan memecahnya menjadi sub-masalah yang lebih kecil dari jenis yang sama, dan fungsi memanggil dirinya sendiri.

**Dua Bagian Penting (TAR):**
1.  **Base Case (Kasus Dasar):** Kondisi yang menghentikan rekursi untuk menghindari *infinite loop*.
2.  **Recursive Step (Langkah Rekursif):** Fungsi memanggil dirinya sendiri dengan input yang lebih kecil, bergerak menuju *base case*.

**Kelebihan (TAR):**
1.  **Elegansi:** Kode seringkali lebih ringkas dan secara alami mencerminkan definisi matematis dari masalah (misalnya, *Fibonacci*, *Traversals Tree*).
2.  **Cocok untuk Struktur Hierarkis:** Ideal untuk bekerja dengan struktur data seperti Pohon atau Graf.

**Contoh:** Menghitung Faktorial secara Rekursif. 

## 3. Masalah Stack Overflow

**Isu Kritis (TAR):** Rekursi yang terlalu dalam atau tidak memiliki *base case* yang benar akan menyebabkan *Stack Overflow*. Ini terjadi ketika *call stack* (yang menyimpan informasi setiap fungsi yang sedang berjalan) melebihi batas memori yang dialokasikan.

**Solusi Lanjutan:** Beberapa bahasa pemrograman mendukung **Tail Call Optimization (TCO)**, di mana pemanggilan rekursif terakhir dioptimalkan agar tidak memerlukan *stack frame* baru, secara efektif mengubah rekursi menjadi iterasi di belakang layar.