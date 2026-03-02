# Analisis Kompleksitas Algoritma (Big O Notation)

Analisis kompleksitas adalah proses untuk menentukan bagaimana sumber daya (waktu atau ruang memori) yang dibutuhkan algoritma meningkat seiring dengan peningkatan ukuran input.

## 1. Notasi Big O ($O$)

Notasi Big O mendeskripsikan **batas atas** (worst-case scenario) dari laju pertumbuhan waktu/ruang algoritma.  Kita hanya tertarik pada suku dengan derajat tertinggi karena ia mendominasi pertumbuhan seiring $n \to \infty$.

**Ordo Umum:**
1.  **$O(1)$ (Konstan):** Operasi yang sama, terlepas dari ukuran input. *Contoh: Akses elemen Array berdasarkan indeks.*
2.  **$O(\log n)$ (Logaritmik):** Waktu yang dibutuhkan berkurang setengahnya setiap kali ukuran input berlipat ganda. *Contoh: Binary Search.*
3.  **$O(n)$ (Linear):** Waktu berbanding lurus dengan ukuran input. *Contoh: Linear Search, mencetak semua elemen list.*
4.  **$O(n \log n)$ (Linearithmic):** Algoritma *Divide and Conquer* efisien. *Contoh: Merge Sort, Quick Sort (rata-rata).*
5.  **$O(n^2)$ (Kuadratik):** Waktu tumbuh dua kali lipat lebih cepat dari input. *Contoh: Nested loops, Bubble Sort.*

## 2. Kompleksitas Waktu vs Kompleksitas Ruang

* **Waktu (Time Complexity):** Jumlah operasi dasar yang dieksekusi. Paling sering disingkat menjadi $O(...)$.
* **Ruang (Space Complexity):** Jumlah memori ekstra yang digunakan, tidak termasuk input. Misalnya, Merge Sort memerlukan $O(n)$ ruang ekstra untuk menggabungkan sub-array.

## 3. Kasus Terbaik, Rata-rata, dan Terburuk (Best, Average, Worst Case)

* **Kasus Terbaik ($\Omega$ - Omega Notation):** Batas bawah. Kondisi input yang memberikan kinerja tercepat.
* **Kasus Terburuk ($O$ - Big O Notation):** Batas atas. Kondisi input yang memberikan kinerja paling lambat.
* **Kasus Rata-rata ($\Theta$ - Theta Notation):** Laju pertumbuhan rata-rata yang diharapkan.

**Contoh: Quick Sort**
* **Terburuk:** $O(n^2)$ (terjadi jika pivot selalu elemen terkecil/terbesar).
* **Rata-rata:** $O(n \log n)$.
* **Implikasi (PAR):** Dalam praktik, Quick Sort sering lebih cepat karena *overhead* yang lebih rendah, meskipun secara teoretis Merge Sort lebih stabil.