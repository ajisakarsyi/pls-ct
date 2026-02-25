# Algoritma Pencarian (Searching)

Algoritma pencarian adalah prosedur untuk menemukan elemen tertentu (nilai) di dalam struktur data, seperti *array* atau *list*.

**Pencarian Linear (Linear Search):**
Algoritma paling sederhana. Memeriksa setiap elemen satu per satu dari awal hingga akhir.
-   **Kelebihan:** Bekerja pada data yang tidak terurut.
-   **Kekurangan:** Lambat ($O(n)$).

**Pencarian Biner (Binary Search):**
Hanya bekerja pada data yang **sudah terurut**.
1.  Periksa elemen tengah.
2.  Jika elemen tengah adalah nilai yang dicari, selesai.
3.  Jika nilai yang dicari lebih kecil, cari di paruh kiri.
4.  Jika lebih besar, cari di paruh kanan.
5.  Ulangi langkah ini hingga ditemukan.
-   **Kecepatan:** Sangat cepat ($O(\log n)$).
-   **Prasyarat:** Data harus terurut.