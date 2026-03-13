# Dekomposisi (Decomposition)

**Dekomposisi** adalah proses memecah masalah kompleks yang besar menjadi beberapa sub-masalah yang lebih kecil dan lebih mudah dikelola. Ini adalah pilar utama dari Computational Thinking (CT).

**Pentingnya:**
1.  **Mengurangi Kompleksitas:** Masalah besar terasa menakutkan, masalah kecil lebih fokus.
2.  **Memungkinkan Solusi Paralel:** Setiap sub-masalah bisa dikerjakan/diselesaikan secara independen.
3.  **Memfasilitasi Pengujian:** Lebih mudah menguji bagian per bagian daripada keseluruhan sistem.

**Contoh Praktis:**
Membuat aplikasi *e-commerce* dapat didekomposisi menjadi:
1.  Sistem Manajemen Pengguna (Login, Register).
2.  Sistem Katalog Produk (Tampilkan, Cari).
3.  Sistem Keranjang Belanja.
4.  Sistem Pembayaran.
5.  Sistem Pengiriman.

Setiap bagian ini kemudian dapat dipecah lagi. Misalnya, Sistem Pembayaran dapat dipecah menjadi: integrasi *payment gateway*, validasi *voucher*, dan konfirmasi transaksi.