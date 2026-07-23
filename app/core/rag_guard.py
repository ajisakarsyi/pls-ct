"""
app/core/rag_guard.py
──────────────────────
BUKTI RUNTIME bahwa Kondisi B benar-benar LLM murni tanpa RAG.

Kekhawatiran dosen penguji: "saat pengujian Kondisi B, bagaimana memastikan
Kondisi B memang tidak menggunakan RAG?"  Modul ini menjawabnya dengan TIGA
lapis bukti, bukan sekadar janji di kode:

  LAPIS 1 — Pemisahan struktural
      Jalur Kondisi B (app/services/tutor.py → generate_reply(mode="B"))
      tidak pernah memanggil retrieve() maupun get_embedding(); prompt-nya
      dibangun hanya dari pertanyaan mentah (identik dengan Lampiran 4).

  LAPIS 2 — Guard runtime (modul ini)
      Selama permintaan Kondisi B berlangsung, guard diaktifkan.
      JIKA ada baris kode mana pun (sekarang atau di masa depan) yang
      mencoba memanggil retrieve() / get_embedding(), guard akan
      MELEMPAR RAGBlockedError sehingga permintaan gagal dengan keras —
      mustahil RAG "diam-diam" ikut bekerja.

  LAPIS 3 — Jejak audit yang bisa diperlihatkan
      Setiap permintaan B menghasilkan laporan audit (no_rag_proof) berisi:
      jumlah panggilan retrieval yang dicegat (harus 0), jumlah panggilan
      embedding (harus 0), serta prompt final utuh yang dikirim ke LLM —
      terlihat sendiri bahwa tidak ada satu pun potongan materi di dalamnya.
      Terminal juga mencetak prompt tersebut (app/core/verbose.py).

Implementasi memakai threading.local sehingga aman ketika Kondisi A dan B
diakses bersamaan dari dua tab peramban: guard hanya berlaku pada thread
yang sedang melayani permintaan Kondisi B.
"""

import threading
from typing import Dict, List, Optional

_state = threading.local()


class RAGBlockedError(RuntimeError):
    """Dilempar bila ada upaya memakai RAG saat guard Kondisi B aktif."""


def _st() -> threading.local:
    if not hasattr(_state, "active"):
        _state.active = False
        _state.blocked = []          # daftar upaya yang dicegat
        _state.embedding_calls = 0   # penghitung panggilan embedding
        _state.retrieval_calls = 0   # penghitung panggilan retrieval
    return _state


class NoRAGGuard:
    """
    Context manager yang mengaktifkan mode 'tanpa-RAG' untuk thread ini.

    Pemakaian (lihat app/services/tutor.py):

        with NoRAGGuard() as guard:
            reply = query_llm_ollama_raw(prompt_buta)
        proof = guard.report()
    """

    def __enter__(self) -> "NoRAGGuard":
        st = _st()
        st.active = True
        st.blocked = []
        st.embedding_calls = 0
        st.retrieval_calls = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _st().active = False

    # ── laporan audit ────────────────────────────────────────────────────
    def report(self) -> Dict:
        """Ringkasan audit untuk payload `no_rag_proof` dan log terminal."""
        st = _st()
        return {
            "guard_enforced":           True,
            "retrieval_calls_blocked":  st.retrieval_calls,
            "embedding_calls_blocked":  st.embedding_calls,
            "blocked_operations":       list(st.blocked),
            "keterangan": (
                "Guard aktif selama permintaan Kondisi B. Setiap upaya "
                "retrieval/embedding akan dicegat dan melempar RAGBlockedError; "
                "nilai 0 berarti tidak ada satu pun upaya penggunaan RAG."
            ),
        }


def guard_active() -> bool:
    """True bila thread ini sedang berada dalam mode Kondisi B (tanpa RAG)."""
    return bool(getattr(_st(), "active", False))


def assert_rag_allowed(operation: str) -> None:
    """
    Dipanggil di gerbang masuk RAG (retrieve, get_embedding).
    Bila guard aktif → catat upaya lalu lempar RAGBlockedError.
    """
    st = _st()
    if not st.active:
        return
    st.blocked.append(operation)
    if operation.startswith("retrieve"):
        st.retrieval_calls += 1
    else:
        st.embedding_calls += 1
    raise RAGBlockedError(
        f"[KONDISI B] Operasi '{operation}' DIBLOKIR — Kondisi B adalah LLM "
        f"murni tanpa RAG. Jika error ini muncul, ada jalur kode yang keliru "
        f"mencoba memakai retrieval; laporkan sebagai bug."
    )
