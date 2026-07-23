"""
app/core/verbose.py
────────────────────
Utilitas TRANSPARANSI TERMINAL — jawaban atas catatan sidang bahwa sistem
terasa "blackbox". Semua aktivitas latar belakang dicetak ke terminal:

  - pemotongan dokumen menjadi chunk (per file: jumlah chunk & ukuran),
  - pembuatan embedding (model, dimensi vektor, durasi),
  - pemilihan chunk saat retrieval (peringkat, skor, NAMA FILE, TOPIK,
    lolos/tidaknya terhadap ambang θ dan θc),
  - prompt final yang dikirim ke LLM (bukti isi konteks Kondisi A vs
    kosongnya konteks Kondisi B),
  - perhitungan metrik langkah demi langkah saat demo maupun evaluasi.

Kendali:
  LOGICT_VERBOSE=0        → matikan seluruh keluaran transparansi (default: 1/aktif)
  LOGICT_PROMPT_ECHO=head → cetak hanya 800 karakter pertama prompt
                            (default: full — cetak prompt utuh)

Semua fungsi aman dipanggil dari thread mana pun (hanya menulis ke stdout).
"""

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

WIDTH = 78


def enabled() -> bool:
    """True bila mode transparansi terminal aktif (env LOGICT_VERBOSE != '0')."""
    return os.getenv("LOGICT_VERBOSE", "1") != "0"


def _emit(text: str = "") -> None:
    if enabled():
        print(text, file=sys.stdout, flush=True)


def rule(char: str = "─") -> None:
    """Cetak garis pemisah selebar WIDTH."""
    _emit(char * WIDTH)


def banner(title: str, char: str = "═") -> None:
    """Cetak judul blok besar, mis. banner('RETRIEVAL — KONDISI A')."""
    _emit()
    _emit(char * WIDTH)
    _emit(f"  {title}")
    _emit(char * WIDTH)


def section(title: str) -> None:
    """Cetak sub-judul blok."""
    _emit(f"\n──[ {title} ]" + "─" * max(0, WIDTH - len(title) - 6))


def step(msg: str) -> None:
    """Cetak satu langkah proses."""
    _emit(f"  ▸ {msg}")


def kv(key: str, value: Any, indent: int = 4) -> None:
    """Cetak pasangan kunci-nilai rata kiri."""
    _emit(f"{' ' * indent}{key:<28}: {value}")


def note(msg: str) -> None:
    """Catatan kecil / keterangan tambahan."""
    _emit(f"      · {msg}")


def calc(label: str, formula: str = None, substitution: str = None,
         result: Any = None) -> None:
    """
    Cetak satu perhitungan metrik secara eksplisit.

    Dua cara pakai:
      calc("P@6 = |{s ≥ 0,25}| / 6 = 6/6 = 1,0000")
          → satu baris calc_str siap pakai (dari *_detail()["calc_str"])
      calc("Coverage (Pers. 8)", "|{s ≥ θc}| / K", "4 / 6", 0.6667)
          → format panjang: label + rumus + substitusi + hasil.
    """
    if formula is None:
        _emit(f"    ∴ {label}")
        return
    _emit(f"    {label}")
    _emit(f"        rumus      : {formula}")
    _emit(f"        substitusi : {substitution}")
    _emit(f"        hasil      : {result}")


def chunk_table(rows: Iterable[Dict[str, Any]],
                theta: Optional[float] = None,
                theta_c: Optional[float] = None) -> None:
    """
    Cetak tabel chunk hasil retrieval.

    Setiap row: {rank, score, source, topic}
    Kolom ≥θ / ≥θc ditampilkan bila ambang diberikan — memperlihatkan
    keputusan Precision@K (θ) dan Coverage (θc) per chunk.
    """
    if not enabled():
        return
    head = f"    {'#':>2}  {'skor':>7}"
    if theta is not None:
        head += f"  {'≥θ=' + format(theta, '.2f'):>8}"
    if theta_c is not None:
        head += f"  {'≥θc=' + format(theta_c, '.2f'):>9}"
    head += f"  {'file sumber':<38}  topik"
    _emit(head)
    for i, r in enumerate(rows, 1):
        line = f"    {r.get('rank', i):>2}  {r.get('score', 0.0):>7.4f}"
        if theta is not None:
            line += f"  {'✓' if r.get('score', 0.0) >= theta else '✗':>8}"
        if theta_c is not None:
            line += f"  {'✓' if r.get('score', 0.0) >= theta_c else '✗':>9}"
        line += f"  {str(r.get('source', '-')):<38}  {r.get('topic', '-')}"
        _emit(line)


def prompt_echo(title: str, prompt: str) -> None:
    """
    Cetak prompt final yang dikirim ke LLM.

    Ini bukti transparansi paling penting untuk Kondisi B: terlihat bahwa
    prompt HANYA berisi pertanyaan mahasiswa, tanpa konteks materi, tanpa
    profil kognitif, tanpa instruksi tutor.
    """
    if not enabled():
        return
    mode = os.getenv("LOGICT_PROMPT_ECHO", "full").lower()
    shown = prompt if mode == "full" else prompt[:800] + (
        f"\n… [{len(prompt) - 800} karakter berikutnya disembunyikan; "
        f"set LOGICT_PROMPT_ECHO=full untuk melihat utuh]" if len(prompt) > 800 else "")
    section(title)
    kv("panjang prompt", f"{len(prompt)} karakter")
    _emit("    ┌" + "─" * (WIDTH - 6) + "┐")
    for ln in shown.splitlines() or [""]:
        _emit(f"    │ {ln}")
    _emit("    └" + "─" * (WIDTH - 6) + "┘")
