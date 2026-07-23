"""
topics.py  (root-level shared module)
─────────────────────────────────────
Resolusi TOPIK dokumen materi dari nama file — dipakai bersama oleh:
  - app/services/rag.py          → log retrieval "chunk X dari file Y (topik Z)"
  - app/services/live_metrics.py → panel transparansi demo
  - evaluation/runner.py         → log evaluasi & sheet "Chunk & Sumber" di Excel

Mengapa modul terpisah di root?
  Supaya `app/` dan `evaluation/` sama-sama bisa mengimpor tanpa saling
  bergantung (evaluation suite tetap bisa berjalan mandiri tanpa FastAPI).

Cara kerja `topic_of()`:
  1. Baca beberapa baris pertama file materi (header Ground Truth).
     Semua file materi GT memiliki header seperti:
        "Pertemuan 4: Problem Solving — Abstraction"     (file GT_CT*.txt)
        "Subtopik: Rekursi (Recursion)"                  (file GT_SUBTOPIK_*.txt)
        "KOMBINASI KOGNITIF: 3TAR"                       (file profil kognitif)
  2. Jika header tidak ditemukan, jatuhkan ke derivasi dari nama file
     (misal GT_DETAIL_PT11_Perulangan.txt → "Detail Pertemuan 11 — Perulangan").
  3. Hasil di-cache per nama file agar file tidak dibaca berulang kali.
"""

import os
import re
from functools import lru_cache
from typing import Optional

# Pola kode profil kognitif: 1-6 + P/T + A/G + I/R  (contoh: 3TAR, 1PGI)
_COG_CODE_RE = re.compile(r"^[1-6][PT][AG][IR]$")

_MATERIALS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials")


def _from_filename(stem: str) -> str:
    """Derivasi topik dari nama file bila header tidak tersedia."""
    if _COG_CODE_RE.match(stem.upper()):
        return f"Profil Kognitif {stem.upper()}"

    m = re.match(r"GT_SUBTOPIK_(\d+)_(.+)", stem, re.IGNORECASE)
    if m:
        return f"Subtopik {int(m.group(1)):02d} — {m.group(2).replace('_', ' ')}"

    m = re.match(r"GT_DETAIL_PT(\d+)_(.+)", stem, re.IGNORECASE)
    if m:
        return f"Detail Pertemuan {int(m.group(1))} — {m.group(2).replace('_', ' ')}"

    m = re.match(r"GT_CT(\d+)", stem, re.IGNORECASE)
    if m:
        return f"Materi Pertemuan CT-{int(m.group(1)):02d}"

    return stem.replace("_", " ")


def _from_header(path: str) -> Optional[str]:
    """Ambil topik dari baris header file GT (jika ada)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = [fh.readline() for _ in range(8)]
    except Exception:
        return None

    for line in head:
        line = (line or "").strip()
        if not line or set(line) <= {"=", "-", "─"}:
            continue  # garis dekorasi
        m = re.match(r"Pertemuan\s+\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return line  # "Pertemuan 4: Problem Solving — Abstraction"
        m = re.match(r"Subtopik\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return f"Subtopik: {m.group(1).strip()}"
        m = re.match(r"KOMBINASI\s+KOGNITIF\s*:\s*([0-9A-Z]+)", line, re.IGNORECASE)
        if m:
            return f"Profil Kognitif {m.group(1).upper()}"
    return None


@lru_cache(maxsize=512)
def topic_of(source_fname: str, materials_dir: Optional[str] = None) -> str:
    """
    Kembalikan deskripsi topik untuk sebuah nama file materi.

    Parameters
    ----------
    source_fname : nama file materi, mis. "GT_SUBTOPIK_11_Rekursi.txt"
    materials_dir: direktori materials (default: <root>/materials)

    Returns
    -------
    str — topik terbaca-manusia, mis. "Subtopik: Rekursi (Recursion)".
    """
    stem = os.path.splitext(os.path.basename(source_fname))[0]
    mdir = materials_dir or _MATERIALS_DIR_DEFAULT
    path = os.path.join(mdir, os.path.basename(source_fname))

    if os.path.isfile(path):
        topic = _from_header(path)
        if topic:
            return topic
    return _from_filename(stem)
