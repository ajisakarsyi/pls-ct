"""
evaluation/lexicon.py
──────────────────────
LEKSIKON Uncertainty & Contradiction — versi revisi pasca-sidang.
Satu-satunya sumber kebenaran (single source of truth) untuk:

  - evaluation/faithfulness.py  → komponen Contradiction & Uncertainty pada
                                  Hallucination Risk (Persamaan 18 skripsi)
  - app/services/live_metrics.py→ pemindaian langsung saat demo
  - evaluation/excel_report.py  → sheet "Leksikon" dan sheet
                                  "Uncertainty & Contradiction" (lokasi frasa)

PERUBAHAN DARI VERSI SIDANG (untuk pembaruan Lampiran 5):
  1. Frasa penanda wacana yang bukan keraguan epistemik DIHAPUS dari daftar
     Uncertainty ("perlu dicatat", "perlu diingat", "namun perlu",
     "tergantung pada") — frasa-frasa itu adalah gaya penulisan tutor yang
     wajar dan terbukti memicu false positive (lihat pembahasan Tabel 32).
  2. Frasa kontradiksi generik yang sering muncul di penjelasan komparatif
     yang sah DIHAPUS dari daftar Contradiction ("sebaliknya" berdiri
     sendiri, "berbeda dengan") dan diganti bentuk yang lebih tegas
     ("justru sebaliknya", "bertolak belakang").
  3. Ditambahkan padanan bahasa Inggris karena llama3 sesekali menyisipkan
     frasa Inggris dalam respons berbahasa Indonesia.
  4. Pencocokan kini berbasis batas kata (word boundary) dan melaporkan
     LOKASI setiap temuan (indeks kalimat + posisi karakter) agar hasil
     evaluasi dapat menunjukkan "frasa X ada di kalimat ke-N".

Definisi komponen mengikuti Persamaan 18 skripsi:
  Uncertainty   = 0,20 jika terdapat ≥1 frasa ketidakpastian (biner).
  Contradiction = proporsi kalimat yang mengandung frasa kontradiksi kuat.
"""

import re
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════════
# DAFTAR FRASA (huruf kecil; pencocokan case-insensitive + word boundary)
# ══════════════════════════════════════════════════════════════════════════

#: Frasa keraguan epistemik — memicu komponen Uncertainty (0,20 biner).
UNCERTAINTY_PHRASES: List[str] = [
    # ── pengakuan ketidaktahuan eksplisit ────────────────────────────────
    "saya tidak yakin", "saya kurang yakin", "saya tidak tahu",
    "saya kurang tahu", "saya tidak memiliki informasi",
    # ── ketidakpastian faktual ───────────────────────────────────────────
    "belum pasti", "tidak pasti", "belum dapat dipastikan",
    "tidak dapat dipastikan", "belum jelas", "kurang jelas apakah",
    # ── penanda spekulasi / dugaan ───────────────────────────────────────
    "mungkin", "mungkin saja", "kemungkinan", "kemungkinan besar",
    "bisa jadi", "boleh jadi", "barangkali", "diperkirakan", "diduga",
    # ── inferensi lemah ──────────────────────────────────────────────────
    "sepertinya", "tampaknya", "kelihatannya", "agaknya",
    # ── padanan bahasa Inggris (llama3 kadang code-switching) ────────────
    "i'm not sure", "i am not sure", "not sure", "uncertain",
    "it seems", "it appears", "perhaps", "possibly", "probably",
    "might be", "may be", "presumably",
]

#: Frasa kontradiksi kuat — komponen Contradiction (proporsi kalimat).
CONTRADICTION_PHRASES: List[str] = [
    # ── negasi/koreksi eksplisit ─────────────────────────────────────────
    "tidak benar", "sebenarnya salah", "salah besar", "salah kaprah",
    "keliru", "bukan demikian", "tidak demikian", "menyesatkan",
    "anggapan itu salah", "pernyataan itu salah",
    # ── pembalikan / pertentangan tegas ──────────────────────────────────
    "justru sebaliknya", "yang benar adalah", "bertentangan dengan",
    "bertolak belakang", "berlawanan dengan",
    # ── penanda inkonsistensi ────────────────────────────────────────────
    "kontradiksi", "kontradiktif", "tidak konsisten",
    # ── padanan bahasa Inggris ───────────────────────────────────────────
    "contradicts", "contradiction", "on the contrary",
    "that is incorrect", "this is wrong", "actually false",
    "inconsistent with",
]

# ══════════════════════════════════════════════════════════════════════════
# PEMINDAI BERBASIS POSISI
# ══════════════════════════════════════════════════════════════════════════

def _compile(phrases: List[str]) -> List[Tuple[str, "re.Pattern"]]:
    """
    Kompilasi frasa → regex batas kata, diurutkan dari frasa TERPANJANG
    agar "mungkin saja" tertangkap sebagai satu temuan (bukan "mungkin"
    lalu "saja" terpisah).
    """
    out = []
    for p in sorted(phrases, key=len, reverse=True):
        pat = re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)", re.IGNORECASE)
        out.append((p, pat))
    return out


_UNC_COMPILED = _compile(UNCERTAINTY_PHRASES)
_CON_COMPILED = _compile(CONTRADICTION_PHRASES)


def split_sentences_with_spans(text: str) -> List[Dict]:
    """
    Pecah teks menjadi kalimat SAMBIL mempertahankan posisi karakter asli.

    Returns list of {index, start, end, text} — dipakai agar lokasi frasa
    dapat dilaporkan sebagai "kalimat ke-N, karakter X–Y".
    """
    sentences: List[Dict] = []
    if not text:
        return sentences
    # Batas kalimat: setelah . ! ? atau baris baru.
    boundaries = [m.end() for m in re.finditer(r"[.!?]+(?=\s)|\n+", text)]
    starts = [0] + boundaries
    ends = boundaries + [len(text)]
    idx = 0
    for s, e in zip(starts, ends):
        seg = text[s:e].strip()
        if not seg:
            continue
        # posisi awal segmen ter-strip di teks asli
        offset = text.index(seg[0], s, e) if seg else s
        sentences.append({"index": idx, "start": offset,
                          "end": offset + len(seg), "text": seg})
        idx += 1
    return sentences


def _scan(text: str, compiled) -> List[Dict]:
    """Temukan seluruh frasa beserta lokasinya (tanpa tumpang tindih)."""
    matches: List[Dict] = []
    covered: List[Tuple[int, int]] = []
    sentences = split_sentences_with_spans(text)

    def _sentence_of(pos: int) -> Dict:
        for s in sentences:
            if s["start"] <= pos < s["end"]:
                return s
        return {"index": -1, "text": "", "start": pos, "end": pos}

    for phrase, pat in compiled:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            if any(a < span[1] and span[0] < b for a, b in covered):
                continue  # sudah tercakup frasa yang lebih panjang
            covered.append(span)
            sent = _sentence_of(m.start())
            matches.append({
                "phrase":         phrase,
                "char_start":     m.start(),
                "char_end":       m.end(),
                "sentence_index": sent["index"],      # kalimat ke-(index+1)
                "sentence":       sent["text"][:300],
            })
    matches.sort(key=lambda d: d["char_start"])
    return matches


def uncertainty_scan(text: str) -> Dict:
    """
    Komponen Uncertainty (Persamaan 18): biner 0,20 bila ada frasa keraguan.

    Returns
    -------
    {value: 0.0|0.2, flag: bool, matches: [ {phrase, sentence_index,
     char_start, char_end, sentence} ]}
    """
    matches = _scan(text or "", _UNC_COMPILED)
    return {
        "value":   0.20 if matches else 0.0,
        "flag":    bool(matches),
        "matches": matches,
    }


def contradiction_scan(text: str) -> Dict:
    """
    Komponen Contradiction (Persamaan 18): PROPORSI KALIMAT yang memuat
    ≥1 frasa kontradiksi kuat (sesuai definisi skripsi — bukan +0,2 per
    pola seperti implementasi lama).

    Returns
    -------
    {value: float 0..1, n_sentences, n_flagged_sentences, matches: [...]}
    """
    text = text or ""
    matches = _scan(text, _CON_COMPILED)
    sentences = split_sentences_with_spans(text)
    n = len(sentences)
    flagged = {m["sentence_index"] for m in matches if m["sentence_index"] >= 0}
    value = round(len(flagged) / n, 4) if n else 0.0
    return {
        "value":               value,
        "n_sentences":         n,
        "n_flagged_sentences": len(flagged),
        "matches":             matches,
    }


def lexicon_table() -> List[Dict]:
    """Tabel leksikon lengkap untuk sheet Excel & pembaruan Lampiran 5."""
    rows = []
    for p in UNCERTAINTY_PHRASES:
        rows.append({"jenis": "Uncertainty", "frasa": p})
    for p in CONTRADICTION_PHRASES:
        rows.append({"jenis": "Contradiction", "frasa": p})
    return rows
