"""
app/core/cognitive.py
──────────────────────
All 48 cognitive-type codes and their human-readable labels.
"""

from typing import List

# 6 levels × 2 (P/T) × 2 (A/G) × 2 (I/R) = 48 types
VALID_COGNITIVE_TYPES: List[str] = [
    f"{n}{pt}{ag}{ir}"
    for n  in ["1", "2", "3", "4", "5", "6"]
    for pt in ["P", "T"]
    for ag in ["A", "G"]
    for ir in ["I", "R"]
]

DEFAULT_COGNITIVE_TYPE = "1PAR"


def cognitive_label(code: str) -> str:
    """Return a human-readable label for a 4-char cognitive code, e.g. '3TGR'."""
    code = code.upper()
    if len(code) != 4:
        return f"Tipe Kognitif {code}"
    pt = "Praktis"    if code[1] == "P" else "Teoretis"
    ag = "Analitis"   if code[2] == "A" else "Global"
    ir = "Individual" if code[3] == "I" else "Relasional"
    return f"Level {code[0]} — {pt}, {ag}, {ir} ({code})"


def is_valid(code: str) -> bool:
    return code.upper() in VALID_COGNITIVE_TYPES
