"""
app/utils/code_detector.py
───────────────────────────
Heuristic to detect whether a user message contains code.
"""

import re

_CODE_REGEX = re.compile(
    r"```[\s\S]*?```|(\bfor\b|\bwhile\b|\bif\b|\bdef\b|\bprint\b|\breturn\b|;|==)"
)


def is_code_like(text: str) -> bool:
    """Return True if *text* looks like it contains code snippets."""
    return bool(_CODE_REGEX.search(text or ""))
