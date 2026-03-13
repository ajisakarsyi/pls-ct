"""
tests/test_core.py
───────────────────
Unit tests for pure utility functions — no API calls or file I/O required.
"""

import pytest

from app.core.cognitive import cognitive_label, is_valid, VALID_COGNITIVE_TYPES
from app.utils.latex import normalize_latex
from app.utils.code_detector import is_code_like
from evaluation.metrics import (
    cosine_similarity,
    precision_at_k,
    recall_at_k,
    mean_similarity,
    coverage_score,
    source_diversity,
)

import numpy as np


# ── Cognitive helpers ──────────────────────────────────────────────────────

def test_valid_cognitive_count():
    assert len(VALID_COGNITIVE_TYPES) == 48


def test_is_valid_true():
    assert is_valid("3TGR")
    assert is_valid("1PAI")
    assert is_valid("6TAR")


def test_is_valid_false():
    assert not is_valid("7PAR")
    assert not is_valid("ABC")
    assert not is_valid("")


def test_cognitive_label_format():
    label = cognitive_label("3TGR")
    assert "Level 3" in label
    assert "Teoretis" in label
    assert "Global" in label
    assert "Relasional" in label


# ── LaTeX normaliser ───────────────────────────────────────────────────────

def test_normalize_dollar_inline():
    result = normalize_latex("Let $x = 1$ be true.")
    assert r"\(" in result and r"\)" in result


def test_normalize_double_dollar_block():
    result = normalize_latex("We have $$E = mc^2$$")
    assert r"\[" in result and r"\]" in result


def test_normalize_preserves_code():
    code = "```python\nprint('hello')\n```"
    result = normalize_latex(code)
    assert "print('hello')" in result


# ── Code detector ──────────────────────────────────────────────────────────

def test_code_like_for_loop():
    assert is_code_like("for i in range(10):")


def test_code_like_def():
    assert is_code_like("def my_func():")


def test_code_like_plain_text():
    assert not is_code_like("Jelaskan apa itu algoritma?")


# ── RAG metrics ────────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(v, v), abs=1e-6) == 1.0


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == 0.0


def test_precision_at_k():
    # 3 out of 4 scores above threshold 0.3
    scores = [0.8, 0.6, 0.1, 0.5]
    assert precision_at_k(scores, 4, threshold=0.3) == pytest.approx(0.75)


def test_recall_at_k():
    scores = [0.8, 0.6, 0.1, 0.5]
    # 3 relevant found out of 3 total relevant
    assert recall_at_k(scores, 3, 4, threshold=0.3) == pytest.approx(1.0)


def test_mean_similarity():
    assert mean_similarity([0.5, 0.7, 0.3]) == pytest.approx(0.5)


def test_coverage_score():
    assert coverage_score([0.8, 0.3, 0.1], threshold=0.25) == pytest.approx(2 / 3)


def test_source_diversity_all_unique():
    assert source_diversity(["a", "b", "c"]) == pytest.approx(1.0)


def test_source_diversity_all_same():
    assert source_diversity(["a", "a", "a"]) == pytest.approx(1 / 3)
