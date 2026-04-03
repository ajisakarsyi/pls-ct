"""
evaluation/metrics.py
──────────────────────
Pure metric functions for RAG evaluation.

All functions are stateless and take plain Python / NumPy values so they
can be unit-tested without any live API calls.
"""

from typing import List, Optional

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    sim(q, d) = (q · d) / (||q|| · ||d||)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def precision_at_k(scores: List[float], k: int, threshold: float = 0.30) -> float:
    """
    Precision@K = |{s ∈ Top-K : s >= threshold}| / K
    """
    if k == 0:
        return 0.0
    return sum(1 for s in scores[:k] if s >= threshold) / k


def recall_at_k(
    scores: List[float], total_relevant: int, k: int, threshold: float = 0.30
) -> float:
    """
    Recall@K = |{s ∈ Top-K : s >= threshold}| / total_relevant
    """
    if total_relevant == 0:
        return 1.0
    return sum(1 for s in scores[:k] if s >= threshold) / total_relevant


def mean_similarity(scores: List[float]) -> float:
    """
    MeanSim = (1/K) * Σ si
    """
    return sum(scores) / len(scores) if scores else 0.0


def coverage_score(scores: List[float], threshold: float = 0.25) -> float:
    """
    Coverage = |{s : s >= threshold}| / len(scores)
    """
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0


def source_diversity(sources: List[str]) -> float:
    """
    Diversity = |unique sources| / K
    """
    return len(set(sources)) / len(sources) if sources else 0.0
