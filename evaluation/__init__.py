from .metrics import (
    cosine_similarity,
    precision_at_k,
    recall_at_k,
    mean_similarity,
    coverage_score,
    source_diversity,
)
from .faithfulness import evaluate_faithfulness, detect_hallucination
from .test_cases import TEST_CASES
from .runner import run_evaluation, compute_aggregates, run_offline_analysis, save_results

__all__ = [
    "cosine_similarity",
    "precision_at_k",
    "recall_at_k",
    "mean_similarity",
    "coverage_score",
    "source_diversity",
    "evaluate_faithfulness",
    "detect_hallucination",
    "TEST_CASES",
    "run_evaluation",
    "compute_aggregates",
    "run_offline_analysis",
    "save_results",
]
