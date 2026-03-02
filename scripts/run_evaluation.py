"""
scripts/run_evaluation.py
──────────────────────────
CLI entry-point for the RAG evaluation suite.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --base-url http://localhost:8000
"""

import argparse
import os
import sys

# Make sure the project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import get_settings
from evaluation.runner import (
    run_evaluation,
    compute_aggregates,
    run_offline_analysis,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run the CSIPBLLM RAG evaluation suite.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the running FastAPI server.",
    )
    args = parser.parse_args()

    # Patch the base URL so runner uses the CLI value
    os.environ["EVAL_BASE_URL"] = args.base_url

    settings = get_settings()

    print(f"🔬 RAG Evaluation Suite starting…")
    print(f"   Server : {args.base_url}")
    print(f"   Make sure OPENAI_API_KEY is set.\n")

    results   = run_evaluation()
    aggregate = compute_aggregates(results)

    history_path = os.path.join(settings.history_dir, "conversation_log.json")
    offline = run_offline_analysis(history_path)

    json_path, csv_path, txt_path = save_results(
        results, aggregate, offline, settings.eval_results_dir
    )

    print("\n✅  Evaluation complete.")
    print(f"   JSON   : {json_path}")
    print(f"   CSV    : {csv_path}")
    print(f"   Report : {txt_path}")

    ret = aggregate.get("retrieval", {})
    gen = aggregate.get("generation", {})
    aq  = aggregate.get("answer_quality", {})
    print(f"\n  Precision@K        : {ret.get('avg_precision_at_k')}")
    print(f"  Recall@K           : {ret.get('avg_recall_at_k')}")
    print(f"  Mean Similarity    : {ret.get('avg_mean_similarity')}")
    print(f"  Faithfulness       : {gen.get('avg_faithfulness')}")
    print(f"  Hallucination Risk : {gen.get('avg_hallucination_risk')}")
    print(f"  Answer Accuracy    : {aq.get('accuracy')} "
          f"({aq.get('correct')}/{aq.get('total_evaluated')})\n")


if __name__ == "__main__":
    main()
