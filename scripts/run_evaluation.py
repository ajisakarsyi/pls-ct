"""
scripts/run_evaluation.py
──────────────────────────
CLI entry-point for the RAG evaluation suite.

The evaluator uses LOCAL Ollama for both embeddings and answer-quality scoring
so no remote API token is consumed during automated testing.

Prerequisites
─────────────
1. Ollama must be running:
       ollama serve

2. Required models must be pulled:
       ollama pull nomic-embed-text   # embeddings
       ollama pull llama3             # or your chat model

3. The FastAPI app server must be running:
       python main.py

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --ollama-url http://localhost:11434 \\
                                     --chat-model llama3 \\
                                     --embed-model nomic-embed-text \\
                                     --app-url http://127.0.0.1:8000 \\
                                     --pace-min 1.0 --pace-max 3.0
"""

import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import get_settings
from evaluation.runner import (
    run_evaluation,
    compute_aggregates,
    run_offline_analysis,
    save_results,
)


def _check_ollama(base_url: str, embed_model: str, chat_model: str) -> bool:
    """Verify Ollama is reachable and required models are available."""
    print(f"  Checking Ollama at {base_url}…")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
        # Full names like "llama3:latest", base names like "llama3"
        full_names = [m["name"] for m in raw_models]
        base_names = [n.split(":")[0] for n in full_names]

        if not full_names:
            print("  ⚠️  Ollama is running but NO models are pulled.")
            print(f"     Pull what you need, e.g.:")
            print(f"       ollama pull {embed_model}")
            print(f"       ollama pull {chat_model}")
            return False

        print(f"  Available models: {', '.join(full_names)}")
        ok = True
        for requested in [embed_model, chat_model]:
            base_req = requested.split(":")[0]
            if requested in full_names or base_req in base_names:
                print(f"  ✅ {requested} — found")
            else:
                # Find close matches to help the user
                close = [n for n in full_names if base_req in n or n.split(":")[0] in requested]
                print(f"  ❌ {requested} — NOT found.")
                if close:
                    print(f"     Did you mean one of: {', '.join(close)}?")
                    print(f"     Use --chat-model or --embed-model to override, e.g.:")
                    print(f"       --chat-model {close[0]}")
                else:
                    print(f"     Pull it with:  ollama pull {requested}")
                ok = False
        return ok
    except Exception as exc:
        print(f"  ❌ Cannot reach Ollama: {exc}")
        print(f"     Make sure Ollama is running:  ollama serve")
        return False


def _check_app(app_url: str) -> bool:
    """Verify the FastAPI app server is reachable."""
    print(f"  Checking app server at {app_url}…")
    try:
        resp = requests.get(f"{app_url}/cognitive-types", timeout=5)
        resp.raise_for_status()
        print(f"  ✅ App server responding")
        return True
    except Exception as exc:
        print(f"  ❌ App server unreachable: {exc}")
        print(f"     Start it with:  python main.py")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the CSIPBLLM RAG evaluation suite (Ollama local mode)."
    )
    parser.add_argument("--app-url",     default=os.getenv("EVAL_BASE_URL",    "http://127.0.0.1:8000"))
    parser.add_argument("--ollama-url",  default=os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434"))
    parser.add_argument("--chat-model",  default=os.getenv("OLLAMA_CHAT_MODEL", os.getenv("CHAT_MODEL", "llama3")))
    parser.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--pace-min",    type=float, default=float(os.getenv("PACE_MIN", "2.5")))
    parser.add_argument("--pace-max",    type=float, default=float(os.getenv("PACE_MAX", "5.5")))
    parser.add_argument("--skip-checks", action="store_true", help="Skip pre-flight connectivity checks")
    parser.add_argument(
        "--batch", choices=["1", "2", "all"], default="all",
        help="Batch test case: 1=eval-001..110, 2=eval-201..300, all=gabungan (default).",
    )
    args = parser.parse_args()

    # Propagate to env so runner.py picks them up
    os.environ["EVAL_BASE_URL"]    = args.app_url
    os.environ["OLLAMA_BASE_URL"]  = args.ollama_url
    os.environ["OLLAMA_CHAT_MODEL"]= args.chat_model
    os.environ["OLLAMA_EMBED_MODEL"]= args.embed_model
    os.environ["PACE_MIN"]         = str(args.pace_min)
    os.environ["PACE_MAX"]         = str(args.pace_max)
    os.environ["TEST_BATCH"]       = args.batch

    print("\n" + "=" * 60)
    print("  CSIPBLLM — RAG Evaluation Suite (Ollama local mode)")
    print("=" * 60)

    if not args.skip_checks:
        print("\n🔍 Pre-flight checks:")
        ollama_ok = _check_ollama(args.ollama_url, args.embed_model, args.chat_model)
        app_ok    = _check_app(args.app_url)
        if not (ollama_ok and app_ok):
            print("\n❌  Pre-flight failed. Fix the issues above and re-run.")
            sys.exit(1)
        print()

    settings = get_settings()
    results   = run_evaluation()
    aggregate = compute_aggregates(results)

    history_path = os.path.join(settings.history_dir, "conversation_log.json")
    offline = run_offline_analysis(history_path)

    json_path, csv_path, txt_path, response_path = save_results(
        results, aggregate, offline, settings.eval_results_dir
    )

    print("\n✅  Evaluation complete.")
    print(f"   JSON      : {json_path}")
    print(f"   CSV       : {csv_path}")
    print(f"   Report    : {txt_path}")
    print(f"   Responses : {response_path}")  # full LLM replies for thesis justification

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