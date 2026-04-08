"""
simulate_rl/__main__.py
────────────────────────
CLI entry point.  Run as:
    python -m simulate_rl [options]
    python simulate_rl.py [options]   (via __init__.py shim)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pedagogy_selector import SEEDING_QUESTIONS
from rl_metrics import LEARNING_TYPES
from simulate_rl.profiles import PROFILES, RADAR_CLOCKWISE, G, B, M, X
from simulate_rl.runners import (
    run_simulation,
    run_fixed_lt_simulation,
    run_story_simulation,
    run_per_lt_simulation,
    run_cross_lt_simulation,
    run_correlation_simulation,
    run_single_line_simulation,
)
from simulate_rl.plots import (
    HAS_PLOT,
    make_plots, make_qvalue_plots, make_lt_change_plot,
    make_phase_bar_plot, make_mpe_plots,
    make_per_lt_plots, make_cross_lt_plots,
    make_fixed_lt_plots, make_story_plots,
    make_correlation_plots, make_single_line_plot,
)

CATEGORY_CHOICES = ["SiKecil", "Siaga", "Penggalang", "Penegak"]


def _export_json(steps, out_dir: str, tag: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"steps_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [export] {path}  ({len(steps)} steps)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PLS RL Session Simulator v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (mutually exclusive, first match wins):
  (default)        Normal multi-session simulation
  --fixed-lt LT    Lock seeding LT and run free after seeding
  --per-lt N       Isolated N-question benchmark per LT
  --cross-lt N     8×8 student-type × question-LT matrix
  --story          3-phase narrative (assigned → explore → converge)
  --correlation    PAR/TAR positive-correlation demo
  --single-line    One student, continuous coloured reward line
        """,
    )

    # ── Common ────────────────────────────────────────────────────────────────
    parser.add_argument("--sessions",         type=int,  default=1)
    parser.add_argument("--questions",        type=int,  default=15)
    parser.add_argument("--profile",          type=str,  default="random",
                        choices=sorted(PROFILES.keys()))
    parser.add_argument("--all-profiles",     action="store_true")
    parser.add_argument("--category",         type=str,  default="Penggalang",
                        choices=CATEGORY_CHOICES)
    parser.add_argument("--seed",             type=int,  default=None)
    parser.add_argument("--quiet",            action="store_true")
    parser.add_argument("--no-plots",         action="store_true")
    parser.add_argument("--out",              type=str,  default="plots")
    parser.add_argument("--export",           action="store_true")

    # ── Normal mode ───────────────────────────────────────────────────────────
    parser.add_argument("--initial-epsilon",  type=float, default=0.20)
    parser.add_argument("--isolated",         action="store_true")
    parser.add_argument("--seeding-lt",       type=str,  default=None,
                        choices=LEARNING_TYPES)
    parser.add_argument("--ai-sim",           action="store_true")
    parser.add_argument("--api-key",          type=str,  default="")

    # ── Fixed-LT ──────────────────────────────────────────────────────────────
    parser.add_argument("--fixed-lt",         type=str,  default=None,
                        choices=LEARNING_TYPES)
    parser.add_argument("--seed-questions",   type=int,  default=SEEDING_QUESTIONS)

    # ── Per-LT ────────────────────────────────────────────────────────────────
    parser.add_argument("--per-lt",           type=int,  default=None, metavar="N")

    # ── Cross-LT ──────────────────────────────────────────────────────────────
    parser.add_argument("--cross-lt",         type=int,  default=None, metavar="N")

    # ── Story ─────────────────────────────────────────────────────────────────
    parser.add_argument("--story",            action="store_true")
    parser.add_argument("--assigned-lt",      type=str,  default="PAI",
                        choices=LEARNING_TYPES)
    parser.add_argument("--true-lt",          type=str,  default="TAR",
                        choices=LEARNING_TYPES)
    parser.add_argument("--seed-sessions",    type=int,  default=2)
    parser.add_argument("--explore-sessions", type=int,  default=2)
    parser.add_argument("--converge-sessions",type=int,  default=3)

    # ── Correlation ───────────────────────────────────────────────────────────
    parser.add_argument("--correlation",      action="store_true",
                        help="PAR/TAR positive-correlation simulation")
    parser.add_argument("--corr-questions",   type=int,  default=40, metavar="N")

    # ── Single-line ───────────────────────────────────────────────────────────
    parser.add_argument("--single-line",      action="store_true",
                        help="One continuous reward line coloured by LT")

    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    verbose = not args.quiet

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if args.ai_sim and not api_key:
        print(f"{G}[warn] No API key — ai-sim disabled.{X}")
        args.ai_sim = False

    print(f"\n{B}{'='*70}{X}")
    print(f"{B}  PLS RL Session Simulator  v5{X}")
    print(f"{B}  Seeding: {SEEDING_QUESTIONS} Q  |  8 LTs  |  epsilon-greedy bandit{X}")
    print(f"{B}{'='*70}{X}")

    # ── SINGLE-LINE ───────────────────────────────────────────────────────────
    if args.single_line:
        prof = args.profile if args.profile != "random" else "par_tar_student"
        print(f"  Mode: Single-line  |  profile={prof}  "
              f"|  sessions={args.sessions}  |  Q/session={args.questions}")
        steps = run_single_line_simulation(
            profile_name=prof, n_questions=args.questions,
            n_sessions=args.sessions, category=args.category,
            seed=args.seed if args.seed is not None else 42,
            verbose=verbose,
        )
        if not args.no_plots:
            print(f"\n{B}  Generating single-line plot…{X}")
            make_single_line_plot(steps, out_dir=args.out, profile=prof)
        if args.export:
            _export_json(steps, args.out, f"singleline_{prof}")

    # ── CORRELATION ───────────────────────────────────────────────────────────
    elif args.correlation:
        print(f"  Mode: Correlation  |  sessions={args.sessions}  "
              f"|  Q/session={args.corr_questions}  |  seed={args.seed or 42}")
        result = run_correlation_simulation(
            n_questions=args.corr_questions, n_sessions=args.sessions,
            category=args.category,
            seed=args.seed if args.seed is not None else 42,
            verbose=verbose,
        )
        if not args.no_plots:
            print(f"\n{B}  Generating correlation plots…{X}")
            make_correlation_plots(result, out_dir=args.out)
        if args.export:
            _export_json(result["steps"], args.out, "correlation")

    # ── FIXED-LT ──────────────────────────────────────────────────────────────
    elif args.fixed_lt:
        seed_q = min(args.seed_questions, args.questions - 1)
        print(f"  Mode: Fixed-LT  |  LT={args.fixed_lt}  "
              f"|  Seed={seed_q}Q  |  Sessions={args.sessions}")
        steps, reg = run_fixed_lt_simulation(
            fixed_lt=args.fixed_lt, profile_name=args.profile,
            n_sessions=args.sessions, n_questions=args.questions,
            seed_questions=seed_q, category=args.category,
            seed=args.seed, ai_sim=args.ai_sim, api_key=api_key, verbose=verbose,
        )
        if not args.no_plots:
            make_fixed_lt_plots(steps, args.fixed_lt, args.profile, args.out)
        if args.export:
            _export_json(steps, args.out, f"fixed_{args.fixed_lt}_{args.profile}")

    # ── PER-LT ────────────────────────────────────────────────────────────────
    elif args.per_lt:
        print(f"  Mode: Per-LT  |  {args.per_lt} Q each × 8 LTs  "
              f"|  profile={args.profile}")
        results = run_per_lt_simulation(
            n_questions_per_lt=args.per_lt, profile_name=args.profile,
            category=args.category, seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            make_per_lt_plots(results, args.profile, args.out)
            flat = [s for steps in results.values() for s in steps]
            make_mpe_plots(flat, args.out, profile=args.profile)
        if args.export:
            _export_json([s for v in results.values() for s in v],
                         args.out, f"per_lt_{args.profile}")

    # ── CROSS-LT ──────────────────────────────────────────────────────────────
    elif args.cross_lt:
        print(f"  Mode: Cross-LT  |  {args.cross_lt} Q/cell  "
              f"|  8×8={args.cross_lt*64} total Q")
        result = run_cross_lt_simulation(
            n_questions=args.cross_lt, category=args.category,
            seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            make_cross_lt_plots(result, args.out)

    # ── STORY ─────────────────────────────────────────────────────────────────
    elif args.story:
        profile = args.profile if args.profile != "random" else "pai_misfit"
        n_total = args.seed_sessions + args.explore_sessions + args.converge_sessions
        print(f"  Mode: Story  |  assigned={args.assigned_lt}  true={args.true_lt}  "
              f"|  sessions={n_total}×{args.questions}Q")
        steps, _ = run_story_simulation(
            assigned_lt=args.assigned_lt, true_lt=args.true_lt,
            profile_name=profile,
            n_seed_sessions=args.seed_sessions,
            n_explore_sessions=args.explore_sessions,
            n_converge_sessions=args.converge_sessions,
            n_questions=args.questions, seed_questions=args.seed_questions,
            category=args.category, seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            make_story_plots(steps, args.assigned_lt, args.true_lt, profile, args.out)
        if args.export:
            _export_json(steps, args.out, f"story_{args.assigned_lt}_{args.true_lt}")

    # ── NORMAL ────────────────────────────────────────────────────────────────
    else:
        profiles_to_run = list(PROFILES.keys()) if args.all_profiles else [args.profile]
        print(f"  Mode: Normal  |  sessions={args.sessions}  "
              f"|  Q/session={args.questions}  |  profiles={profiles_to_run}")

        all_combined = []
        for profile_name in profiles_to_run:
            prof = PROFILES[profile_name]
            print(f"\n{M}{B}> Profile: {prof['label']} — {prof['desc']}{X}")
            steps, reg = run_simulation(
                profile_name=profile_name, n_sessions=args.sessions,
                n_questions=args.questions, category=args.category,
                seed=args.seed, ai_sim=args.ai_sim, api_key=api_key,
                verbose=verbose, initial_epsilon=args.initial_epsilon,
                isolated=args.isolated, seeding_lt=args.seeding_lt,
            )
            all_combined.extend(steps)
            if not args.no_plots:
                print(f"\n{B}  Generating plots…{X}")
                make_plots(steps,          out_dir=args.out, profile=profile_name)
                make_qvalue_plots(steps,   out_dir=args.out, profile=profile_name, registry=reg)
                make_lt_change_plot(steps, out_dir=args.out, profile=profile_name)
                make_phase_bar_plot(steps, out_dir=args.out, profile=profile_name)
                make_mpe_plots(steps,      out_dir=args.out, profile=profile_name)
            if args.export:
                _export_json(steps, args.out, profile_name)

        if args.all_profiles and not args.no_plots and len(profiles_to_run) > 1:
            make_plots(all_combined, out_dir=args.out, profile="all_profiles")
            if args.export:
                _export_json(all_combined, args.out, "all_profiles")

    print(f"\n{B}{'='*70}{X}")
    print(f"{G}{B}  Simulation complete.{X}")
    if not args.no_plots and HAS_PLOT:
        print(f"  Plots saved → {os.path.abspath(args.out)}/")
    print(f"{B}{'='*70}{X}\n")


if __name__ == "__main__":
    main()
