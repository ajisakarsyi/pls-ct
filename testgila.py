"""
test_8students.py — 8-Student Cross-LT Benchmark
=================================================
Two-part experiment:

  PART 1 — HOME LT
    Each student is tested exclusively on their own strongest LT.
    Shows the reward baseline when student ↔ content are well-matched.

  PART 2 — CROSS-LT  (8 × 8 grid)
    Every student is tested on every LT (including their own).
    Reveals how much reward drops when content is mismatched,
    and whether any LTs happen to transfer well across student types.

Output
------
  plots/8s_1_home_lt_rewards.png          — bar chart: home-LT avg reward per student
  plots/8s_2_home_lt_mpe.png              — M/P/E profile on home LT
  plots/8s_3_cross_heatmap.png            — 8×8 avg-reward matrix
  plots/8s_4_home_vs_best_away.png        — home vs best-away scatter
  plots/8s_5_advantage_heatmap.png        — Δreward (home − other) per cell
  plots/8s_6_per_student_radar.png        — radar chart per student across all LTs
  plots/8s_7_correlation_matrix.png       — Pearson-r between LT reward vectors

Usage
-----
  python test_8students.py                     # default 30 Q/LT
  python test_8students.py --questions 50      # more questions, smoother estimates
  python test_8students.py --seed 42
  python test_8students.py --quiet             # skip per-step terminal output
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pedagogy_selector import SessionRegistry, LEARNING_TYPES
from simulate_rl import StudentSimulator, PROFILES

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found — pip install matplotlib")

OUT_DIR = "plots"

MPL_COLOURS = {
    "PAR": "#2ecc71", "TAR": "#f39c12", "PAI": "#3498db", "TAI": "#9b59b6",
    "TGR": "#1abc9c", "TGI": "#e74c3c", "PGR": "#27ae60", "PGI": "#8e44ad",
}
STUDENT_MARKERS = ["o","s","D","^","v","<",">","P"]


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def simulate_lt(student_lt: str, question_lt: str, n_questions: int,
                verbose: bool = False) -> dict:
    """
    One student (profiled as student_{student_lt}) answers n_questions
    all in question_lt. Returns step-level stats.
    """
    profile_name = f"student_{student_lt.lower()}"
    registry     = SessionRegistry()
    sid          = f"{student_lt}_on_{question_lt}"
    agent        = registry.get_agent(sid, seeding_lt=student_lt)
    student      = StudentSimulator(profile_name)

    rewards, corrects, masteries, perfs, engs = [], [], [], [], []

    for _ in range(n_questions):
        is_correct, n_attempt, t_ans = student.answer(question_lt)
        mc   = f"{agent.mastery_levels[question_lt]}{question_lt}"
        step = agent.record_response(question_lt, is_correct, n_attempt, t_ans, mc)

        rewards.append(step["reward"])
        corrects.append(1 if is_correct else 0)
        masteries.append(step["mastery_score"])
        perfs.append(step["performance"])
        engs.append(step["engagement"])

        if verbose:
            ok = "✓" if is_correct else "✗"
            print(f"  [{student_lt}→{question_lt}] {ok} R={step['reward']:+.4f} "
                  f"M={step['mastery_score']:.2f} P={step['performance']:.2f}")

    return {
        "student_lt":   student_lt,
        "question_lt":  question_lt,
        "n_questions":  n_questions,
        "rewards":      rewards,
        "avg_reward":   float(np.mean(rewards)),
        "sum_reward":   float(np.sum(rewards)),
        "correct_pct":  float(np.mean(corrects)) * 100,
        "avg_mastery":  float(np.mean(masteries)),
        "avg_perf":     float(np.mean(perfs)),
        "avg_eng":      float(np.mean(engs)),
        "final_mastery": masteries[-1],
    }


def run_all(n_questions: int, verbose: bool, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── PART 1: each student on their home LT ──────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  PART 1 — HOME LT  ({n_questions} Q each)")
    print(f"{'═'*60}")
    home_results = {}
    for lt in LEARNING_TYPES:
        print(f"  Student {lt} on home LT {lt} ...", end=" ", flush=True)
        r = simulate_lt(lt, lt, n_questions, verbose=verbose)
        home_results[lt] = r
        print(f"avg_reward={r['avg_reward']:+.4f}  "
              f"correct={r['correct_pct']:.0f}%  "
              f"mastery={r['avg_mastery']:.2f}")

    # ── PART 2: every student on every LT ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  PART 2 — CROSS-LT  (8 × 8 = {8*8} combos × {n_questions} Q)")
    print(f"{'═'*60}")
    cross_results = {slt: {} for slt in LEARNING_TYPES}
    for s_idx, student_lt in enumerate(LEARNING_TYPES):
        print(f"\n  Student {student_lt}  ({s_idx+1}/8):")
        for q_lt in LEARNING_TYPES:
            r = simulate_lt(student_lt, q_lt, n_questions, verbose=verbose)
            cross_results[student_lt][q_lt] = r
            tag = " ← HOME" if q_lt == student_lt else ""
            print(f"    vs {q_lt}: avg={r['avg_reward']:+.4f}  "
                  f"corr={r['correct_pct']:.0f}%{tag}")

    return home_results, cross_results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(home_results, cross_results):
    print(f"\n{'═'*70}")
    print("  FINAL SUMMARY")
    print(f"{'═'*70}")

    # Build matrix
    matrix = np.array([
        [cross_results[s][q]["avg_reward"] for q in LEARNING_TYPES]
        for s in LEARNING_TYPES
    ])

    print(f"\n  {'Student':>8}  {'Home':>7}  {'Best Away':>9}  "
          f"{'Worst Away':>10}  {'Home Adv':>9}  {'Best Away LT':>12}")
    print(f"  {'─'*65}")

    for i, slt in enumerate(LEARNING_TYPES):
        home_r   = cross_results[slt][slt]["avg_reward"]
        away_rs  = {lt: cross_results[slt][lt]["avg_reward"]
                    for lt in LEARNING_TYPES if lt != slt}
        best_lt  = max(away_rs, key=away_rs.get)
        worst_lt = min(away_rs, key=away_rs.get)
        best_r   = away_rs[best_lt]
        worst_r  = away_rs[worst_lt]
        adv      = home_r - np.mean(list(away_rs.values()))
        print(f"  {slt:>8}  {home_r:>+7.4f}  {best_r:>+9.4f}  "
              f"{worst_r:>+10.4f}  {adv:>+9.4f}  {best_lt:>12}")

    print(f"\n  Cross-LT avg reward matrix (rows=student, cols=question LT):")
    print(f"  {'':>6}  " + "  ".join(f"{lt:>7}" for lt in LEARNING_TYPES))
    print(f"  {'─'*75}")
    for i, slt in enumerate(LEARNING_TYPES):
        row = "  ".join(
            f"{'*' if slt==qlt else ' '}{matrix[i, j]:>6.3f}"
            for j, qlt in enumerate(LEARNING_TYPES)
        )
        print(f"  {slt:>6}  {row}")
    print(f"  (* = home LT diagonal)")

    # Correlation matrix
    corr = np.corrcoef(matrix.T)
    print(f"\n  Top 3 most correlated LT pairs (by student response pattern):")
    pairs = []
    for i in range(8):
        for j in range(i+1, 8):
            pairs.append((corr[i,j], LEARNING_TYPES[i], LEARNING_TYPES[j]))
    pairs.sort(reverse=True)
    for r, a, b in pairs[:3]:
        print(f"    {a} ↔ {b}  r={r:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_all_plots(home_results, cross_results, n_questions):
    if not HAS_PLOT:
        print("[skip] matplotlib not available"); return
    os.makedirs(OUT_DIR, exist_ok=True)

    matrix = np.array([
        [cross_results[s][q]["avg_reward"] for q in LEARNING_TYPES]
        for s in LEARNING_TYPES
    ])
    home_avgs  = [cross_results[s][s]["avg_reward"] for s in LEARNING_TYPES]
    away_avgs  = [
        np.mean([cross_results[s][q]["avg_reward"]
                 for q in LEARNING_TYPES if q != s])
        for s in LEARNING_TYPES
    ]
    best_aways = [
        max(cross_results[s][q]["avg_reward"]
            for q in LEARNING_TYPES if q != s)
        for s in LEARNING_TYPES
    ]
    best_away_lts = [
        max((q for q in LEARNING_TYPES if q != s),
            key=lambda q: cross_results[s][q]["avg_reward"])
        for s in LEARNING_TYPES
    ]

    # ── 1. HOME LT BAR CHART ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x    = np.arange(len(LEARNING_TYPES))
    cols = [MPL_COLOURS[lt] for lt in LEARNING_TYPES]
    bars = ax.bar(x, home_avgs, color=cols, alpha=0.88,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot(x, away_avgs, color="dimgrey", lw=2.0, ls="--",
            marker="o", ms=6, label="avg reward on other 7 LTs", zorder=4)
    for b, v in zip(bars, home_avgs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.003,
                f"{v:+.3f}", ha="center", fontsize=8.5, fontweight="bold",
                color=b.get_facecolor())
    ax.axhline(np.mean(home_avgs), color="black", lw=1.0, ls=":",
               alpha=0.55, label=f"grand mean (home) = {np.mean(home_avgs):+.3f}")
    ax.set_xticks(x); ax.set_xticklabels(LEARNING_TYPES, fontsize=10)
    for tick, lt in zip(ax.get_xticklabels(), LEARNING_TYPES):
        tick.set_color(MPL_COLOURS[lt]); tick.set_fontweight("bold")
    ax.set_ylabel("Average Reward"); ax.set_xlabel("Student Type (= Home LT)")
    ax.set_title(f"Part 1 — Home LT Avg Reward per Student  ({n_questions} Q each)\n"
                 "Coloured bars = home LT reward   Grey line = avg on all other LTs",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "8s_1_home_lt_rewards")

    # ── 2. HOME LT  M / P / E PROFILE ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, key, label, col in zip(
            axes,
            ["avg_mastery", "avg_perf", "avg_eng"],
            ["Mastery (M)", "Performance (P)", "Engagement (E)"],
            ["#e74c3c", "#3498db", "#2ecc71"]):
        vals = [home_results[lt][key] for lt in LEARNING_TYPES]
        bars = ax.bar(LEARNING_TYPES, vals,
                      color=[MPL_COLOURS[lt] for lt in LEARNING_TYPES],
                      alpha=0.82, edgecolor="white")
        ax.axhline(np.mean(vals), color=col, lw=1.5, ls="--", alpha=0.7,
                   label=f"mean={np.mean(vals):.2f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8, color=b.get_facecolor())
        ax.set_title(label, fontsize=11, color=col, fontweight="bold")
        ax.set_ylim(0, 1.15); ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        for tick, lt in zip(ax.get_xticklabels(), LEARNING_TYPES):
            tick.set_color(MPL_COLOURS[lt])
    fig.suptitle(f"Part 1 — M / P / E on Home LT  ({n_questions} Q each)", fontsize=12)
    plt.tight_layout()
    _save(fig, "8s_2_home_lt_mpe")

    # ── 3. CROSS-LT REWARD HEATMAP ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = matrix.min(), matrix.max()
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Avg Reward")
    ax.set_xticks(range(8)); ax.set_xticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_yticks(range(8)); ax.set_yticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_xlabel("Question LT (what they were tested on)", fontsize=11)
    ax.set_ylabel("Student Type (natural LT)", fontsize=11)
    ax.set_title(f"Part 2 — Cross-LT Reward Matrix  ({n_questions} Q/cell)\n"
                 "Diagonal = home LT  |  Blue border = home cell per student",
                 fontsize=12)
    for i in range(8):
        for j in range(8):
            v = matrix[i, j]
            dark = v > (vmin + (vmax - vmin) * 0.65)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else "black",
                    fontweight="bold" if i == j else "normal")
        # highlight diagonal
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#2980b9", lw=2.5, zorder=5))
    plt.tight_layout()
    _save(fig, "8s_3_cross_heatmap")

    # ── 4. HOME vs BEST AWAY scatter ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, slt in enumerate(LEARNING_TYPES):
        ax.scatter(home_avgs[i], best_aways[i],
                   color=MPL_COLOURS[slt], s=160, zorder=5,
                   marker=STUDENT_MARKERS[i], edgecolors="white", linewidths=1.2,
                   label=f"{slt} (best away: {best_away_lts[i]})")
        ax.annotate(slt, xy=(home_avgs[i], best_aways[i]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=9, color=MPL_COLOURS[slt], fontweight="bold")

    lim = [min(home_avgs + best_aways) - 0.01, max(home_avgs + best_aways) + 0.01]
    ax.plot(lim, lim, color="grey", lw=1.2, ls="--", alpha=0.6, label="home = best away")
    ax.fill_between(lim, lim, [lim[1]]*2, alpha=0.05, color="green",
                    label="best away > home")
    ax.fill_between(lim, [lim[0]]*2, lim, alpha=0.05, color="orange",
                    label="home > best away")
    ax.set_xlabel("Home LT Avg Reward", fontsize=11)
    ax.set_ylabel("Best Away-LT Avg Reward", fontsize=11)
    ax.set_title("Part 2 — Home LT vs Best Away-LT Reward\n"
                 "Points above dashed line = student does better on a different LT",
                 fontsize=12)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, "8s_4_home_vs_best_away")

    # ── 5. ADVANTAGE HEATMAP  (home − other per cell) ──────────────────────
    home_col = np.array(home_avgs)[:, np.newaxis]
    adv_mat  = home_col - matrix   # positive = home is better than this cell

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(adv_mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=-adv_mat.max(), vmax=adv_mat.max())
    plt.colorbar(im, ax=ax, label="Home reward − Cell reward  (green = cell wins)")
    ax.set_xticks(range(8)); ax.set_xticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_yticks(range(8)); ax.set_yticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_xlabel("Question LT"); ax.set_ylabel("Student Type")
    ax.set_title("Part 2 — Home LT Advantage per Cell\n"
                 "Red = home LT better  |  Green = this LT beats home  |  "
                 "Diagonal = 0 (same)", fontsize=12)
    for i in range(8):
        for j in range(8):
            v = adv_mat[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > adv_mat.max()*0.7 else "black",
                    fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    _save(fig, "8s_5_advantage_heatmap")

    # ── 6. PER-STUDENT RADAR / LINE CHART ────────────────────────────────────
    # Spider-style using a polar axes grid — one per student
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40)
    angles = np.linspace(0, 2 * np.pi, len(LEARNING_TYPES), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for idx, slt in enumerate(LEARNING_TYPES):
        ax = fig.add_subplot(gs[idx // 4, idx % 4], polar=True)
        vals = [cross_results[slt][q]["avg_reward"] for q in LEARNING_TYPES]
        vals += vals[:1]

        ax.plot(angles, vals, color=MPL_COLOURS[slt], lw=2.2, zorder=4)
        ax.fill(angles, vals, color=MPL_COLOURS[slt], alpha=0.25, zorder=3)

        # Mark home point
        home_angle = angles[LEARNING_TYPES.index(slt)]
        home_val   = cross_results[slt][slt]["avg_reward"]
        ax.scatter([home_angle], [home_val], color=MPL_COLOURS[slt],
                   s=80, zorder=6, edgecolors="white", linewidths=1.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(LEARNING_TYPES, fontsize=8)
        ax.set_ylim(min(0, min(vals)) - 0.005, max(vals) + 0.01)
        ax.set_title(f"Student {slt}", fontsize=11,
                     color=MPL_COLOURS[slt], fontweight="bold", pad=12)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#fafafa")

    fig.suptitle(f"Part 2 — Per-Student Reward Radar across all 8 LTs  ({n_questions} Q/LT)\n"
                 "Dot = home LT  |  Polygon = reward profile across all question types",
                 fontsize=13, y=1.02)
    plt.savefig(os.path.join(OUT_DIR, "8s_6_per_student_radar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] 8s_6_per_student_radar.png")

    # ── 7. PEARSON-R CORRELATION MATRIX ──────────────────────────────────────
    corr = np.corrcoef(matrix.T)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r  (how similarly students responded to these two LTs)")
    ax.set_xticks(range(8)); ax.set_xticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_yticks(range(8)); ax.set_yticklabels(LEARNING_TYPES, fontsize=11)
    ax.set_title("Part 2 — LT Correlation Matrix  (Pearson r)\n"
                 "High r = similar reward patterns across student types",
                 fontsize=12)
    for i in range(8):
        for j in range(8):
            r = corr[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(r) > 0.75 else "black",
                    fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    _save(fig, "8s_7_correlation_matrix")

    print(f"\n  All 7 plots saved to ./{OUT_DIR}/")


def _save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="8-Student Cross-LT Benchmark")
    parser.add_argument("--questions", type=int, default=30,
                        help="Questions per (student, LT) combo (default: 30)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress per-step output")
    parser.add_argument("--no-plots",  action="store_true")
    parser.add_argument("--out",       type=str, default="plots")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = args.out

    print(f"\n{'═'*70}")
    print(f"  8-Student Cross-LT Benchmark")
    print(f"  {len(LEARNING_TYPES)} students × {len(LEARNING_TYPES)} LTs × {args.questions} Q "
          f"= {len(LEARNING_TYPES)**2 * args.questions} total questions")
    print(f"  Seed: {args.seed}")
    print(f"{'═'*70}")

    home_results, cross_results = run_all(
        n_questions=args.questions,
        verbose=not args.quiet,
        seed=args.seed,
    )

    print_summary(home_results, cross_results)

    if not args.no_plots:
        print(f"\n{'─'*50}")
        print("  Generating plots…")
        make_all_plots(home_results, cross_results, args.questions)

    print(f"\n{'═'*70}")
    print(f"  Done.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()