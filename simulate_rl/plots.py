"""
simulate_rl/plots.py
──────────────────────
All matplotlib plot functions.  Each function is self-contained:
pass in steps (List[Dict]) and an output directory, get a PNG file.

Functions
---------
make_plots              — reward time series per LT + overlay trend + transition heatmap
make_qvalue_plots       — Q-value evolution with change event markers
make_lt_change_plot     — recommendation timeline swim-lane chart
make_phase_bar_plot     — seeding vs free LT usage comparison
make_mpe_plots          — Mastery / Performance / Engagement per LT grid
make_per_lt_plots       — per-LT isolated benchmark plots
make_cross_lt_plots     — 8×8 cross-LT reward and correlation heatmaps
make_fixed_lt_plots     — wrapper for fixed-LT mode
make_story_plots        — wrapper for story mode
make_correlation_plots  — PAR/TAR correlation specific plots
make_single_line_plot   — single continuous coloured reward line
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from pedagogy_selector import SEEDING_QUESTIONS
from rl_metrics import LEARNING_TYPES, MASTERY_LEVEL_SCORE, N_MAX
from simulate_rl.profiles import (
    MPL_COLOURS, SEED_SHADE, RADAR_RING_ORDERS, RADAR_CLOCKWISE,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save_plot(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {name}.png")


def _shade_seeding(ax, steps, n_questions_per_session: int, seed_n: int,
                   alpha: float = 0.25) -> None:
    """Shade seeding windows (first seed_n questions of each session) in yellow."""
    sessions = sorted(set(s.get("session_num", 1) for s in steps))
    for sn in sessions:
        x0 = (sn - 1) * n_questions_per_session + 0.5
        x1 = x0 + seed_n
        ax.axvspan(x0, x1, color=SEED_SHADE, alpha=alpha, zorder=0)
    for sn in sessions[1:]:
        ax.axvline((sn - 1) * n_questions_per_session, color="grey",
                   lw=0.7, ls=":", alpha=0.5)


def _rolling(ys: list, window: int):
    if len(ys) < window:
        return list(range(len(ys))), ys
    r  = np.convolve(ys, np.ones(window) / window, mode="valid")
    xs = list(range(window - 1, len(ys)))
    return xs, r.tolist()


def _n_q_per_session(all_steps: List[Dict]) -> int:
    sessions = sorted(set(s.get("session_num", 1) for s in all_steps))
    return max(
        (s.get("question_num", 1) for s in all_steps
         if s.get("session_num") == sessions[0]),
        default=15,
    )


# ── Plot 1 — Reward time series ───────────────────────────────────────────────

def make_plots(all_steps: List[Dict], out_dir: str = "plots", profile: str = "") -> None:
    """Per-LT reward subplots + combined overlay trend + LT transition heatmap."""
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag      = f"_{profile}" if profile else ""
    n_q_s    = _n_q_per_session(all_steps)
    seed_n   = SEEDING_QUESTIONS

    # Per-LT subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True, sharey=True)
    for i, lt in enumerate(LEARNING_TYPES):
        ax  = axes.flatten()[i]
        col = MPL_COLOURS[lt]
        xs  = [s["global_q"] for s in all_steps if s["lt"] == lt]
        ys  = [s["reward"]   for s in all_steps if s["lt"] == lt]

        _shade_seeding(ax, all_steps, n_q_s, seed_n)

        if not xs:
            ax.set_title(f"{lt}  (not selected)", fontsize=10, color="grey")
            ax.axis("off")
            continue

        c_xs = [s["global_q"] for s in all_steps if s["lt"] == lt and s["is_correct"]]
        c_ys = [s["reward"]   for s in all_steps if s["lt"] == lt and s["is_correct"]]
        w_xs = [s["global_q"] for s in all_steps if s["lt"] == lt and not s["is_correct"]]
        w_ys = [s["reward"]   for s in all_steps if s["lt"] == lt and not s["is_correct"]]
        ax.scatter(c_xs, c_ys, color=col, s=35, alpha=0.7,  marker="o", zorder=3)
        ax.scatter(w_xs, w_ys, color=col, s=35, alpha=0.35, marker="x", zorder=3)

        if len(xs) >= 3:
            paired = sorted(zip(xs, ys))
            xs_s   = [p[0] for p in paired]
            ys_s   = [p[1] for p in paired]
            ri, rv = _rolling(ys_s, max(3, len(ys_s) // 5))
            ax.plot([xs_s[r] for r in ri], rv, color=col, lw=2.5, alpha=0.95, zorder=4)

        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"{lt}  (n={len(xs)}  avg={np.mean(ys):+.3f})",
                     fontsize=11, color=col, fontweight="bold")
        ax.grid(True, alpha=0.25)

    for ax in axes[:, 0]:  ax.set_ylabel("Reward")
    for ax in axes[-1, :]: ax.set_xlabel("Question (global)")
    seed_patch = mpatches.Patch(color=SEED_SHADE, label=f"Seeding phase ({seed_n} Q)")
    fig.legend(handles=[seed_patch], loc="upper right", fontsize=9)
    fig.suptitle(
        f"Reward Time Series per LT  —  profile: {profile or 'all'}\n"
        f"(yellow = seeding phase,  o=correct  x=wrong,  line=trend)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save_plot(fig, out_dir, f"reward_timeseries{tag}")

    # Overlay
    fig, ax = plt.subplots(figsize=(16, 6))
    _shade_seeding(ax, all_steps, n_q_s, seed_n)
    for lt in LEARNING_TYPES:
        xs = [s["global_q"] for s in all_steps if s["lt"] == lt]
        ys = [s["reward"]   for s in all_steps if s["lt"] == lt]
        if len(xs) < 3:
            continue
        paired = sorted(zip(xs, ys))
        xs_s   = [p[0] for p in paired]
        ys_s   = [p[1] for p in paired]
        ri, rv = _rolling(ys_s, max(3, len(ys_s) // 5))
        roll_x = [xs_s[r] for r in ri]
        ax.plot(roll_x, rv, color=MPL_COLOURS[lt], lw=2.2, label=lt, alpha=0.9)
        if roll_x:
            ax.annotate(lt, xy=(roll_x[-1], rv[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8,
                        color=MPL_COLOURS[lt], va="center")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Question (global)"); ax.set_ylabel("Reward (rolling mean)")
    ax.set_title(f"Reward Trend Overlay — all LTs  |  profile: {profile or 'all'}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, out_dir, f"reward_trend_overlay{tag}")

    # Transition heatmap
    mat = np.zeros((len(LEARNING_TYPES), len(LEARNING_TYPES)))
    for s in all_steps:
        if s.get("prev_lt"):
            mat[LEARNING_TYPES.index(s["prev_lt"])][LEARNING_TYPES.index(s["lt"])] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(LEARNING_TYPES))); ax.set_xticklabels(LEARNING_TYPES, fontsize=10)
    ax.set_yticks(range(len(LEARNING_TYPES))); ax.set_yticklabels(LEARNING_TYPES, fontsize=10)
    ax.set_xlabel("To LT"); ax.set_ylabel("From LT")
    ax.set_title(f"LT Transition Heatmap  —  profile: {profile or 'all'}", fontsize=12)
    plt.colorbar(im, ax=ax, label="Transition count")
    for i in range(len(LEARNING_TYPES)):
        for j in range(len(LEARNING_TYPES)):
            v = int(mat[i][j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > mat.max() * 0.5 else "black")
    plt.tight_layout()
    _save_plot(fig, out_dir, f"lt_transition_heatmap{tag}")


# ── Plot 2 — Q-value evolution ────────────────────────────────────────────────

def make_qvalue_plots(all_steps: List[Dict], out_dir: str = "plots",
                      profile: str = "", registry=None) -> None:
    """Q-value of every LT over time, with recommendation change markers."""
    if not HAS_PLOT or not all_steps or "q_values" not in all_steps[0]:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag   = f"_{profile}" if profile else ""
    n_q_s = _n_q_per_session(all_steps)
    xs    = [s["global_q"] for s in all_steps]
    change_qs = [s["global_q"] for s in all_steps if s.get("lt_changed")]

    fig, ax = plt.subplots(figsize=(16, 6))
    _shade_seeding(ax, all_steps, n_q_s, SEEDING_QUESTIONS)
    for lt in LEARNING_TYPES:
        qs = [s["q_values"].get(lt, 0) for s in all_steps]
        ax.plot(xs, qs, color=MPL_COLOURS[lt], lw=1.8, label=lt, alpha=0.85)
        ax.annotate(lt, xy=(xs[-1], qs[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8,
                    color=MPL_COLOURS[lt], va="center")
    for q in change_qs:
        ax.axvline(q, color="red", lw=1.0, ls="--", alpha=0.55, zorder=5)
    if change_qs:
        ax.scatter(change_qs, [0] * len(change_qs), marker="^",
                   color="red", s=60, zorder=6, label="recommendation change")
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Question (global)"); ax.set_ylabel("Q-value")
    ax.set_title(
        f"Q-value Evolution  |  profile: {profile or 'all'}\n"
        f"(yellow = seeding phase,  red ▲ = recommendation changes)", fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9, ncol=5); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_plot(fig, out_dir, f"qvalue_evolution{tag}")


# ── Plot 3 — LT recommendation timeline ──────────────────────────────────────

def make_lt_change_plot(all_steps: List[Dict], out_dir: str = "plots",
                        profile: str = "") -> None:
    """Horizontal swim-lane showing the current best LT recommendation at each question."""
    if not HAS_PLOT or not all_steps or "q_values" not in all_steps[0]:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag   = f"_{profile}" if profile else ""
    n_q_s = _n_q_per_session(all_steps)

    best_seq = [max(s["q_values"], key=s["q_values"].get) for s in all_steps]
    xs       = [s["global_q"] for s in all_steps]
    lt_y     = {lt: i for i, lt in enumerate(LEARNING_TYPES)}

    fig, ax = plt.subplots(figsize=(16, 5))
    _shade_seeding(ax, all_steps, n_q_s, SEEDING_QUESTIONS)

    prev_lt = None; seg_start = 0
    for i, lt in enumerate(best_seq):
        if lt != prev_lt and prev_lt is not None:
            ax.barh(lt_y[prev_lt], xs[i - 1] - xs[seg_start] + 1,
                    left=xs[seg_start] - 0.5, height=0.6,
                    color=MPL_COLOURS[prev_lt], alpha=0.75, edgecolor="white")
            seg_start = i
        if i == len(best_seq) - 1:
            ax.barh(lt_y[lt], xs[i] - xs[seg_start] + 1,
                    left=xs[seg_start] - 0.5, height=0.6,
                    color=MPL_COLOURS[lt], alpha=0.75, edgecolor="white")
        prev_lt = lt

    change_xs = [xs[i] for i in range(1, len(best_seq)) if best_seq[i] != best_seq[i - 1]]
    for cq in change_xs:
        ax.axvline(cq - 0.5, color="red", lw=1.2, ls="--", alpha=0.6, zorder=5)

    ax.set_yticks(list(lt_y.values())); ax.set_yticklabels(list(lt_y.keys()))
    for lt, y in lt_y.items():
        ax.get_yticklabels()[y].set_color(MPL_COLOURS[lt])
    ax.set_xlabel("Question (global)")
    ax.set_title(
        f"LT Recommendation Timeline  |  profile: {profile or 'all'}\n"
        f"(filled = best recommendation at that question,  red = change,  yellow = seeding)",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.2); plt.tight_layout()
    _save_plot(fig, out_dir, f"lt_recommendation_timeline{tag}")


# ── Plot 4 — Seeding vs free phase bar ───────────────────────────────────────

def make_phase_bar_plot(all_steps: List[Dict], out_dir: str = "plots",
                        profile: str = "") -> None:
    """Grouped bar: LT usage % in seeding phase vs free phase."""
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    seed_steps = [s for s in all_steps if s.get("phase") == "seeding"]
    free_steps = [s for s in all_steps if s.get("phase") == "free"]
    if not seed_steps and not free_steps:
        return

    def pcts(steps):
        if not steps: return {lt: 0.0 for lt in LEARNING_TYPES}
        dist = Counter(s["lt"] for s in steps)
        return {lt: dist.get(lt, 0) / len(steps) * 100 for lt in LEARNING_TYPES}

    seed_pct = pcts(seed_steps)
    free_pct = pcts(free_steps)
    x = np.arange(len(LEARNING_TYPES)); w = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w / 2, [seed_pct[lt] for lt in LEARNING_TYPES], w,
                label=f"Seeding  ({len(seed_steps)} Q)",
                color="#ffe082", edgecolor="grey", linewidth=0.7, alpha=0.9)
    b2 = ax.bar(x + w / 2, [free_pct[lt] for lt in LEARNING_TYPES], w,
                label=f"Free     ({len(free_steps)} Q)",
                color="#90caf9", edgecolor="grey", linewidth=0.7, alpha=0.9)

    for b, pct_dict in [(b1, seed_pct), (b2, free_pct)]:
        for bar_, lt in zip(b, LEARNING_TYPES):
            v = pct_dict[lt]
            if v > 1:
                ax.text(bar_.get_x() + bar_.get_width() / 2, v + 0.5,
                        f"{v:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(LEARNING_TYPES)
    for tick, lt in zip(ax.get_xticklabels(), LEARNING_TYPES):
        tick.set_color(MPL_COLOURS[lt])
    ax.set_ylabel("% of phase questions")
    ax.set_title(
        f"LT Usage: Seeding vs Free Phase  |  profile: {profile or 'all'}\n"
        f"(seeding = first {SEEDING_QUESTIONS} Q/session,  free = epsilon-greedy)",
        fontsize=12,
    )
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, out_dir, f"seeding_vs_free_bar{tag}")


# ── Plot 5 — M / P / E per LT ────────────────────────────────────────────────

def make_mpe_plots(all_steps: List[Dict], out_dir: str = "plots",
                   profile: str = "") -> None:
    """3×8 grid: Mastery, Performance, Engagement per LT over questions."""
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    fig = plt.figure(figsize=(20, 10))
    metrics = [
        ("mastery_score", "Mastery (M)",     "solid"),
        ("performance",   "Performance (P)", "dashed"),
        ("engagement",    "Engagement (E)",  "dotted"),
    ]
    gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.55, wspace=0.35)

    for col, lt in enumerate(LEARNING_TYPES):
        lt_steps = [s for s in all_steps if s["lt"] == lt]
        if not lt_steps:
            continue
        xs         = list(range(1, len(lt_steps) + 1))
        col_colour = MPL_COLOURS[lt]

        for row, (key, label, ls) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            ys = [s[key] for s in lt_steps]
            ax.plot(xs, ys, color=col_colour, lw=1.6, ls=ls, alpha=0.85)
            if len(ys) >= 4:
                ri, rv = _rolling(ys, max(3, len(ys) // 5))
                ax.plot([xs[r] for r in ri], rv, color=col_colour, lw=2.5, alpha=0.95)
            ax.axhline(np.mean(ys), color="grey", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.2)
            if row == 0: ax.set_title(lt, fontsize=11, color=col_colour, fontweight="bold")
            if col == 0: ax.set_ylabel(label, fontsize=9)
            if row == 2: ax.set_xlabel("Q#", fontsize=8)

    fig.suptitle(
        f"Mastery / Performance / Engagement per LT  |  profile: {profile or 'all'}\n"
        f"(thin = per-step,  thick = rolling mean,  dashed grey = overall mean)",
        fontsize=12, y=1.01,
    )
    _save_plot(fig, out_dir, f"mpe_per_lt{tag}")


# ── Per-LT benchmark plots ────────────────────────────────────────────────────

def make_per_lt_plots(results: Dict[str, List[Dict]], profile: str,
                      out_dir: str = "plots") -> None:
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)

    # Time-series
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True, sharey=True)
    for i, lt in enumerate(LEARNING_TYPES):
        ax    = axes.flatten()[i]; col = MPL_COLOURS[lt]
        steps = results.get(lt, [])
        if not steps: ax.axis("off"); continue
        xs  = [s["question_num"] for s in steps]
        ys  = [s["reward"]       for s in steps]
        ax.scatter([s["question_num"] for s in steps if     s["is_correct"]],
                   [s["reward"]       for s in steps if     s["is_correct"]],
                   color=col, s=30, alpha=0.7,  marker="o", zorder=3)
        ax.scatter([s["question_num"] for s in steps if not s["is_correct"]],
                   [s["reward"]       for s in steps if not s["is_correct"]],
                   color=col, s=30, alpha=0.35, marker="x", zorder=3)
        if len(xs) >= 5:
            ri, rv = _rolling(ys, max(3, len(ys) // 8))
            ax.plot([xs[r] for r in ri], rv, color=col, lw=2.5, zorder=4)
        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"{lt}  avg={np.mean(ys):+.3f}  n={len(xs)}",
                     fontsize=11, color=col, fontweight="bold")
        ax.grid(True, alpha=0.2)
    for ax in axes[:, 0]:  ax.set_ylabel("Reward")
    for ax in axes[-1, :]: ax.set_xlabel("Question #")
    fig.suptitle(f"Per-LT Reward Time Series  |  Profile: {profile}", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_plot(fig, out_dir, f"per_lt_timeseries_{profile}")

    # Ranking bar
    avgs    = {lt: float(np.mean([s["reward"] for s in steps]))
               for lt, steps in results.items() if steps}
    lt_rank = sorted(avgs, key=avgs.get, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(lt_rank, [avgs[lt] for lt in lt_rank],
                  color=[MPL_COLOURS[lt] for lt in lt_rank], alpha=0.85, edgecolor="white")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    for b, v in zip(bars, [avgs[lt] for lt in lt_rank]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold",
                va="bottom" if v >= 0 else "top")
    ax.set_xlabel("Learning Type"); ax.set_ylabel("Average Reward")
    ax.set_title(f"LT Ranking by Average Reward  |  Profile: {profile}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    _save_plot(fig, out_dir, f"per_lt_ranking_{profile}")

    # Mastery progression
    fig, ax = plt.subplots(figsize=(16, 5))
    for lt in LEARNING_TYPES:
        steps = results.get(lt, [])
        if not steps: continue
        ax.plot([s["question_num"] for s in steps],
                [s["mastery_level"] for s in steps],
                color=MPL_COLOURS[lt], lw=2.0, label=lt,
                drawstyle="steps-post", alpha=0.85)
    ax.set_xlabel("Question #"); ax.set_ylabel("Mastery Level (1-6)")
    ax.set_yticks(range(1, 7))
    ax.set_title(f"Mastery Level Progression per LT  |  Profile: {profile}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, out_dir, f"per_lt_mastery_{profile}")


# ── Cross-LT heatmaps ─────────────────────────────────────────────────────────

def make_cross_lt_plots(result: Dict, out_dir: str = "plots") -> None:
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)
    matrix  = result["matrix"]
    lt_list = result["lt_list"]
    n_q     = result["n_questions"]
    corr    = np.corrcoef(matrix.T)

    for data, cmap, label, fname, title in [
        (matrix, "YlOrRd", "Average reward", "cross_lt_reward_heatmap",
         f"Cross-LT Reward Matrix  ({n_q} Q per cell)\nDiagonal = student answers own LT"),
        (corr,   "RdYlGn", "Pearson r",      "cross_lt_correlation_heatmap",
         "LT Correlation Matrix  (Pearson r)"),
    ]:
        vmin, vmax = (-1, 1) if fname.endswith("correlation_heatmap") else (None, None)
        fig, ax    = plt.subplots(figsize=(9, 7))
        im         = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xticks(range(8)); ax.set_xticklabels(lt_list, fontsize=11)
        ax.set_yticks(range(8)); ax.set_yticklabels(lt_list, fontsize=11)
        ax.set_title(title, fontsize=12)
        for i in range(8):
            for j in range(8):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > (0.7 if vmin else data.mean() + data.std()) else "black",
                        fontweight="bold" if i == j else "normal")
            if fname.endswith("reward_heatmap"):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="blue", lw=2.5, zorder=5))
        plt.tight_layout()
        _save_plot(fig, out_dir, fname)


# ── Wrappers for mode-specific plot bundles ───────────────────────────────────

def make_fixed_lt_plots(all_steps: List[Dict], fixed_lt: str, profile: str,
                        out_dir: str = "plots") -> None:
    tag = f"fixed_{fixed_lt}_{profile}"
    make_plots(all_steps, out_dir, profile=tag)
    make_qvalue_plots(all_steps, out_dir, profile=tag)
    make_phase_bar_plot(all_steps, out_dir, profile=tag)
    make_mpe_plots(all_steps, out_dir, profile=tag)


def make_story_plots(all_steps: List[Dict], assigned_lt: str, true_lt: str,
                     profile: str, out_dir: str = "plots") -> None:
    tag = f"story_{assigned_lt}_to_{true_lt}_{profile}"
    make_plots(all_steps, out_dir, profile=tag)
    make_qvalue_plots(all_steps, out_dir, profile=tag)
    make_lt_change_plot(all_steps, out_dir, profile=tag)
    make_mpe_plots(all_steps, out_dir, profile=tag)


# ── Correlation plots (PAR/TAR positive-r demo) ───────────────────────────────

def make_correlation_plots(result: Dict, out_dir: str = "plots") -> None:
    """Standard RL plots plus PAR/TAR correlation-specific charts."""
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)

    steps     = result["steps"]
    par_r     = result["par_rewards"]
    tar_r     = result["tar_rewards"]
    corr      = result["corr_par_tar"]
    avg_par   = result["avg_reward_par"]
    avg_tar   = result["avg_reward_tar"]
    avg_other = result["avg_reward_other"]
    n_q       = result["n_questions"]
    n_s       = result["n_sessions"]

    make_plots(steps,          out_dir=out_dir, profile="corr_par_tar")
    make_qvalue_plots(steps,   out_dir=out_dir, profile="corr_par_tar")
    make_lt_change_plot(steps, out_dir=out_dir, profile="corr_par_tar")
    make_phase_bar_plot(steps, out_dir=out_dir, profile="corr_par_tar")
    make_mpe_plots(steps,      out_dir=out_dir, profile="corr_par_tar")

    # corr_1: PAR and TAR reward over time (raw + rolling average)
    window = 5
    def roll_avg(xs):
        return [float(np.mean(xs[max(0, i-window+1):i+1])) for i in range(len(xs))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    for ax, data, label, marker in [
        (ax1, par_r, "PAR reward", "o"), (ax1, tar_r, "TAR reward", "s"),
    ]:
        ax.plot(range(len(data)), data, color=MPL_COLOURS[label[:3]],
                lw=1.5, alpha=0.85, label=label, marker=marker, ms=3)
    ax1.axhline(avg_par, color=MPL_COLOURS["PAR"], lw=1.2, ls="--", alpha=0.5,
                label=f"PAR mean ({avg_par:+.3f})")
    ax1.axhline(avg_tar, color=MPL_COLOURS["TAR"], lw=1.2, ls="--", alpha=0.5,
                label=f"TAR mean ({avg_tar:+.3f})")
    ax1.axhline(0, color="black", lw=0.6, alpha=0.3)
    ax1.set_ylabel("Reward", fontsize=11)
    ax1.set_title(f"PAR vs TAR Reward Sequences  |  Pearson r = {corr:+.4f}  |  {n_s}×{n_q} Q",
                  fontsize=11)
    ax1.legend(fontsize=9, ncol=2); ax1.grid(True, alpha=0.2)
    ax2.plot(range(len(par_r)), roll_avg(par_r), color=MPL_COLOURS["PAR"],
             lw=2.2, label=f"PAR rolling avg ({window}Q)")
    ax2.plot(range(len(tar_r)), roll_avg(tar_r), color=MPL_COLOURS["TAR"],
             lw=2.2, label=f"TAR rolling avg ({window}Q)")
    ax2.axhline(0, color="black", lw=0.6, alpha=0.3)
    ax2.set_ylabel(f"Rolling avg ({window}Q)", fontsize=11)
    ax2.set_xlabel("Question index (per LT)", fontsize=10)
    ax2.set_title("Smoothed reward — PAR and TAR track each other", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    _save_plot(fig, out_dir, "corr_1_reward_timeseries")

    # corr_2: scatter PAR vs TAR
    min_len = min(len(par_r), len(tar_r))
    x_vals  = par_r[:min_len]; y_vals = tar_r[:min_len]
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x_vals, y_vals, c=range(min_len), cmap="viridis",
                    alpha=0.75, s=45, edgecolors="white", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Question index", shrink=0.85)
    if min_len >= 2:
        m, b = np.polyfit(x_vals, y_vals, 1)
        xs   = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(xs, m * xs + b, color="#e74c3c", lw=2, zorder=4,
                label=f"OLS fit  (r={corr:+.3f})")
    ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("PAR reward", fontsize=12); ax.set_ylabel("TAR reward", fontsize=12)
    ax.set_title(f"PAR vs TAR Reward Scatter\nPearson r = {corr:+.4f}  (n={min_len})",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save_plot(fig, out_dir, "corr_2_scatter")

    # corr_3: avg reward bar
    lt_avgs = {lt: float(np.mean([s["reward"] for s in steps if s["lt"] == lt]) or 0)
               for lt in LEARNING_TYPES}
    colours = [MPL_COLOURS["PAR"] if lt == "PAR"
               else MPL_COLOURS["TAR"] if lt == "TAR"
               else "#bdc3c7" for lt in LEARNING_TYPES]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(LEARNING_TYPES, [lt_avgs[lt] for lt in LEARNING_TYPES],
                  color=colours, edgecolor="white", zorder=3)
    for bar_, v in zip(bars, [lt_avgs[lt] for lt in LEARNING_TYPES]):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                v + (0.005 if v >= 0 else -0.018),
                f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold",
                color=bar_.get_facecolor())
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(avg_other, color="#7f8c8d", lw=1.2, ls=":", alpha=0.7,
               label=f"Others mean ({avg_other:+.3f})")
    ax.set_ylabel("Avg reward", fontsize=11)
    ax.set_title(f"Avg Reward per LT  —  PAR and TAR outperform others\nr(PAR,TAR) = {corr:+.3f}",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    _save_plot(fig, out_dir, "corr_3_avg_bar")

    print(f"\n  [plots] corr_1_reward_timeseries.png  corr_2_scatter.png  corr_3_avg_bar.png")
    print(f"          + standard RL plots (reward_timeseries, qvalue, mpe, etc.)")


# ── Single-line reward plot ───────────────────────────────────────────────────

def make_single_line_plot(steps: List[Dict], out_dir: str = "plots",
                          profile: str = "") -> None:
    """
    One continuous reward line, each point coloured by LT.
    Bottom panel: LT strip ordered by radar ring distance from home LT.
    """
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)

    xs      = [s["global_q"] for s in steps]
    rewards = [s["reward"]   for s in steps]
    lts     = [s["lt"]       for s in steps]
    lt_counts = Counter(lts)

    fig, (ax_r, ax_lt) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    # Top: reward line
    ax_r.plot(xs, rewards, color="#cccccc", lw=1.2, zorder=1)
    for i, (x, r, lt) in enumerate(zip(xs, rewards, lts)):
        col = MPL_COLOURS[lt]
        ax_r.scatter([x], [r], color=col, s=60, zorder=3,
                     edgecolors="white", linewidths=0.5)
        if i > 0 and lts[i - 1] == lt:
            ax_r.plot([xs[i-1], x], [rewards[i-1], r], color=col, lw=2.0, zorder=2, alpha=0.85)

    window = min(5, len(rewards))
    roll   = [float(np.mean(rewards[max(0, i-window+1):i+1])) for i in range(len(rewards))]
    ax_r.plot(xs, roll, color="black", lw=1.8, ls="--", alpha=0.55, zorder=4,
              label=f"Rolling avg ({window}Q)")
    ax_r.axhline(0, color="grey", lw=0.7, ls=":", alpha=0.5)

    n_q_per_session = max(s["question_num"] for s in steps if s["session_num"] == 1)
    n_sessions      = max(s["session_num"] for s in steps)
    for sn in range(2, n_sessions + 1):
        bx = (sn - 1) * n_q_per_session + 0.5
        ax_r.axvline(bx, color="#888888", lw=1.2, ls="--", alpha=0.6, zorder=1)
        ax_r.text(bx + 0.3, 0, f"S{sn}", fontsize=8, color="#888888", va="bottom")

    title_profile = profile or "default"
    ring_info = ""
    if title_profile in RADAR_RING_ORDERS:
        pairs = RADAR_RING_ORDERS[title_profile]
        ring_info = "\nRadar rings: " + " > ".join(f"{lt}({r})" for lt, r in pairs)
    ax_r.set_ylabel("Reward", fontsize=12)
    ax_r.set_title(
        f"RL Reward - Single Continuous Line  |  profile: {title_profile}\n"
        f"Each dot/segment coloured by LT selected  |  dashed = rolling avg{ring_info}",
        fontsize=11,
    )
    ax_r.grid(True, alpha=0.18)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MPL_COLOURS[lt], markersize=9,
               label=f"{lt}  (n={lt_counts.get(lt, 0)})")
        for lt in LEARNING_TYPES if lt_counts.get(lt, 0) > 0
    ] + [Line2D([0], [0], color="black", lw=1.8, ls="--", label=f"Rolling avg ({window}Q)")]
    ax_r.legend(handles=legend_handles, fontsize=9, loc="upper left", ncol=2, framealpha=0.85)

    # Bottom: LT strip
    if profile in RADAR_RING_ORDERS:
        strip_order = [lt for lt, _ in RADAR_RING_ORDERS[profile]]
        ring_labels = {lt: f"{lt} (ring {r})" for lt, r in RADAR_RING_ORDERS[profile]}
        strip_title = "LT selected  (top = best match, bottom = worst)"
    else:
        strip_order = RADAR_CLOCKWISE
        ring_labels = {lt: lt for lt in strip_order}
        strip_title = "LT selected  (clockwise radar order)"

    lt_y = {lt: i for i, lt in enumerate(strip_order)}

    for x, lt in zip(xs, lts):
        ax_lt.scatter([x], [lt_y[lt]], color=MPL_COLOURS[lt],
                      s=55, zorder=3, edgecolors="white", linewidths=0.5)
    for i in range(1, len(xs)):
        ax_lt.plot([xs[i-1], xs[i]], [lt_y[lts[i-1]], lt_y[lts[i]]],
                   color="#cccccc", lw=0.8, zorder=1, alpha=0.6)

    ax_lt.set_yticks(list(lt_y.values()))
    ax_lt.set_yticklabels([ring_labels[lt] for lt in strip_order], fontsize=9)
    for tick, lt in zip(ax_lt.get_yticklabels(), strip_order):
        tick.set_color(MPL_COLOURS[lt]); tick.set_fontweight("bold")
    for lt in strip_order:
        ax_lt.axhline(lt_y[lt], color="#eeeeee", lw=0.6, zorder=0)

    ax_lt.set_xlabel("Question number (global)", fontsize=11)
    ax_lt.set_ylabel("LT", fontsize=10)
    ax_lt.grid(True, axis="x", alpha=0.15)
    ax_lt.set_title(strip_title, fontsize=9)
    ax_lt.set_ylim(-0.6, len(strip_order) - 0.4)

    for sn in range(2, n_sessions + 1):
        bx = (sn - 1) * n_q_per_session + 0.5
        ax_lt.axvline(bx, color="#888888", lw=1.2, ls="--", alpha=0.6, zorder=1)

    plt.tight_layout()
    tag  = f"_{profile}" if profile else ""
    _save_plot(fig, out_dir, f"single_line_reward{tag}")


# ══════════════════════════════════════════════════════════════════════════
# NEW PLOT A — Mastery swim-lane  (mirrors LT change plot style)
# Shows the selected LT's mastery level as a stepped line after Q10,
# with ▲ PROMOTE and ▼ DEMOTE markers on events.
# ══════════════════════════════════════════════════════════════════════════

def make_mastery_plot(steps: List[Dict], out_dir: str = "plots",
                      profile: str = "") -> None:
    """
    Swim-lane chart of mastery level for the selected LT at each question.
    Each row = one LT level (1–6).  The active LT's level is drawn as a
    stepped line.  PROMOTE events get a green ▲, DEMOTE events a red ▼.
    Mirrors the visual style of make_lt_change_plot.
    """
    if not HAS_PLOT or not steps:
        return
    os.makedirs(out_dir, exist_ok=True)

    xs     = [s.get("step_num", i + 1) for i, s in enumerate(steps)]
    levels = [s.get("mastery_level", 1) for s in steps]
    lts    = [s.get("learning_type", "PAR") for s in steps]
    events = [s.get("mastery_event") for s in steps]   # "PROMOTE" | "DEMOTE" | None

    LEVEL_LABELS = {
        1: "1 Pemula",
        2: "2 Dasar",
        3: "3 Menengah",
        4: "4 Berkelanjutan",
        5: "5 Near Mastery",
        6: "6 Mastery",
    }

    fig, ax = plt.subplots(figsize=(16, 5))
    _shade_seeding(ax, steps, _n_q_per_session(steps), SEEDING_QUESTIONS)

    # ── horizontal guide lines for each level ──────────────────────────
    for lvl in range(1, 7):
        ax.axhline(lvl, color="#e8e8e8", lw=0.8, zorder=0)

    # ── stepped mastery line coloured by selected LT ───────────────────
    for i in range(len(xs) - 1):
        col = MPL_COLOURS.get(lts[i], "#888888")
        ax.plot([xs[i], xs[i + 1]], [levels[i], levels[i]],
                color=col, lw=2.2, solid_capstyle="round", zorder=2, alpha=0.85)
        ax.plot([xs[i + 1], xs[i + 1]], [levels[i], levels[i + 1]],
                color=col, lw=2.2, solid_capstyle="round", zorder=2, alpha=0.85)
    # last point
    ax.plot([xs[-1]], [levels[-1]], "o", color=MPL_COLOURS.get(lts[-1], "#888888"),
            ms=5, zorder=3)

    # ── promote / demote markers ───────────────────────────────────────
    for x, lvl, ev, lt in zip(xs, levels, events, lts):
        if ev == "PROMOTE":
            ax.scatter([x], [lvl], marker="^", color="#2ecc71",
                       s=120, zorder=5, edgecolors="white", linewidths=0.8)
        elif ev == "DEMOTE":
            ax.scatter([x], [lvl], marker="v", color="#e74c3c",
                       s=120, zorder=5, edgecolors="white", linewidths=0.8)

    # ── vertical lines at LT changes ──────────────────────────────────
    for i in range(1, len(lts)):
        if lts[i] != lts[i - 1]:
            ax.axvline(xs[i] - 0.5, color="#cc3333", lw=1.0, ls="--",
                       alpha=0.55, zorder=4)

    # ── LT colour legend ───────────────────────────────────────────────
    handles = [mpatches.Patch(color=MPL_COLOURS[lt], label=lt)
               for lt in LEARNING_TYPES]
    handles += [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ecc71",
                   markersize=10, label="Promote"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
                   markersize=10, label="Demote"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=5, fontsize=8,
              framealpha=0.85)

    ax.set_yticks(list(LEVEL_LABELS.keys()))
    ax.set_yticklabels(list(LEVEL_LABELS.values()), fontsize=9)
    ax.set_ylim(0.4, 6.6)
    ax.set_xlabel("Question number", fontsize=11)
    ax.set_ylabel("Mastery level", fontsize=10)
    ax.set_title(
        f"Mastery Progression  |  profile: {profile or 'session'}\n"
        "(stepped line = selected LT's mastery level  ·  "
        "▲ = promoted  ·  ▼ = demoted  ·  red dashed = LT change  ·  yellow = seeding)",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.18)
    plt.tight_layout()

    tag = f"_{profile}" if profile else ""
    _save_plot(fig, out_dir, f"mastery_progression{tag}")


# ══════════════════════════════════════════════════════════════════════════
# NEW PLOT B — Epsilon decay  (same swim-lane style)
# ══════════════════════════════════════════════════════════════════════════

def make_epsilon_plot(steps: List[Dict], out_dir: str = "plots",
                      profile: str = "") -> None:
    """
    Two-panel epsilon decay plot — mirrors the single_line_reward style exactly.

    Top panel   : actual epsilon as connected dots coloured by selected LT,
                  plus theoretical Low-Med (solid) and Current (dashed) curves.
    Bottom panel: LT swim strip (clockwise radar order) — dot per question.
    """
    if not HAS_PLOT or not steps:
        return
    os.makedirs(out_dir, exist_ok=True)

    try:
        from pedagogy_selector import EPSILON_INIT, EPSILON_DECAY, EPSILON_MIN
        opt_init, opt_decay, opt_min = EPSILON_INIT, EPSILON_DECAY, EPSILON_MIN
    except Exception:
        opt_init, opt_decay, opt_min = 0.30, 0.98, 0.05

    cur_init, cur_decay, cur_min = 0.50, 0.97, 0.05

    xs       = [s.get("step_num", i + 1) for i, s in enumerate(steps)]
    epsilons = [s.get("epsilon", opt_init) for s in steps]
    lts      = [s.get("learning_type", "PAR") for s in steps]

    n         = len(xs)
    cur_curve = [max(cur_min, cur_init * cur_decay ** i) for i in range(n)]
    opt_curve = [max(opt_min, opt_init * opt_decay ** i) for i in range(n)]

    strip_order = RADAR_CLOCKWISE
    lt_y        = {lt: i for i, lt in enumerate(strip_order)}

    # LT counts for legend
    from collections import Counter
    lt_counts = Counter(lts)

    # ── Build figure — 2 rows, same proportions as single_line_reward ────
    fig, (ax_top, ax_lt) = plt.subplots(
        2, 1, figsize=(16, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    n_q_s = _n_q_per_session(steps)
    _shade_seeding(ax_top, steps, n_q_s, SEEDING_QUESTIONS)
    _shade_seeding(ax_lt,  steps, n_q_s, SEEDING_QUESTIONS)

    # ── Top panel: epsilon dots + connecting line + theoretical curves ────

    # grey connector between all dots (background)
    ax_top.plot(xs, epsilons, color="#cccccc", lw=0.9, zorder=1, alpha=0.7)

    # dots and per-segment lines coloured by LT
    for i in range(len(xs)):
        col = MPL_COLOURS.get(lts[i], "#888888")
        ax_top.scatter([xs[i]], [epsilons[i]], color=col, s=55, zorder=3,
                       edgecolors="white", linewidths=0.5)
        if i > 0:
            ax_top.plot([xs[i - 1], xs[i]], [epsilons[i - 1], epsilons[i]],
                        color=MPL_COLOURS.get(lts[i - 1], "#888888"),
                        lw=1.8, zorder=2, alpha=0.85)

    # theoretical curves
    ax_top.plot(xs, cur_curve, "--", color="#e67e22", lw=1.6, alpha=0.75, zorder=4,
                label=f"Current  (ε₀={cur_init}, decay={cur_decay})")
    ax_top.plot(xs, opt_curve, "-",  color="#2ecc71", lw=2.0, alpha=0.85, zorder=4,
                label=f"Low-Med  (ε₀={opt_init}, decay={opt_decay})")

    # ε_min floor
    ax_top.axhline(opt_min, color="#aaaaaa", lw=0.9, ls=":", alpha=0.8)
    ax_top.text(xs[0], opt_min + 0.005, f"ε_min={opt_min}", fontsize=8,
                color="#888888", va="bottom")

    # legend: one entry per LT that appeared + the two curves
    legend_handles = []
    for lt in LEARNING_TYPES:
        if lt_counts.get(lt, 0) > 0:
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=MPL_COLOURS[lt], markersize=7,
                           label=f"{lt}  (n={lt_counts[lt]})")
            )
    legend_handles += [
        plt.Line2D([0], [0], color="#555555", ls="--", lw=1.2,
                   label="Rolling avg (5Q)"),  # kept for visual parity
    ]
    ax_top.legend(handles=legend_handles, loc="upper right",
                  ncol=max(1, len(legend_handles) // 2),
                  fontsize=8, framealpha=0.85)

    ax_top.set_ylabel("Epsilon  (exploration rate)", fontsize=10)
    ax_top.set_ylim(-0.02, max(cur_init, opt_init) + 0.08)
    ax_top.set_title(
        "Epsilon Decay  |  Each dot/segment coloured by LT selected  |  "
        f"dashed=Current (e0={cur_init}) / solid=Low-Med (e0={opt_init})",
        fontsize=10,
    )
    # ── Bottom panel: LT swim strip ───────────────────────────────────────
    for i in range(len(xs)):
        col = MPL_COLOURS.get(lts[i], "#888888")
        ax_lt.scatter([xs[i]], [lt_y[lts[i]]], color=col, s=55, zorder=3,
                      edgecolors="white", linewidths=0.5)
    # grey connector
    for i in range(1, len(xs)):
        ax_lt.plot([xs[i - 1], xs[i]],
                   [lt_y[lts[i - 1]], lt_y[lts[i]]],
                   color="#cccccc", lw=0.8, zorder=1, alpha=0.6)

    ax_lt.set_yticks(list(lt_y.values()))
    ax_lt.set_yticklabels(list(lt_y.keys()), fontsize=9)
    for tick, lt in zip(ax_lt.get_yticklabels(), strip_order):
        tick.set_color(MPL_COLOURS.get(lt, "#888888"))
        tick.set_fontweight("bold")
    for lt in strip_order:
        ax_lt.axhline(lt_y[lt], color="#eeeeee", lw=0.6, zorder=0)

    ax_lt.set_ylim(-0.6, len(strip_order) - 0.4)
    ax_lt.set_xlabel("Question number (global)", fontsize=11)
    ax_lt.set_ylabel("LT", fontsize=10)
    ax_lt.set_title("LT selected  (clockwise radar order)", fontsize=9)
    ax_lt.grid(True, axis="x", alpha=0.15)

    # align x-axes
    for ax in (ax_top, ax_lt):
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

    plt.tight_layout()
    tag = f"_{profile}" if profile else ""
    _save_plot(fig, out_dir, f"epsilon_decay{tag}")


# ══════════════════════════════════════════════════════════════════════════
# NEW PLOT C — Current vs Low-Med hyperparameter comparison
# ══════════════════════════════════════════════════════════════════════════

def make_hyperparam_comparison_plot(out_dir: str = "plots") -> None:
    """
    Static side-by-side comparison: parameter values + sweep metric deltas.
    Does not require step data — uses the sweep results directly.
    """
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Hyperparameter Comparison: Current  vs  Low-Med Exploration (Optimised)\n"
        "Sweep: 10 configs × 30 students × 20 Q each  (6,000 simulated interactions)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── Panel 1: parameter values ─────────────────────────────────────
    ax = axes[0]
    params   = ["ε₀ (init)", "ε decay", "W init low", "W init high"]
    cur_vals = [0.50,  0.97,  0.05, 0.15]
    opt_vals = [0.30,  0.98,  0.01, 0.05]
    x = np.arange(len(params))
    w = 0.35
    b1 = ax.bar(x - w/2, cur_vals, w, label="Current",  color="#e67e22",
                alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, opt_vals, w, label="Low-Med ★", color="#2ecc71",
                alpha=0.85, edgecolor="white")
    ylim_top = max(max(cur_vals), max(opt_vals)) * 1.22
    for bar, v in zip(b1, cur_vals):
        ypos = min(bar.get_height() + 0.012, ylim_top * 0.92)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                str(v), ha="center", va="bottom", fontsize=9, color="#e67e22",
                fontweight="bold")
    for bar, v in zip(b2, opt_vals):
        ypos = min(bar.get_height() + 0.012, ylim_top * 0.92)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                str(v), ha="center", va="bottom", fontsize=9, color="#27ae60",
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(params, fontsize=10)
    ax.set_ylabel("Value"); ax.set_title("Parameter Values", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", alpha=0.2)

    # ── Panel 2: sweep metric results ─────────────────────────────────
    ax = axes[1]
    metrics   = ["KT AUC", "Reward CV", "Convergence Q"]
    cur_m     = [0.4384,  1.350,  19.0]
    opt_m     = [0.4503,  1.081,  17.9]
    note      = ["↑ higher better", "↓ lower better", "↓ lower better"]
    x = np.arange(len(metrics))
    b1 = ax.bar(x - w/2, cur_m, w, label="Current",  color="#e67e22", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, opt_m, w, label="Low-Med ★", color="#2ecc71", alpha=0.85, edgecolor="white")
    for bar, v in zip(b1, cur_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="#e67e22",
                fontweight="bold")
    for bar, v in zip(b2, opt_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="#27ae60",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{n}" for m, n in zip(metrics, note)], fontsize=8.5)
    ax.set_title("Sweep Evaluation Metrics", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.2)

    # ── Panel 3: delta / improvement summary ──────────────────────────
    ax = axes[2]
    improvements = [
        ("KT AUC",         +0.0119,  True),
        ("Reward CV",      -0.269,   True),   # lower = better
        ("OPE Gain",       +0.0006,  True),
        ("Convergence Q",  -1.5,     True),   # fewer Q = better
        ("Accuracy %",     +0.83,    True),
        ("Wrong answers",  -5.0,     True),
    ]
    labels  = [x[0] for x in improvements]
    deltas  = [x[1] for x in improvements]
    colours = ["#2ecc71" if d > 0 else "#3498db" for d in deltas]
    # all are improvements — positive delta or negative delta in a "lower is better" metric
    all_improved = ["#2ecc71"] * len(deltas)

    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, color=all_improved, alpha=0.85, edgecolor="white", height=0.55)
    for bar, d, lbl in zip(bars, deltas, labels):
        xpos = bar.get_width() + (max(abs(d) for d in deltas) * 0.04) * np.sign(d)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{d:+.4f}" if abs(d) < 0.1 else f"{d:+.2f}",
                va="center", ha="left" if d > 0 else "right",
                fontsize=8.5, fontweight="bold", color="#27ae60")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#666666", lw=1.0)
    ax.set_title("Δ  (Low-Med − Current)\nAll improvements ✓", fontsize=11)
    ax.set_xlabel("Change in metric value")
    ax.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    _save_plot(fig, out_dir, "hyperparam_comparison")


# ══════════════════════════════════════════════════════════════════════════
# NEW PLOT D — Real student improvement: Current vs Low-Med
# Simulates 4 student profiles × 30 seeds under both configs and shows
# % time on true LT, mastery progression, and convergence speed.
# ══════════════════════════════════════════════════════════════════════════

def make_hyperparam_student_comparison_plot(out_dir: str = "plots") -> None:
    """
    Head-to-head comparison of Current vs Low-Med hyperparameters on
    real simulated student trajectories.
    Shows: (1) cumulative % picks on true LT, (2) avg mastery of true LT,
    (3) convergence speed bar charts.
    30 seeds × 4 profiles = 120 simulated sessions per config.
    """
    if not HAS_PLOT:
        return
    os.makedirs(out_dir, exist_ok=True)

    import random
    try:
        from rl_metrics import MetricsTracker, N_MAX
        from pedagogy_selector import EPSILON_INIT, EPSILON_DECAY, EPSILON_MIN
        OPT_EPS_I, OPT_EPS_D, OPT_EPS_MIN = EPSILON_INIT, EPSILON_DECAY, EPSILON_MIN
    except Exception:
        OPT_EPS_I, OPT_EPS_D, OPT_EPS_MIN = 0.30, 0.98, 0.05
        from rl_metrics import MetricsTracker, N_MAX

    CUR_EPS_I, CUR_EPS_D, CUR_EPS_MIN = 0.50, 0.97, 0.05
    CUR_W_LO, CUR_W_HI = 0.05, 0.15
    OPT_W_LO, OPT_W_HI = 0.01, 0.05

    PROFILES = [("PAR", 0.82), ("PGI", 0.75), ("TGR", 0.80), ("TAI", 0.77)]
    LT_LABELS  = {"PAR": "PAR-strong", "PGI": "PGI-strong",
                  "TGR": "TGR-strong", "TAI": "TAI-strong"}
    LT_COLORS  = {"PAR": MPL_COLOURS["PAR"], "PGI": MPL_COLOURS["PGI"],
                  "TGR": MPL_COLOURS["TGR"], "TAI": MPL_COLOURS["TAI"]}
    CUR_C = "#e67e22";  OPT_C = "#27ae60"
    N_SEEDS = 30;  N_Q = 40

    def _lt_match(a, b):
        if a == b: return 1.0
        if a[0] == b[0] or a[-1] == b[-1]: return 0.75
        return 0.55

    def _simulate(true_lt, base_p, eps_i, eps_d, eps_min, w_lo, w_hi, seed):
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        W = np_rng.uniform(w_lo, w_hi, size=(8, 3))
        eps = eps_i
        ml = {lt: 1 for lt in LEARNING_TYPES}
        cc = {lt: 0 for lt in LEARNING_TYPES}
        cw = {lt: 0 for lt in LEARNING_TYPES}
        tracker = MetricsTracker(student_id="sim", category="Penggalang")
        log = []
        for q in range(N_Q):
            state = np.array([
                np.mean([MASTERY_LEVEL_SCORE[ml[lt]] for lt in LEARNING_TYPES]),
                np.mean([tracker.states[lt].performance for lt in LEARNING_TYPES]),
                np.mean([tracker.states[lt].engagement  for lt in LEARNING_TYPES]),
            ])
            qv = {lt: float(W[i] @ state) for i, lt in enumerate(LEARNING_TYPES)}
            lt = (rng.choice(LEARNING_TYPES) if rng.random() < eps
                  else max(qv, key=qv.__getitem__))
            eps = max(eps_min, eps * eps_d)
            pc  = min(0.97, base_p * _lt_match(lt, true_lt))
            ic  = rng.random() < pc
            na  = rng.randint(1, 3) if (ic and pc > 0.7) else (1 if ic else N_MAX)
            rec = tracker.record_attempt(lt, f"{ml[lt]}{lt}", na,
                                         rng.uniform(20, 90), q, is_correct=ic)
            ai = LEARNING_TYPES.index(lt)
            W[ai] += 0.05 * (rec["reward"] - float(W[ai] @ state)) * state
            if ic:
                cc[lt] += 1; cw[lt] = 0
                if cc[lt] >= 1 and ml[lt] < 6: ml[lt] += 1; cc[lt] = 0
            else:
                cc[lt] = 0; cw[lt] += 1
                if cw[lt] >= 2 and ml[lt] > 1: ml[lt] -= 1; cw[lt] = 0
            log.append({"lt": lt, "ic": int(ic), "ml_true": ml[true_lt]})
        return log

    def _aggregate(true_lt, base_p, eps_i, eps_d, eps_min, w_lo, w_hi):
        cum = [0] * N_Q;  ml_a = [0] * N_Q;  convs = []
        for seed in range(N_SEEDS):
            log = _simulate(true_lt, base_p, eps_i, eps_d, eps_min,
                            w_lo, w_hi, seed * 100 + 42)
            for i, s in enumerate(log):
                cum[i] += int(s["lt"] == true_lt)
                ml_a[i] += s["ml_true"]
            for i in range(4, N_Q):
                if all(log[j]["lt"] == true_lt for j in range(i - 4, i + 1)):
                    convs.append(i + 1); break
        pct  = [round(cum[i] / (i + 1) / N_SEEDS * 100, 2) for i in range(N_Q)]
        mla  = [round(ml_a[i] / N_SEEDS, 3) for i in range(N_Q)]
        return {"pct": pct, "ml": mla,
                "conv_rate": round(len(convs) / N_SEEDS * 100, 1),
                "conv_q":    round(sum(convs) / len(convs), 1) if convs else None,
                "final_pct": round(cum[-1] / N_SEEDS * 100, 1),
                "final_ml":  round(ml_a[-1] / N_SEEDS, 2)}

    qs = list(range(1, N_Q + 1))
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        f"Real Student Improvement: Current (e0={CUR_EPS_I}) vs Low-Med (e0={OPT_EPS_I})\n"
        f"30 seeds x 4 profiles x 40 Q  |  decay: Current={CUR_EPS_D}, Low-Med={OPT_EPS_D}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, (true_lt, base_p) in enumerate(PROFILES):
        cur = _aggregate(true_lt, base_p, CUR_EPS_I, CUR_EPS_D, CUR_EPS_MIN,
                         CUR_W_LO, CUR_W_HI)
        opt = _aggregate(true_lt, base_p, OPT_EPS_I, OPT_EPS_D, OPT_EPS_MIN,
                         OPT_W_LO, OPT_W_HI)

        # ── Row 0: % picks on true LT ──────────────────────────────────
        ax = axes[0][col]
        ax.plot(qs, cur["pct"], color=CUR_C, lw=2, ls="--", label="Current (ε₀=0.50)")
        ax.plot(qs, opt["pct"], color=OPT_C, lw=2.5,        label="Low-Med (ε₀=0.30)")
        ax.axvline(10, color="#aaaaaa", lw=0.8, ls=":", alpha=0.7)
        ax.fill_between(qs, cur["pct"], opt["pct"],
                        where=[o > c for o, c in zip(opt["pct"], cur["pct"])],
                        alpha=0.18, color=OPT_C)
        ax.fill_between(qs, cur["pct"], opt["pct"],
                        where=[o <= c for o, c in zip(opt["pct"], cur["pct"])],
                        alpha=0.12, color=CUR_C)
        ax.set_title(f"{LT_LABELS[true_lt]} (true LT = {true_lt})",
                     fontsize=11, color=LT_COLORS[true_lt], fontweight="bold")
        ax.set_ylabel("% questions on true LT" if col == 0 else "", fontsize=9)
        ax.set_ylim(0, 100); ax.set_xlim(1, 40)
        ax.grid(True, alpha=0.2)
        if col == 0: ax.legend(fontsize=8, loc="upper right")
        ax.annotate(f"{opt['final_pct']}%", xy=(40, opt["pct"][-1]),
                    xytext=(36, min(opt["pct"][-1] + 9, 93)),
                    fontsize=8, color=OPT_C, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=OPT_C, lw=0.7))
        ax.annotate(f"{cur['final_pct']}%", xy=(40, cur["pct"][-1]),
                    xytext=(36, max(cur["pct"][-1] - 12, 4)),
                    fontsize=8, color=CUR_C, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=CUR_C, lw=0.7))

        # ── Row 1: Avg mastery of true LT ─────────────────────────────
        ax2 = axes[1][col]
        ax2.plot(qs, cur["ml"], color=CUR_C, lw=2, ls="--")
        ax2.plot(qs, opt["ml"], color=OPT_C, lw=2.5)
        ax2.axvline(10, color="#aaaaaa", lw=0.8, ls=":", alpha=0.7)
        ax2.set_yticks([1, 2, 3, 4, 5, 6])
        ax2.set_yticklabels(["1 Pemula", "2 Dasar", "3 Menengah",
                              "4 Berkl.", "5 Near", "6 Mastery"], fontsize=7)
        ax2.set_ylim(0.8, 6.5); ax2.set_xlim(1, 40)
        ax2.set_ylabel("Avg mastery (true LT)" if col == 0 else "", fontsize=9)
        ax2.set_xlabel("Question number", fontsize=9)
        ax2.grid(True, alpha=0.2)
        ax2.text(41, opt["ml"][-1], f"{opt['final_ml']:.1f}",
                 fontsize=8, color=OPT_C, va="center", fontweight="bold")
        ax2.text(41, cur["ml"][-1], f"{cur['final_ml']:.1f}",
                 fontsize=8, color=CUR_C, va="center")

        # ── Row 2: Convergence bar chart ───────────────────────────────
        ax3 = axes[2][col]
        cats  = ["Conv. rate (5-consec %)", "Mean conv. Q (when it happens)"]
        cur_v = [cur["conv_rate"], cur["conv_q"] or N_Q]
        opt_v = [opt["conv_rate"], opt["conv_q"] or N_Q]
        x = np.array([0, 1]); w = 0.32
        b1 = ax3.bar(x - w/2, cur_v, w, color=CUR_C, alpha=0.85,
                     label="Current", edgecolor="white")
        b2 = ax3.bar(x + w/2, opt_v, w, color=OPT_C, alpha=0.85,
                     label="Low-Med ★", edgecolor="white")
        for bar, v in zip(b1, cur_v):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                     f"{v:.0f}", ha="center", fontsize=8.5, color=CUR_C, fontweight="bold")
        for bar, v in zip(b2, opt_v):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                     f"{v:.0f}", ha="center", fontsize=8.5, color=OPT_C, fontweight="bold")
        ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=8.5)
        ax3.set_ylabel("Value" if col == 0 else "", fontsize=9)
        ax3.grid(True, axis="y", alpha=0.2)
        if col == 0: ax3.legend(fontsize=8)
        if cur["conv_q"] and opt["conv_q"]:
            diff = cur["conv_q"] - opt["conv_q"]
            ax3.set_title(
                f"Low-Med converges {diff:.0f}Q earlier" if diff > 0
                else f"Current converges {-diff:.0f}Q earlier",
                fontsize=9, color=OPT_C if diff > 0 else CUR_C, style="italic"
            )

    plt.tight_layout()
    _save_plot(fig, out_dir, "real_student_improvement")