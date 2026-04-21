"""
app/services/rl.py
──────────────────
RL service layer — wraps pedagogy_selector and rl_metrics for use across
API routes.  Provides a single shared SessionRegistry, cognitive selection,
response recording, console summaries, and plot generation.

All other modules import from here rather than from pedagogy_selector directly.
"""

import base64
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Add project root to sys.path so pedagogy_selector is importable ────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pedagogy_selector import (
    SessionRegistry,
    LEARNING_TYPES as RL_LTS,
    EPSILON_MIN,
    SEEDING_SESSIONS,
)
from rl_metrics import MASTERY_LEVEL_SCORE, MASTERY_LABELS, N_MAX as RL_N_MAX
from app.core.config import get_settings

_settings = get_settings()

# ── Singleton registry shared across all requests ──────────────────────────
rl_registry = SessionRegistry()
rl_registry.log_dir = _settings.rl_logs_dir
os.makedirs(_settings.rl_logs_dir,   exist_ok=True)
os.makedirs(_settings.rl_plots_dir,  exist_ok=True)

print(f"[RL] Learning types: {RL_LTS}  |  Seeding sessions: {SEEDING_SESSIONS}")

# Per-session question-start timestamps for t_answer calculation
session_question_start: Dict[str, float] = {}

# ── Optional matplotlib ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    logger.warning("matplotlib not installed — /rl/plots will be disabled.")


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_cognitive(code: str) -> Optional[Tuple[int, str]]:
    """Parse '2PAR' → (2, 'PAR').  Returns None if invalid."""
    code = code.strip().upper()
    if len(code) == 4 and code[0].isdigit() and code[1:] in RL_LTS:
        return int(code[0]), code[1:]
    return None


def build_cognitive_code(level: int, lt: str) -> str:
    return f"{max(1, min(6, level))}{lt}"


# ── LT selection — called from /chat ──────────────────────────────────────

def rl_select_cognitive(
    session_id:          str,
    category:            str,
    requested_cognitive: Optional[str],
) -> Tuple[str, str, bool, str]:
    """
    Epsilon-greedy LT selection with seeding phase awareness.

    Returns (cognitive_code, lt, rl_selected, phase).

    If requested_cognitive is provided:
      - Its LT component becomes the seeding_lt hint for the agent.
      - During the seeding phase, seeding_lt is still forced.
      - During the free phase, the explicit LT is honoured directly.
    """
    seeding_lt = None
    if requested_cognitive:
        parsed = parse_cognitive(requested_cognitive.upper())
        if parsed:
            seeding_lt = parsed[1]   # e.g. 'PAR' from '2PAR'

    agent = rl_registry.get_agent(session_id, category=category, seeding_lt=seeding_lt)

    if requested_cognitive:
        parsed = parse_cognitive(requested_cognitive.upper())
        if parsed:
            level, lt = parsed
            agent.mastery_levels[lt] = level
            sel_lt, _ = agent.select_action()
            # During free phase honour the explicit request; seeding forces its own LT
            if agent.phase == "free":
                sel_lt = lt
            session_question_start[session_id] = time.time()
            return (
                build_cognitive_code(agent.mastery_levels[sel_lt], sel_lt),
                sel_lt, False, agent.phase,
            )

    lt, _  = agent.select_action()
    level  = agent.mastery_levels[lt]
    session_question_start[session_id] = time.time()
    return build_cognitive_code(level, lt), lt, True, agent.phase


# ── Response recording — called from /evaluate ────────────────────────────

def rl_record_response(
    session_id:       str,
    category:         str,
    lt:               str,
    mastery_level:    int,
    cognitive_code:   str,
    is_correct:       bool,
    wrong_count:      int,
    t_answer_seconds: Optional[float],
) -> Tuple[Optional[Dict], str, Optional[Dict]]:
    """
    Record the student's answer in the RL agent and update Q-values.

    Bug-fix: only record to the RL agent (and therefore to MLR history) when
    the question is *resolved* — i.e. the student answered correctly OR
    exhausted all N_MAX attempts.  Intermediate wrong answers return early
    without touching the bandit so that:
      1. MLR trains on one observation per question, not 4 for 3-wrong+1-correct.
      2. The reward signal is clean — partial credit is not given for wrong tries.

    Returns (rl_step, next_cognitive_code, lt_change_info).
    """
    try:
        agent = rl_registry.get_agent(session_id, category=category)
        agent.mastery_levels[lt] = mastery_level

        # ── Check whether the question is resolved ────────────────────────
        # is_correct=False AND wrong_count+1 < N_MAX → still mid-question
        question_resolved = is_correct or (wrong_count + 1 >= RL_N_MAX)
        if not question_resolved:
            # Do NOT record to RL — this is an intermediate wrong attempt.
            # Return a lightweight info dict so the route can still respond.
            next_cognitive = build_cognitive_code(agent.mastery_levels[lt], lt)
            return (
                None,
                next_cognitive,
                {"changed": False, "current_lt": agent._current_best_lt(),
                 "note": "intermediate_wrong_not_recorded"},
            )

        q_start      = session_question_start.pop(session_id, None)
        t_answer_sec = (
            t_answer_seconds if t_answer_seconds is not None
            else (time.time() - q_start if q_start else 60.0)
        )
        # n_attempt reflects total tries: correct on attempt (wrong_count+1)
        # or N_MAX if exhausted
        n_attempt = (wrong_count + 1) if is_correct else RL_N_MAX

        prev_rec = agent._last_recommendation

        rl_step = agent.record_response(
            learning_type    = lt,
            is_correct       = is_correct,
            n_attempt        = n_attempt,
            t_answer_seconds = t_answer_sec,
            mastery_code     = cognitive_code,
        )

        next_level     = agent.mastery_levels[lt]
        next_cognitive = build_cognitive_code(next_level, lt)

        new_rec = rl_step.get("current_recommendation")
        if new_rec and new_rec != prev_rec and prev_rec is not None:
            lt_change_info = {
                "changed":    True,
                "from_lt":    prev_rec,
                "to_lt":      new_rec,
                "reason":     (
                    f"Q-value of {new_rec} ({agent.q_values()[new_rec]:+.5f}) "
                    f"now exceeds {prev_rec} ({agent.q_values().get(prev_rec, 0):+.5f}) "
                    f"after this step's update."
                ),
                "q_values":   {l: round(q, 5) for l, q in agent.q_values().items()},
                "change_num": len(agent.recommendation_history),
            }
        else:
            lt_change_info = {"changed": False, "current_lt": new_rec}

        rl_registry.save_logs()

        # Console log line
        print(
            f"[RL] {session_id} | {lt} | correct={is_correct} | attempts={n_attempt} | "
            f"phase={agent.phase} | seeding_rem={agent.seeding_remaining} | "
            f"R={rl_step['reward']:+.4f} | M={rl_step['mastery_score']:.2f} "
            f"P={rl_step['performance']:.2f} E={rl_step['engagement']:.2f} "
            f"→ next: {next_cognitive}"
        )

        n_steps = len(agent.step_log)

        # Every 10 resolved questions: print summary AND auto-save live plots
        if n_steps > 0 and n_steps % 10 == 0:
            print_session_summary(session_id, agent, agent.step_log)
            if HAS_PLOT:
                _auto_save_live_plots(session_id)

        return rl_step, next_cognitive, lt_change_info

    except Exception as exc:
        logger.error("[RL] record_response error: %s", exc)
        print(f"[RL] ⚠️  record_response error: {exc}")
        return None, cognitive_code, None


def _auto_save_single_line_plot(session_id: str) -> None:
    """Legacy shim — delegates to _auto_save_live_plots."""
    _auto_save_live_plots(session_id)


def _auto_save_live_plots(session_id: str) -> None:
    """
    Save the three live-update plots after every 10 resolved steps.

    Plots saved (cumulative — full step log each time, never truncated):
      {session_id}_lt_changes.png       — LT recommendation swim-lane
      {session_id}_mastery.png          — mastery level stepped line
      {session_id}_epsilon.png          — epsilon decay coloured by LT

    Each call re-renders with ALL steps so far, meaning the chart grows
    left-to-right rather than being replaced with only the latest 10.
    """
    try:
        from simulate_rl.plots import (
            make_lt_change_plot,
            make_mastery_plot,
            make_epsilon_plot,
        )
        import tempfile, shutil

        agent = rl_registry.get_agent(session_id)
        steps = agent.step_log
        if len(steps) < 2:
            return

        enriched = _enrich_steps(steps, session_id)
        out_dir  = _settings.rl_plots_dir

        # map: plot function → output filename suffix
        LIVE_PLOTS = [
            (make_lt_change_plot, f"lt_recommendation_timeline_{session_id}",
             f"{session_id}_lt_changes.png"),
            (make_mastery_plot,   f"mastery_progression_{session_id}",
             f"{session_id}_mastery.png"),
            (make_epsilon_plot,   f"epsilon_decay_{session_id}",
             f"{session_id}_epsilon.png"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for fn, profile, dest_name in LIVE_PLOTS:
                try:
                    fn(enriched, out_dir=tmpdir, profile=session_id)
                    # plot functions save as e.g. lt_recommendation_timeline_{session_id}.png
                    src_name = profile + ".png"
                    src  = os.path.join(tmpdir, src_name)
                    dest = os.path.join(out_dir, dest_name)
                    if os.path.exists(src):
                        shutil.copy2(src, dest)
                except Exception as exc:
                    logger.error("[RL] live plot %s error: %s", dest_name, exc)

        n = len(steps)
        print(
            f"[RL] Live plots saved at Q{n} → "
            f"{session_id}_lt_changes.png  "
            f"{session_id}_mastery.png  "
            f"{session_id}_epsilon.png"
        )

        # Also keep the single_line plot for backward compat
        generate_single_line_plot(session_id)

    except Exception as exc:
        logger.error("[RL] _auto_save_live_plots error: %s", exc)


# ── Console summary ────────────────────────────────────────────────────────

def print_session_summary(session_id: str, agent: Any, steps: list) -> None:
    """Pretty-print end-of-session summary to the uvicorn console."""
    rewards    = [s["reward"] for s in steps]
    seed_steps = [s for s in steps if s.get("phase") == "seeding"]
    free_steps = [s for s in steps if s.get("phase") == "free"]
    correct    = sum(1 for s in steps if s.get("is_correct"))
    dist       = Counter(s["learning_type"] for s in steps)

    SEP = "═" * 64
    print(f"\n[RL SUMMARY] {SEP}")
    print(f"[RL SUMMARY]  Session : {session_id}")
    print(
        f"[RL SUMMARY]  Steps   : {len(steps)}  |  "
        f"Σ reward: {sum(rewards):+.4f}  |  "
        f"avg: {np.mean(rewards):+.4f}  |  "
        f"correct: {correct}/{len(steps)} ({correct/len(steps)*100:.0f}%)"
    )
    if seed_steps:
        sr = [s["reward"] for s in seed_steps]
        print(
            f"[RL SUMMARY]  Seeding : {len(seed_steps)} Q  "
            f"avg={np.mean(sr):+.4f}  (LT locked to {agent.seeding_lt})"
        )
    if free_steps:
        fr = [s["reward"] for s in free_steps]
        print(f"[RL SUMMARY]  Free    : {len(free_steps)} Q  avg={np.mean(fr):+.4f}")

    print("[RL SUMMARY]  LT distribution (sorted by avg reward):")
    lt_avgs: Dict[str, Any] = {}
    for lt in RL_LTS:
        r = [s["reward"] for s in steps if s["learning_type"] == lt]
        lt_avgs[lt] = (np.mean(r) if r else -999, dist.get(lt, 0))
    for lt, (avg, cnt) in sorted(lt_avgs.items(), key=lambda x: -x[1][0]):
        if cnt == 0:
            continue
        bar  = "█" * cnt + "░" * max(0, 10 - cnt)
        mark = " ◀ recommended" if lt == agent._current_best_lt() else ""
        print(f"[RL SUMMARY]    {lt:<5} [{bar}] {cnt:>3} Q  avg={avg:+.4f}{mark}")

    changes = agent.recommendation_history
    if changes:
        print(f"[RL SUMMARY]  LT recommendation changes ({len(changes)}):")
        for c in changes:
            frm = c["from_lt"] or "(init)"
            print(
                f"[RL SUMMARY]    #{c['change_index']}  step {c['step_num']:>3}  "
                f"{frm:<5} → {c['to_lt']:<5}  "
                f"phase={c['phase']:<8}  ε={c['epsilon']:.3f}  [{c['trigger']}]"
            )
    else:
        print("[RL SUMMARY]  No recommendation changes (still exploring).")

    q    = agent.q_values()
    best = max(q, key=q.get)
    print(f"[RL SUMMARY]  Q-values  (best → {best}):")
    for lt, qv in sorted(q.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(qv / max(q.values()) * 12)) if max(q.values()) > 0 else ""
        print(
            f"[RL SUMMARY]    {lt:<5} {bar:<12} {qv:+.5f}  "
            f"L{agent.mastery_levels[lt]}/{MASTERY_LABELS.get(agent.mastery_levels[lt], '')[:16]}"
        )
    print(
        f"[RL SUMMARY]  weights α={agent.tracker.alpha:.3f} "
        f"β={agent.tracker.beta:.3f} γ={agent.tracker.gamma:.3f}  "
        f"ε={agent.epsilon:.4f}"
    )
    print(f"[RL SUMMARY] {SEP}\n")


# ── Plot generation ────────────────────────────────────────────────────────

def _enrich_steps(steps: list, session_id: str) -> list:
    """
    Convert agent.step_log entries into the dict shape that simulate_rl
    plot functions expect.  Works for both multi-step and single-line plots.
    """
    enriched = []
    prev_lt  = None
    for i, s in enumerate(steps):
        e = dict(s)
        e["lt"]            = s["learning_type"]
        e["q_values"]      = s.get("q_values_after", {})
        e["global_q"]      = s.get("step_num", i + 1)
        e["question_num"]  = s.get("step_num", i + 1)
        e["session_num"]   = 1
        e["prev_lt"]       = prev_lt
        e["mastery_score"] = s.get("mastery_score", 0)
        enriched.append(e)
        prev_lt = s["learning_type"]
    return enriched


def generate_single_line_plot(session_id: str) -> Optional[str]:
    """
    Generate the single-line reward plot coloured by LT — identical to the
    --single-line CLI output but driven by real session data.

    Cold-start aware: no profile label needed; LT ordering falls back to
    the clockwise radar axis order.

    Returns base64 PNG string, or None if matplotlib is unavailable / too few steps.
    """
    if not HAS_PLOT:
        return None

    agent = rl_registry.get_agent(session_id)
    steps = agent.step_log
    if len(steps) < 2:
        return None

    from simulate_rl.plots import make_single_line_plot

    enriched = _enrich_steps(steps, session_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            make_single_line_plot(enriched, out_dir=tmpdir, profile=session_id)
        except Exception as exc:
            logger.error("[plots] single_line error: %s", exc)
            return None

        # make_single_line_plot saves as single_line_reward_{profile}.png
        fname = os.path.join(tmpdir, f"single_line_reward_{session_id}.png")
        if not os.path.exists(fname):
            # Fallback name pattern
            for f in os.listdir(tmpdir):
                if f.startswith("single_line"):
                    fname = os.path.join(tmpdir, f)
                    break

        if not os.path.exists(fname):
            logger.warning("[plots] single_line plot file not found")
            return None

        dest = os.path.join(_settings.rl_plots_dir, f"{session_id}_single_line.png")
        shutil.copy2(fname, dest)
        with open(dest, "rb") as f:
            return base64.b64encode(f.read()).decode()


def generate_plots(session_id: str) -> Dict[str, str]:
    """
    Generate all 8 plots from the agent step_log (7 standard + single_line).

    Returns dict: plot_name → base64 PNG string.
    Also saves PNGs to rl_plots/{session_id}_{name}.png.
    """
    if not HAS_PLOT:
        return {}

    agent = rl_registry.get_agent(session_id)
    steps = agent.step_log
    if len(steps) < 2:
        return {}

    from simulate_rl.plots import (
        make_plots, make_qvalue_plots, make_lt_change_plot,
        make_phase_bar_plot, make_mpe_plots,
        make_mastery_plot, make_epsilon_plot, make_hyperparam_comparison_plot,
        make_hyperparam_student_comparison_plot,
    )

    enriched = _enrich_steps(steps, session_id)

    PLOT_KEYS = {
        f"reward_timeseries_{session_id}.png":          "reward_timeseries",
        f"reward_trend_overlay_{session_id}.png":        "reward_trend_overlay",
        f"lt_transition_heatmap_{session_id}.png":       "lt_transition_heatmap",
        f"qvalue_evolution_{session_id}.png":            "qvalue_evolution",
        f"lt_recommendation_timeline_{session_id}.png":  "lt_recommendation_timeline",
        f"seeding_vs_free_bar_{session_id}.png":         "seeding_vs_free_bar",
        f"mpe_per_lt_{session_id}.png":                  "mpe_per_lt",
        f"mastery_progression_{session_id}.png":         "mastery_progression",
        f"epsilon_decay_{session_id}.png":               "epsilon_decay",
        f"hyperparam_comparison.png":                    "hyperparam_comparison",
        f"real_student_improvement.png":                 "real_student_improvement",
    }

    result: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for fn, label in [
            (make_plots,          "make_plots"),
            (make_qvalue_plots,   "make_qvalue_plots"),
            (make_lt_change_plot, "make_lt_change_plot"),
            (make_phase_bar_plot, "make_phase_bar_plot"),
            (make_mpe_plots,      "make_mpe_plots"),
            (make_mastery_plot,   "make_mastery_plot"),
            (make_epsilon_plot,   "make_epsilon_plot"),
        ]:
            try:
                fn(enriched, out_dir=tmpdir, profile=session_id)
            except Exception as exc:
                logger.error("[plots] %s error: %s", label, exc)

        # static plots (no step data needed)
        for fn, label in [
            (make_hyperparam_comparison_plot,         "make_hyperparam_comparison_plot"),
            (make_hyperparam_student_comparison_plot, "make_hyperparam_student_comparison_plot"),
        ]:
            try:
                fn(out_dir=tmpdir)
            except Exception as exc:
                logger.error("[plots] %s error: %s", label, exc)

        for fname, key in PLOT_KEYS.items():
            src  = os.path.join(tmpdir, fname)
            dest = os.path.join(_settings.rl_plots_dir, f"{session_id}_{key}.png")
            if os.path.exists(src):
                shutil.copy2(src, dest)
                with open(dest, "rb") as f:
                    result[key] = base64.b64encode(f.read()).decode()
            else:
                logger.warning("[plots] not generated: %s", fname)

    # Always include single_line plot in the full set
    sl = generate_single_line_plot(session_id)
    if sl:
        result["single_line"] = sl

    return result