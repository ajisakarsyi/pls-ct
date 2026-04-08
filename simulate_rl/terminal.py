"""
simulate_rl/terminal.py
────────────────────────
All terminal output helpers: per-question row, Q-value panels,
recommendation change panels, session summary, global summary.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from rl_metrics import LEARNING_TYPES, MASTERY_LABELS
from simulate_rl.profiles import (
    G, R, Y, C, M, B, D, X,
    lt_col, reward_col, bar,
)


def fmt_transition(prev: Optional[str], curr: str) -> str:
    if prev is None: return f"{lt_col(curr, curr)} (start)"
    if prev == curr: return f"{lt_col(curr, curr)} ↺"
    return f"{lt_col(prev, prev)} → {lt_col(curr, curr)}"


def print_qvalues(q_vals: Dict[str, float], agent, label: str) -> None:
    q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
    print(f"\n  {D}  ┌── Q-values @ {label} {'─'*36}{X}")
    max_q = max(q_vals.values()) or 0.001
    for lt, qv in q_sorted:
        lvl   = agent.mastery_levels[lt]
        b     = bar(max(0, qv) / max(0.001, max_q))
        arrow = f" {G}◀ best{X}" if lt == q_sorted[0][0] else ""
        print(
            f"  {D}  │ {lt_col(lt, lt):<14} [{b}] {qv:>+8.5f}  "
            f"L{lvl}/{MASTERY_LABELS[lvl][:14]}{arrow}{X}"
        )
    w = agent.tracker
    print(f"  {D}  │ weights α={w.alpha:.3f} β={w.beta:.3f} γ={w.gamma:.3f}{X}")
    print(f"  {D}  └{'─'*50}{X}\n")


def print_recommendation_change(
    step: Dict, agent, q_before: Dict, q_after: Dict, session_num: int,
) -> None:
    chg = step.get("lt_change_detail")
    if chg is None:
        return

    frm      = chg["from_lt"] or "(none)"
    to       = chg["to_lt"]
    cause    = f"[{chg['trigger']}]"
    n_changes = chg["change_index"]

    q_sorted_after = sorted(q_after.items(), key=lambda x: -x[1])
    margins_before = sorted(q_before.values(), reverse=True)
    margins_after  = sorted(q_after.values(),  reverse=True)
    gap_b = margins_before[0] - margins_before[1] if len(margins_before) > 1 else 0
    gap_a = margins_after[0]  - margins_after[1]  if len(margins_after)  > 1 else 0

    print(
        f"\n  {M}{B}  ╔══ LT RECOMMENDATION CHANGE #{n_changes}  "
        f"S{session_num} step#{step['step_num']} ══╗{X}"
    )
    print(
        f"  {M}  ║  {lt_col(frm, frm)} → {lt_col(to, to)}   "
        f"cause={cause}  phase={chg['phase']}{X}"
    )
    print(
        f"  {M}  ║  trigger: used={step['learning_type']}  "
        f"correct={step['is_correct']}  "
        f"reward={reward_col(step['reward'])}  "
        f"td_err={step['td_error']:+.5f}{X}"
    )
    print(
        f"  {M}  ║  M={step['mastery_score']:.2f}  "
        f"P={step['performance']:.2f}  "
        f"E={step['engagement']:.2f}  "
        f"ε={step['epsilon']:.3f}{X}"
    )
    print(f"  {D}  ╠─ Q-table ({'before':>8} → {'after':>8}  {'delta':>8}  bar){X}")

    for lt, qa in q_sorted_after:
        qb   = q_before.get(lt, 0.0)
        dq   = qa - qb
        b    = bar(max(0, qa) / max(0.001, max(q_after.values())), w=12)
        mark = (
            f" {G}◀ NEW top{X}" if lt == to
            else f" {Y}◀ was top{X}" if lt == frm
            else ""
        )
        print(
            f"  {D}  ║  {lt_col(lt, lt):<14} {qb:>+10.5f} → {qa:>+10.5f}"
            f"  {dq:>+8.5f}  [{b}]{X}{mark}"
        )

    print(f"  {D}  ║  Margin gap: before={gap_b:+.5f}  after={gap_a:+.5f}{X}")
    print(f"  {D}  ╚{'═'*58}{X}\n")


def print_session_summary(agent, steps: List[Dict]) -> None:
    from pedagogy_selector import SEEDING_QUESTIONS
    dist    = Counter(s["lt"] for s in steps)
    rewards = [s["reward"] for s in steps]
    seed_s  = [s for s in steps if s.get("phase") == "seeding"]
    free_s  = [s for s in steps if s.get("phase") == "free"]

    print(f"\n  {B}── Session Summary {'─'*50}{X}")
    print(
        f"  Total steps : {len(steps)}  │  "
        f"Avg reward: {np.mean(rewards):+.4f}  │  "
        f"Sum: {sum(rewards):+.4f}"
    )
    if seed_s:
        print(f"  Seeding ({len(seed_s)} Q): avg reward {np.mean([s['reward'] for s in seed_s]):+.4f}")
    if free_s:
        print(f"  Free    ({len(free_s)} Q): avg reward {np.mean([s['reward'] for s in free_s]):+.4f}")

    changes = agent.recommendation_history
    if changes:
        print(f"\n  {B}LT Recommendation Changes ({len(changes)} total):{X}")
        for c in changes:
            frm = c["from_lt"] or "(none)"
            print(
                f"    #{c['change_index']:>2}  step#{c['step_num']:<3}  "
                f"{lt_col(frm, frm):<14} → {lt_col(c['to_lt'], c['to_lt'])}  "
                f"phase={c['phase']}  ε={c['epsilon']:.3f}  [{c['trigger']}]"
            )
    else:
        print("\n  No recommendation changes yet (agent still exploring).")

    print("\n  LT distribution:")
    for lt in LEARNING_TYPES:
        c   = dist.get(lt, 0)
        pct = c / len(steps) * 100 if steps else 0
        b   = "█" * c
        print(f"    {lt_col(lt, lt):<14} {b:<20} {c:>2} ({pct:.0f}%)")

    rec  = agent.recommend()
    best = rec["recommended_lt"]
    print(
        f"\n  {B}Recommendation → {lt_col(best, best)}  "
        f"Q={rec['per_type'][best]['q_value']:+.5f}{X}"
    )


def print_global_summary(steps: List[Dict], n_sessions: int, n_questions: int) -> None:
    print(f"\n{B}{'═'*72}{X}")
    print(
        f"{B}  GLOBAL SUMMARY  —  {n_sessions} sessions × {n_questions} Q "
        f"= {len(steps)} total steps{X}"
    )
    print(f"{B}{'═'*72}{X}")

    rewards = [s["reward"] for s in steps]
    correct = sum(1 for s in steps if s["is_correct"])
    dist    = Counter(s["lt"] for s in steps)

    print(f"\n  Total reward    : {sum(rewards):+.4f}")
    print(f"  Mean reward/step: {np.mean(rewards):+.4f}")
    print(f"  Correct answers : {correct}/{len(steps)} ({correct/len(steps)*100:.1f}%)")
    print(f"  LT changes      : {sum(1 for s in steps if s.get('lt_changed'))}")

    print("\n  LT frequency:")
    for lt in LEARNING_TYPES:
        c   = dist.get(lt, 0)
        pct = c / len(steps) * 100 if steps else 0
        b   = "█" * int(pct / 3)
        print(f"    {lt_col(lt, lt):<14} {b:<25} {c:>4} ({pct:.1f}%)")

    seed_steps = [s for s in steps if s.get("phase") == "seeding"]
    free_steps = [s for s in steps if s.get("phase") == "free"]
    if seed_steps:
        print(
            f"\n  Seeding phase: {len(seed_steps)} Q  "
            f"avg reward={np.mean([s['reward'] for s in seed_steps]):+.4f}"
        )
    if free_steps:
        print(
            f"  Free phase:    {len(free_steps)} Q  "
            f"avg reward={np.mean([s['reward'] for s in free_steps]):+.4f}"
        )
