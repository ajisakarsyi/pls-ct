"""
simulate_rl.py — Configurable RL Session Simulator  (v4)
=========================================================
Simulates N sessions × Q questions with realistic student profiles,
verbose terminal output showing every LT decision, and publication-
quality plots of reward / Q-value / LT transitions over time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY UPGRADES IN THIS VERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Seeding phase default = 10 questions (was 5)
  • Seeding is now baked into RLAgent.select_action() — works for
    both simulate_rl and the live API
  • Richer LT change panels printed in terminal (before/after Q-table,
    margin delta, mastery snapshot, trigger cause)
  • 5 new plot functions:
      make_plots()            — per-LT time series + overlay trend
      make_qvalue_plots()     — Q-value evolution with change markers
      make_lt_change_plot()   — timeline of every recommendation change
      make_phase_bar_plot()   — seeding-vs-free LT usage comparison
      make_mpe_plots()        — M / P / E per LT progression subplots
  • Seeding phase visually shaded (yellow) in ALL time-series plots
  • Session dividers drawn in all plots

Usage
-----
  python simulate_rl.py                              # 1 session, 15 Qs
  python simulate_rl.py --sessions 3 --questions 30
  python simulate_rl.py --fixed-lt PAR --questions 25
  python simulate_rl.py --profile par_lean --sessions 5 --questions 20
  python simulate_rl.py --story
  python simulate_rl.py --per-lt 50
  python simulate_rl.py --cross-lt 100
  python simulate_rl.py --all-profiles --sessions 2 --questions 20

Profiles
--------
  random / tgi_lean / par_lean / theoretical_lean
  improving / volatile / pai_misfit
  student_PAR … student_PGI   (8 structural profiles)
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pedagogy_selector import SessionRegistry, LEARNING_TYPES, SEEDING_QUESTIONS
from rl_metrics import MASTERY_LEVEL_SCORE, MASTERY_LABELS, CATEGORY_CONFIG

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found — plots disabled.  pip install matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLOURS
# ─────────────────────────────────────────────────────────────────────────────

G  = "\033[92m"; R  = "\033[91m"; Y  = "\033[93m"; C  = "\033[96m"
M  = "\033[95m"; B  = "\033[1m";  D  = "\033[2m";  X  = "\033[0m"
BL = "\033[34m"

LT_COLOURS = {
    "PAR": "\033[92m", "TAR": "\033[93m", "PAI": "\033[96m", "TAI": "\033[95m",
    "TGR": "\033[94m", "TGI": "\033[91m", "PGR": "\033[32m", "PGI": "\033[35m",
}
MPL_COLOURS = {
    "PAR": "#2ecc71", "TAR": "#f39c12", "PAI": "#3498db", "TAI": "#9b59b6",
    "TGR": "#1abc9c", "TGI": "#e74c3c", "PGR": "#27ae60", "PGI": "#8e44ad",
}

def lt_col(lt, text):  return LT_COLOURS.get(lt, "") + text + X
def reward_col(r):
    if r >  0.10: return f"{G}{r:+.4f}{X}"
    if r >= 0:    return f"{Y}{r:+.4f}{X}"
    return f"{R}{r:+.4f}{X}"
def bar(v, w=16):
    filled = max(0, min(w, round(v * w)))
    return "█" * filled + "░" * (w - filled)

SEED_SHADE = "#fffde7"   # light yellow — seeding phase band in plots


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT PROFILES
# ─────────────────────────────────────────────────────────────────────────────

PROFILES: Dict[str, Dict] = {
    "random": {
        "label": "Random Student",
        "desc":  "Equal ability — forces maximum exploration.",
        "lt": {lt: {"p_correct": 0.50, "speed": 0.70} for lt in LEARNING_TYPES},
        "noise": 0.25,
    },
    "tgi_lean": {
        "label": "Slight TGI/TGR tendency",
        "desc":  "Marginally better at group-individual formats.",
        "lt": {
            "PAR": {"p_correct": 0.48, "speed": 0.80},
            "TAR": {"p_correct": 0.45, "speed": 0.82},
            "PAI": {"p_correct": 0.50, "speed": 0.78},
            "TAI": {"p_correct": 0.44, "speed": 0.83},
            "TGR": {"p_correct": 0.62, "speed": 0.60},
            "TGI": {"p_correct": 0.67, "speed": 0.55},
            "PGR": {"p_correct": 0.58, "speed": 0.63},
            "PGI": {"p_correct": 0.63, "speed": 0.58},
        },
        "noise": 0.18,
    },
    "par_lean": {
        "label": "Slight PAR/PAI tendency",
        "desc":  "Marginally better at practical-individual formats.",
        "lt": {
            "PAR": {"p_correct": 0.68, "speed": 0.55},
            "TAR": {"p_correct": 0.50, "speed": 0.75},
            "PAI": {"p_correct": 0.65, "speed": 0.58},
            "TAI": {"p_correct": 0.48, "speed": 0.78},
            "TGR": {"p_correct": 0.52, "speed": 0.72},
            "TGI": {"p_correct": 0.49, "speed": 0.76},
            "PGR": {"p_correct": 0.54, "speed": 0.70},
            "PGI": {"p_correct": 0.51, "speed": 0.73},
        },
        "noise": 0.18,
    },
    "theoretical_lean": {
        "label": "Slight TAR/TAI tendency",
        "desc":  "Marginally better at theoretical formats.",
        "lt": {
            "PAR": {"p_correct": 0.50, "speed": 0.74},
            "TAR": {"p_correct": 0.66, "speed": 0.57},
            "PAI": {"p_correct": 0.52, "speed": 0.72},
            "TAI": {"p_correct": 0.64, "speed": 0.58},
            "TGR": {"p_correct": 0.49, "speed": 0.76},
            "TGI": {"p_correct": 0.47, "speed": 0.78},
            "PGR": {"p_correct": 0.53, "speed": 0.71},
            "PGI": {"p_correct": 0.50, "speed": 0.74},
        },
        "noise": 0.18,
    },
    "improving": {
        "label": "Improving Student",
        "desc":  "Starts weak, steadily improves across all LTs.",
        "lt": {lt: {"p_correct": 0.28, "speed": 0.95} for lt in LEARNING_TYPES},
        "noise": 0.15,
        "improvement_rate": 0.018,
    },
    "volatile": {
        "label": "Volatile Student",
        "desc":  "High variance — tests agent robustness.",
        "lt": {lt: {"p_correct": 0.52, "speed": 0.72} for lt in LEARNING_TYPES},
        "noise": 0.35,
    },
    "pai_misfit": {
        "label": "PAI Mismatch -> TAR",
        "desc":  "Labelled PAI but is actually a TAR learner.",
        "lt": {
            "PAR": {"p_correct": 0.42, "speed": 0.82},
            "TAR": {"p_correct": 0.88, "speed": 0.32},
            "PAI": {"p_correct": 0.18, "speed": 1.25},
            "TAI": {"p_correct": 0.38, "speed": 0.88},
            "TGR": {"p_correct": 0.30, "speed": 0.98},
            "TGI": {"p_correct": 0.28, "speed": 1.02},
            "PGR": {"p_correct": 0.32, "speed": 0.96},
            "PGI": {"p_correct": 0.29, "speed": 1.00},
        },
        "noise": 0.09,
    },
}

# Structural per-LT profiles
def _shared_dims(lt1, lt2):
    return sum([lt1[0] == lt2[0], lt1[1] == lt2[1], lt1[2] == lt2[2]])

for _lt in LEARNING_TYPES:
    _cfg = {}
    for lt in LEARNING_TYPES:
        shared = _shared_dims(_lt, lt)
        if   shared == 3: p, spd = 0.78, 0.40
        elif shared == 2: p, spd = 0.63, 0.55
        elif shared == 1: p, spd = 0.52, 0.70
        else:             p, spd = 0.42, 0.82
        _cfg[lt] = {"p_correct": p, "speed": spd}
    PROFILES[f"student_{_lt.lower()}"] = {
        "label": f"{_lt}-type student",
        "desc":  f"Natural fit is {_lt}; others scale by shared dimensions.",
        "lt":    _cfg,
        "noise": 0.12,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class StudentSimulator:
    def __init__(self, profile_name: str, category: str = "Penggalang",
                 ai_sim: bool = False, api_key: str = ""):
        self.profile_name = profile_name
        self.profile      = PROFILES[profile_name]
        self.category     = category
        self.ai_sim       = ai_sim
        self.api_key      = api_key
        self.t_expected   = CATEGORY_CONFIG[category][0] / CATEGORY_CONFIG[category][1]
        self.question_num = 0

    def answer(self, lt: str) -> Tuple[bool, int, float]:
        self.question_num += 1
        lt_profile = self.profile["lt"][lt]
        p          = lt_profile["p_correct"]
        noise_std  = self.profile.get("noise", 0.18)

        if "improvement_rate" in self.profile:
            p = min(0.88, p + self.profile["improvement_rate"] * self.question_num)

        p = float(np.clip(p + random.gauss(0, noise_std), 0.05, 0.95))

        if self.ai_sim and self.api_key:
            p = self._ai_adjust_p(lt, p)

        is_correct = random.random() < p
        if is_correct:
            n_attempt = 1 if random.random() > (1 - p) * 0.5 else 2
        else:
            n_attempt = random.randint(2, 5)

        speed = lt_profile["speed"]
        ratio = float(np.clip(speed + random.gauss(0, noise_std * 0.6), 0.15, 1.40))
        t_ans = ratio * self.t_expected
        return is_correct, n_attempt, t_ans

    def _ai_adjust_p(self, lt: str, base_p: float) -> float:
        try:
            import urllib.request
            body = json.dumps({
                "model": "claude-haiku-4-5-20251001", "max_tokens": 100,
                "messages": [{"role": "user", "content": (
                    f"Student profile '{self.profile['label']}': {self.profile['desc']}. "
                    f"Answering LT '{lt}' question #{self.question_num}. "
                    f"Base p_correct: {base_p:.2f}. Return ONLY a float 0.0-1.0."
                )}]
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"Content-Type": "application/json",
                         "x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return max(0.05, min(0.95, float(json.loads(resp.read())["content"][0]["text"].strip())))
        except Exception:
            return base_p


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_transition(prev: Optional[str], curr: str) -> str:
    if prev is None:  return f"{lt_col(curr, curr)} (start)"
    if prev == curr:  return f"{lt_col(curr, curr)} ↺"
    return f"{lt_col(prev, prev)} → {lt_col(curr, curr)}"


def _print_qvalues(q_vals: Dict[str, float], agent, label: str):
    q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
    print(f"\n  {D}  ┌── Q-values @ {label} {'─'*36}{X}")
    max_q = max(q_vals.values()) or 0.001
    for lt, qv in q_sorted:
        lvl   = agent.mastery_levels[lt]
        b     = bar(max(0, qv) / max(0.001, max_q))
        arrow = f" {G}◀ best{X}" if lt == q_sorted[0][0] else ""
        print(f"  {D}  │ {lt_col(lt, lt):<14} [{b}] {qv:>+8.5f}  "
              f"L{lvl}/{MASTERY_LABELS[lvl][:14]}{arrow}{X}")
    w = agent.tracker
    print(f"  {D}  │ weights α={w.alpha:.3f} β={w.beta:.3f} γ={w.gamma:.3f}{X}")
    print(f"  {D}  └{'─'*50}{X}\n")


def _print_recommendation_change(step: Dict, agent, q_before: Dict, q_after: Dict,
                                  session_num: int):
    """
    Rich panel printed every time argmax Q flips.
    Shows: trigger, Q-values before/after, margins, mastery snapshot.
    """
    chg = step["lt_change_detail"]
    if chg is None:
        return
    frm   = chg["from_lt"] or "(none)"
    to    = chg["to_lt"]
    cause = f"[{chg['trigger']}]"
    n_changes = chg["change_index"]

    q_sorted_after = sorted(q_after.items(), key=lambda x: -x[1])
    margins_before = sorted(q_before.values(), reverse=True)
    margins_after  = sorted(q_after.values(),  reverse=True)
    gap_b = margins_before[0] - margins_before[1] if len(margins_before) > 1 else 0
    gap_a = margins_after[0]  - margins_after[1]  if len(margins_after)  > 1 else 0

    print(f"\n  {M}{B}  ╔══ LT RECOMMENDATION CHANGE #{n_changes}  "
          f"S{session_num} step#{step['step_num']} ══╗{X}")
    print(f"  {M}  ║  {lt_col(frm, frm)} → {lt_col(to, to)}   "
          f"cause={cause}  phase={chg['phase']}{X}")
    print(f"  {M}  ║  trigger: used={step['learning_type']}  "
          f"correct={step['is_correct']}  "
          f"reward={reward_col(step['reward'])}  "
          f"td_err={step['td_error']:+.5f}{X}")
    print(f"  {M}  ║  M={step['mastery_score']:.2f}  "
          f"P={step['performance']:.2f}  "
          f"E={step['engagement']:.2f}  "
          f"ε={step['epsilon']:.3f}{X}")
    print(f"  {D}  ╠─ Q-table ({'before':>8} → {'after':>8}  {'delta':>8}  bar){X}")

    for lt, qa in q_sorted_after:
        qb   = q_before.get(lt, 0.0)
        dq   = qa - qb
        b    = bar(max(0, qa) / max(0.001, max(q_after.values())), w=12)
        mark = (f" {G}◀ NEW top{X}" if lt == to else
                f" {Y}◀ was top{X}" if lt == frm else "")
        print(f"  {D}  ║  {lt_col(lt, lt):<14} {qb:>+10.5f} → {qa:>+10.5f}"
              f"  {dq:>+8.5f}  [{b}]{X}{mark}")

    print(f"  {D}  ║  Margin gap: before={gap_b:+.5f}  after={gap_a:+.5f}{X}")
    print(f"  {D}  ╚{'═'*58}{X}\n")


def _print_session_summary(agent, steps: List[Dict]):
    dist    = Counter(s["lt"] for s in steps)
    rewards = [s["reward"] for s in steps]
    seed_s  = [s for s in steps if s.get("phase") == "seeding"]
    free_s  = [s for s in steps if s.get("phase") == "free"]

    print(f"\n  {B}── Session Summary {'─'*50}{X}")
    print(f"  Total steps : {len(steps)}  │  "
          f"Avg reward: {np.mean(rewards):+.4f}  │  "
          f"Sum: {sum(rewards):+.4f}")
    if seed_s:
        print(f"  Seeding ({len(seed_s)} Q): avg reward {np.mean([s['reward'] for s in seed_s]):+.4f}")
    if free_s:
        print(f"  Free    ({len(free_s)} Q): avg reward {np.mean([s['reward'] for s in free_s]):+.4f}")

    changes = agent.recommendation_history
    if changes:
        print(f"\n  {B}LT Recommendation Changes ({len(changes)} total):{X}")
        for c in changes:
            frm = c["from_lt"] or "(none)"
            print(f"    #{c['change_index']:>2}  step#{c['step_num']:<3}  "
                  f"{lt_col(frm, frm):<14} → {lt_col(c['to_lt'], c['to_lt'])}  "
                  f"phase={c['phase']}  ε={c['epsilon']:.3f}  [{c['trigger']}]")
    else:
        print(f"\n  No recommendation changes yet (agent is still exploring).")

    print(f"\n  LT distribution:")
    for lt in LEARNING_TYPES:
        c   = dist.get(lt, 0)
        pct = c / len(steps) * 100 if steps else 0
        b   = "█" * c
        print(f"    {lt_col(lt, lt):<14} {b:<20} {c:>2} ({pct:.0f}%)")

    rec  = agent.recommend()
    best = rec["recommended_lt"]
    print(f"\n  {B}Recommendation → {lt_col(best, best)}  "
          f"Q={rec['per_type'][best]['q_value']:+.5f}{X}")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SESSION  (normal mode — uses RLAgent seeding internally)
# ─────────────────────────────────────────────────────────────────────────────

def run_session(
    session_id:     str,
    student:        StudentSimulator,
    n_questions:    int,
    registry:       SessionRegistry,
    verbose:        bool = True,
    session_num:    int  = 1,
    total_sessions: int  = 1,
    exploit_first:  bool = False,
    seeding_lt:     Optional[str] = None,
) -> List[Dict]:
    """
    Run one session.  If seeding_lt is supplied the agent's built-in
    seeding phase will force that LT for the first SEEDING_QUESTIONS calls.
    """
    agent = registry.get_agent(session_id, category=student.category,
                                seeding_lt=seeding_lt)
    steps: List[Dict] = []
    prev_lt: Optional[str] = None

    if verbose:
        seed_info = (f"  seeding={seeding_lt} ({SEEDING_QUESTIONS}Q)"
                     if seeding_lt else "")
        print(f"\n{B}{'═'*72}{X}")
        print(f"{B}  SESSION {session_num}/{total_sessions}  │  {session_id}  │  "
              f"profile={student.profile_name}  │  {n_questions} Q{seed_info}{X}")
        print(f"{B}{'═'*72}{X}")
        print(f"\n  {'#':<4} {'Phase':<8} {'Mode':<9} {'LT':<5} {'OK':<4}"
              f"{'n':<3} {'t_ans':>6}  "
              f"{'M':>5} {'P':>5} {'E':>5}  {'Reward':>8}  {'ε':>6}  Transition")
        print(f"  {'─'*80}")

    for q_num in range(1, n_questions + 1):
        # Exploit first question of non-first sessions (persistent mode)
        if exploit_first and q_num == 1 and session_num > 1:
            saved       = agent.epsilon
            agent.epsilon = 0.0
            lt, _       = agent.select_action()
            agent.epsilon = saved
            mode        = "exploit"
        else:
            lt, _  = agent.select_action()
            mode   = "seeding" if agent.step_log and agent.step_log[-1].get("phase") == "seeding" \
                     else ("explore" if random.random() < agent.epsilon else "exploit")

        q_before = agent.q_values()
        is_correct, n_attempt, t_ans = student.answer(lt)

        mastery_code = f"{agent.mastery_levels[lt]}{lt}"
        step = agent.record_response(
            learning_type=lt, is_correct=is_correct,
            n_attempt=n_attempt, t_answer_seconds=t_ans, mastery_code=mastery_code,
        )
        q_after  = agent.q_values()
        best_lt  = max(q_after, key=q_after.get)

        record = {
            "session_id":    session_id,  "session_num":   session_num,
            "question_num":  q_num,
            "global_q":      (session_num - 1) * n_questions + q_num,
            "profile":       student.profile_name,
            "phase":         step["phase"],
            "lt":            lt,          "prev_lt":       prev_lt,
            "mode":          mode,        "is_correct":    is_correct,
            "n_attempt":     n_attempt,   "t_answer":      t_ans,
            "t_expected":    step["t_expected"],
            "mastery_score": step["mastery_score"],
            "mastery_level": step["mastery_level"],
            "performance":   step["performance"],
            "engagement":    step["engagement"],
            "reward":        step["reward"],
            "delta_m":       step["delta_m"],
            "delta_p":       step["delta_p"],
            "td_error":      step["td_error"],
            "epsilon":       agent.epsilon,
            "mlr_refitted":  step["mlr_refitted"],
            "q_values":      dict(q_after),
            "best_lt":       best_lt,
            "lt_changed":    step["lt_changed"],
        }
        steps.append(record)

        if verbose:
            ok_sym   = f"{G}✓{X}" if is_correct else f"{R}✗{X}"
            phase_s  = f"{Y}seed{X}  " if step["phase"] == "seeding" else f"{C}free{X}  "
            if mode == "seeding": mode_s = f"{Y}forced {X}"
            elif mode == "exploit": mode_s = f"{C}exploit{X}"
            else: mode_s = f"{D}explore{X}"
            mlr_tag  = f" {Y}[MLR]{X}" if step["mlr_refitted"] else ""
            lt_s     = lt_col(lt, f"{lt:<5}")
            trans_s  = _fmt_transition(prev_lt, lt)

            print(f"  {q_num:<4} {phase_s} {mode_s:<16} {lt_s} {ok_sym:<4}"
                  f"{n_attempt:<3} {t_ans:>6.0f}  "
                  f"{step['mastery_score']:>5.2f} {step['performance']:>5.2f} "
                  f"{step['engagement']:>5.2f}  "
                  f"{reward_col(step['reward'])}  "
                  f"{agent.epsilon:>6.3f}  {trans_s}{mlr_tag}")

            if step["lt_changed"]:
                _print_recommendation_change(step, agent, q_before, q_after, session_num)
            if q_num % 5 == 0 or q_num == SEEDING_QUESTIONS:
                _print_qvalues(q_after, agent, f"Q{q_num}")

        prev_lt = lt

    if verbose:
        _print_session_summary(agent, steps)

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    profile_name:    str,
    n_sessions:      int,
    n_questions:     int,
    category:        str   = "Penggalang",
    seed:            Optional[int] = None,
    ai_sim:          bool  = False,
    api_key:         str   = "",
    verbose:         bool  = True,
    initial_epsilon: float = 0.20,
    isolated:        bool  = False,
    seeding_lt:      Optional[str] = None,
) -> Tuple[List[Dict], SessionRegistry]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    registry   = SessionRegistry()
    session_id = f"{profile_name}_student"
    all_steps: List[Dict] = []

    for s in range(1, n_sessions + 1):
        sid = f"{profile_name}_s{s:03d}" if isolated else session_id
        registry.get_agent(sid, category=category,
                           seeding_lt=seeding_lt).epsilon = initial_epsilon
        student = StudentSimulator(profile_name, category, ai_sim, api_key)
        steps   = run_session(
            session_id=sid, student=student, n_questions=n_questions,
            registry=registry, verbose=verbose,
            session_num=s, total_sessions=n_sessions,
            exploit_first=not isolated,
            seeding_lt=seeding_lt,
        )
        all_steps.extend(steps)

    if verbose:
        _print_global_summary(all_steps, n_sessions, n_questions)

    return all_steps, registry


def _print_global_summary(steps, n_sessions, n_questions):
    print(f"\n{B}{'═'*72}{X}")
    print(f"{B}  GLOBAL SUMMARY  —  {n_sessions} sessions × {n_questions} Q "
          f"= {len(steps)} total steps{X}")
    print(f"{B}{'═'*72}{X}")

    rewards  = [s["reward"] for s in steps]
    correct  = sum(1 for s in steps if s["is_correct"])
    dist     = Counter(s["lt"] for s in steps)

    print(f"\n  Total reward    : {sum(rewards):+.4f}")
    print(f"  Mean reward/step: {np.mean(rewards):+.4f}")
    print(f"  Correct answers : {correct}/{len(steps)} ({correct/len(steps)*100:.1f}%)")

    changes = [s for s in steps if s.get("lt_changed")]
    print(f"  LT changes      : {len(changes)}")

    print(f"\n  LT frequency:")
    for lt in LEARNING_TYPES:
        c   = dist.get(lt, 0)
        pct = c / len(steps) * 100 if steps else 0
        b   = "█" * int(pct / 3)
        print(f"    {lt_col(lt, lt):<14} {b:<25} {c:>4} ({pct:.1f}%)")

    seed_steps = [s for s in steps if s.get("phase") == "seeding"]
    free_steps = [s for s in steps if s.get("phase") == "free"]
    if seed_steps:
        print(f"\n  Seeding phase: {len(seed_steps)} Q  "
              f"avg reward={np.mean([s['reward'] for s in seed_steps]):+.4f}")
    if free_steps:
        print(f"  Free phase:    {len(free_steps)} Q  "
              f"avg reward={np.mean([s['reward'] for s in free_steps]):+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIXED-LT SESSION  (uses the agent's built-in seeding with 10Q default)
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_lt_session(
    session_id:     str,
    student:        StudentSimulator,
    n_questions:    int,
    registry:       SessionRegistry,
    fixed_lt:       str,
    seed_questions: int  = SEEDING_QUESTIONS,
    verbose:        bool = True,
    session_num:    int  = 1,
    total_sessions: int  = 1,
) -> List[Dict]:
    """One session seeded on fixed_lt for seed_questions, then free."""
    if fixed_lt not in LEARNING_TYPES:
        raise ValueError(f"fixed_lt '{fixed_lt}' not in {LEARNING_TYPES}")

    return run_session(
        session_id=session_id, student=student, n_questions=n_questions,
        registry=registry, verbose=verbose,
        session_num=session_num, total_sessions=total_sessions,
        seeding_lt=fixed_lt,
    )


def run_fixed_lt_simulation(
    fixed_lt: str, profile_name: str, n_sessions: int, n_questions: int,
    seed_questions: int = SEEDING_QUESTIONS, category: str = "Penggalang",
    seed: Optional[int] = None, ai_sim: bool = False, api_key: str = "",
    verbose: bool = True,
) -> Tuple[List[Dict], SessionRegistry]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    registry   = SessionRegistry()
    session_id = f"fixed_{fixed_lt}_{profile_name}_student"
    all_steps: List[Dict] = []

    for s in range(1, n_sessions + 1):
        student = StudentSimulator(profile_name, category, ai_sim, api_key)
        steps   = run_fixed_lt_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, fixed_lt=fixed_lt, seed_questions=seed_questions,
            verbose=verbose, session_num=s, total_sessions=n_sessions,
        )
        all_steps.extend(steps)

    if verbose:
        _print_global_summary(all_steps, n_sessions, n_questions)

    return all_steps, registry


# ─────────────────────────────────────────────────────────────────────────────
# STORY MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_story_simulation(
    assigned_lt: str, true_lt: str,
    profile_name: str = "pai_misfit",
    n_seed_sessions: int = 2, n_explore_sessions: int = 2,
    n_converge_sessions: int = 3, n_questions: int = 15,
    seed_questions: int = SEEDING_QUESTIONS, category: str = "Penggalang",
    seed: Optional[int] = None, verbose: bool = True,
) -> Tuple[List[Dict], SessionRegistry]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    registry   = SessionRegistry()
    session_id = f"story_{assigned_lt}_{profile_name}"
    all_steps: List[Dict] = []
    session_num = 0
    n_total = n_seed_sessions + n_explore_sessions + n_converge_sessions

    def banner(phase, desc, s1, s2):
        if not verbose: return
        c = {1: Y, 2: M, 3: G}.get(phase, B)
        print(f"\n{c}{B}{'#'*72}{X}")
        print(f"{c}{B}  PHASE {phase}  (Sessions {s1}–{s2})  {desc}{X}")
        print(f"{c}{B}{'#'*72}{X}")

    def phase_summary(steps_p, name, agent):
        if not verbose or not steps_p: return
        rewards = [s["reward"] for s in steps_p]
        correct = sum(1 for s in steps_p if s["is_correct"])
        dist    = Counter(s["lt"] for s in steps_p)
        print(f"\n  {B}Phase {name} summary:{X}")
        print(f"  {len(steps_p)} Q  avg={np.mean(rewards):+.4f}  "
              f"correct={correct}/{len(steps_p)}")
        for lt, c in dist.most_common():
            tag = " ← assigned" if lt == assigned_lt else \
                  " ← TRUE fit" if lt == true_lt else ""
            print(f"    {lt_col(lt, lt):<14} {'|'*c}  ({c}){tag}")

    # Phase 1
    banner(1, f"Student forced onto {assigned_lt} — wrong fit.", 1, n_seed_sessions)
    phase1 = []
    for _ in range(n_seed_sessions):
        session_num += 1
        student = StudentSimulator(profile_name, category)
        steps   = run_fixed_lt_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, fixed_lt=assigned_lt, seed_questions=seed_questions,
            verbose=verbose, session_num=session_num, total_sessions=n_total,
        )
        for s in steps: s["story_phase"] = "assigned"
        phase1.extend(steps); all_steps.extend(steps)
    agent = registry.get_agent(session_id, category=category)
    phase_summary(phase1, "1 ASSIGNED", agent)

    # Phase 2
    banner(2, f"ε=0.50 — aggressive exploration. Looking for {true_lt}.",
           n_seed_sessions + 1, n_seed_sessions + n_explore_sessions)
    agent.epsilon = 0.50
    phase2 = []
    for _ in range(n_explore_sessions):
        session_num += 1
        student = StudentSimulator(profile_name, category)
        steps   = run_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, verbose=verbose,
            session_num=session_num, total_sessions=n_total,
        )
        for s in steps: s["story_phase"] = "explore"
        phase2.extend(steps); all_steps.extend(steps)
    phase_summary(phase2, "2 EXPLORE", agent)

    # Phase 3
    banner(3, f"ε decays normally. Agent exploits {true_lt}.",
           n_seed_sessions + n_explore_sessions + 1, n_total)
    phase3 = []
    for _ in range(n_converge_sessions):
        session_num += 1
        student = StudentSimulator(profile_name, category)
        steps   = run_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, verbose=verbose,
            session_num=session_num, total_sessions=n_total,
        )
        for s in steps: s["story_phase"] = "converge"
        phase3.extend(steps); all_steps.extend(steps)
    phase_summary(phase3, "3 CONVERGE", agent)

    if verbose:
        _print_global_summary(all_steps, n_total, n_questions)

    return all_steps, registry


# ─────────────────────────────────────────────────────────────────────────────
# PER-LT AND CROSS-LT (unchanged logic, same API)
# ─────────────────────────────────────────────────────────────────────────────

def run_per_lt_simulation(n_questions_per_lt, profile_name, category="Penggalang",
                           seed=None, verbose=True):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    results = {}
    for lt in LEARNING_TYPES:
        registry = SessionRegistry()
        agent    = registry.get_agent(f"perlt_{lt}_{profile_name}", category=category)
        student  = StudentSimulator(profile_name, category)
        steps    = []
        if verbose:
            print(f"  [{lt}] {n_questions_per_lt} Q...", flush=True)
        for q in range(1, n_questions_per_lt + 1):
            is_correct, n_attempt, t_ans = student.answer(lt)
            mastery_code = f"{agent.mastery_levels[lt]}{lt}"
            step = agent.record_response(lt, is_correct, n_attempt, t_ans, mastery_code)
            steps.append({
                "lt": lt, "question_num": q,
                "is_correct": is_correct, "n_attempt": n_attempt,
                "t_answer": t_ans, "t_expected": step["t_expected"],
                "mastery_score": step["mastery_score"],
                "mastery_level": step["mastery_level"],
                "performance": step["performance"],
                "engagement":  step["engagement"],
                "reward":      step["reward"],
                "delta_m":     step["delta_m"],
                "mlr_refitted": step["mlr_refitted"],
                "q_value_self": agent.q_values()[lt],
                "epsilon":     agent.epsilon,
            })
        results[lt] = steps
    return results


def run_cross_lt_simulation(n_questions=200, category="Penggalang",
                              seed=None, verbose=True):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    raw = {slt: {} for slt in LEARNING_TYPES}
    for s_idx, student_lt in enumerate(LEARNING_TYPES):
        profile_name = f"student_{student_lt.lower()}"
        if verbose:
            print(f"  Student type {student_lt} ({s_idx+1}/8):", flush=True)
        for q_lt in LEARNING_TYPES:
            registry = SessionRegistry()
            agent    = registry.get_agent(f"cross_{student_lt}_{q_lt}", category=category)
            student  = StudentSimulator(profile_name, category)
            rewards  = []
            for q in range(1, n_questions + 1):
                is_correct, n_attempt, t_ans = student.answer(q_lt)
                mc   = f"{agent.mastery_levels[q_lt]}{q_lt}"
                step = agent.record_response(q_lt, is_correct, n_attempt, t_ans, mc)
                rewards.append(step["reward"])
            raw[student_lt][q_lt] = rewards
            if verbose:
                avg = np.mean(rewards)
                b   = "█" * max(0, int(avg / 0.004)) + "░" * max(0, 20 - int(avg / 0.004))
                print(f"    vs {q_lt}: [{b}] {avg:+.4f}", flush=True)

    matrix     = np.zeros((8, 8))
    std_matrix = np.zeros((8, 8))
    for i, slt in enumerate(LEARNING_TYPES):
        for j, qlt in enumerate(LEARNING_TYPES):
            matrix[i, j]     = np.mean(raw[slt][qlt])
            std_matrix[i, j] = np.std(raw[slt][qlt])
    return {"matrix": matrix, "std_matrix": std_matrix, "raw": raw,
            "lt_list": LEARNING_TYPES, "n_questions": n_questions}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _shade_seeding(ax, steps, n_questions_per_session, seed_n, alpha=0.25):
    """Shade seeding windows (first seed_n questions of each session) in yellow."""
    sessions = sorted(set(s["session_num"] for s in steps if "session_num" in s))
    for sn in sessions:
        x0 = (sn - 1) * n_questions_per_session + 0.5
        x1 = (sn - 1) * n_questions_per_session + seed_n + 0.5
        ax.axvspan(x0, x1, color=SEED_SHADE, alpha=alpha, zorder=0)
    for sn in sessions[1:]:
        ax.axvline((sn - 1) * n_questions_per_session, color="grey",
                   lw=0.7, ls=":", alpha=0.5)


def _rolling(ys, window):
    if len(ys) < window:
        return list(range(len(ys))), ys
    r  = np.convolve(ys, np.ones(window) / window, mode="valid")
    xs = list(range(window - 1, len(ys)))
    return xs, r.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Reward time series  (per-LT subplots + overlay)
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(all_steps: List[Dict], out_dir: str = "plots", profile: str = ""):
    """Per-LT reward time series (8 subplots) + combined overlay."""
    if not HAS_PLOT:
        print("[skip] matplotlib not available"); return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    sessions  = sorted(set(s.get("session_num", 1) for s in all_steps))
    n_q_per_s = max((s.get("question_num", 1) for s in all_steps
                     if s.get("session_num") == sessions[0]), default=15)
    seed_n    = SEEDING_QUESTIONS

    # ── Per-LT subplots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True, sharey=True)
    for i, lt in enumerate(LEARNING_TYPES):
        ax  = axes.flatten()[i]
        col = MPL_COLOURS[lt]
        xs  = [s["global_q"] for s in all_steps if s["lt"] == lt]
        ys  = [s["reward"]   for s in all_steps if s["lt"] == lt]

        _shade_seeding(ax, all_steps, n_q_per_s, seed_n)

        if not xs:
            ax.set_title(f"{lt}  (not selected)", fontsize=10, color="grey")
            ax.axis("off"); continue

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

    for ax in axes[:, 0]: ax.set_ylabel("Reward")
    for ax in axes[-1, :]: ax.set_xlabel("Question (global)")
    seed_patch = mpatches.Patch(color=SEED_SHADE, label=f"Seeding phase ({seed_n} Q)")
    fig.legend(handles=[seed_patch], loc="upper right", fontsize=9)
    fig.suptitle(f"Reward Time Series per LT  —  profile: {profile or 'all'}\n"
                 f"(yellow = seeding phase,  o=correct  x=wrong,  line=trend)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, f"reward_timeseries{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")

    # ── Overlay ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    _shade_seeding(ax, all_steps, n_q_per_s, seed_n)
    for lt in LEARNING_TYPES:
        xs = [s["global_q"] for s in all_steps if s["lt"] == lt]
        ys = [s["reward"]   for s in all_steps if s["lt"] == lt]
        if len(xs) < 3: continue
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
    out = os.path.join(out_dir, f"reward_trend_overlay{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")

    # ── LT transition heatmap ────────────────────────────────────────────────
    mat = np.zeros((len(LEARNING_TYPES), len(LEARNING_TYPES)))
    for s in all_steps:
        if s["prev_lt"] is not None:
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
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color="white" if v > mat.max() * 0.5 else "black")
    plt.tight_layout()
    out = os.path.join(out_dir, f"lt_transition_heatmap{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Q-value evolution with LT-change event markers
# ─────────────────────────────────────────────────────────────────────────────

def make_qvalue_plots(all_steps: List[Dict], out_dir: str = "plots",
                      profile: str = "", registry: Optional[SessionRegistry] = None):
    """
    Q-value of every LT over time.  Vertical dashed lines mark every
    recommendation change event.  Seeding window shaded yellow.
    """
    if not HAS_PLOT: return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    if not all_steps or "q_values" not in all_steps[0]:
        print("  [skip] q_values not in step log"); return

    sessions  = sorted(set(s.get("session_num", 1) for s in all_steps))
    n_q_per_s = max((s.get("question_num", 1) for s in all_steps
                     if s.get("session_num") == sessions[0]), default=15)
    xs = [s["global_q"] for s in all_steps]

    # Gather LT-change events from step log
    change_qs   = [s["global_q"] for s in all_steps if s.get("lt_changed")]
    change_froms = [s["q_values"].get(s.get("prev_lt", ""), 0) for s in all_steps
                    if s.get("lt_changed")]

    fig, ax = plt.subplots(figsize=(16, 6))
    _shade_seeding(ax, all_steps, n_q_per_s, SEEDING_QUESTIONS)

    for lt in LEARNING_TYPES:
        qs = [s["q_values"].get(lt, 0) for s in all_steps]
        ax.plot(xs, qs, color=MPL_COLOURS[lt], lw=1.8, label=lt, alpha=0.85)
        ax.annotate(lt, xy=(xs[-1], qs[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8,
                    color=MPL_COLOURS[lt], va="center")

    # Mark change events
    for q in change_qs:
        ax.axvline(q, color="red", lw=1.0, ls="--", alpha=0.55, zorder=5)

    if change_qs:
        ax.scatter(change_qs, [0] * len(change_qs), marker="^",
                   color="red", s=60, zorder=6, label="recommendation change")

    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Question (global)"); ax.set_ylabel("Q-value")
    ax.set_title(
        f"Q-value Evolution  |  profile: {profile or 'all'}\n"
        f"(yellow = seeding phase,  red ▲ = recommendation changes)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=5)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(out_dir, f"qvalue_evolution{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — LT change timeline panel
# ─────────────────────────────────────────────────────────────────────────────

def make_lt_change_plot(all_steps: List[Dict], out_dir: str = "plots",
                         profile: str = ""):
    """
    Horizontal swim-lane chart showing which LT is the current best at every
    question, with change events annotated.
    """
    if not HAS_PLOT: return
    if not all_steps or "q_values" not in all_steps[0]: return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    sessions  = sorted(set(s.get("session_num", 1) for s in all_steps))
    n_q_per_s = max((s.get("question_num", 1) for s in all_steps
                     if s.get("session_num") == sessions[0]), default=15)

    # Best LT at each step = argmax q_values
    best_seq = [max(s["q_values"], key=s["q_values"].get) for s in all_steps]
    xs       = [s["global_q"] for s in all_steps]

    lt_y = {lt: i for i, lt in enumerate(LEARNING_TYPES)}
    fig, ax = plt.subplots(figsize=(16, 5))
    _shade_seeding(ax, all_steps, n_q_per_s, SEEDING_QUESTIONS)

    # Draw horizontal band per LT showing when it was the best recommendation
    prev_lt = None
    seg_start = 0
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

    # Mark change events
    change_xs = [xs[i] for i in range(1, len(best_seq))
                 if best_seq[i] != best_seq[i - 1]]
    for cq in change_xs:
        ax.axvline(cq - 0.5, color="red", lw=1.2, ls="--", alpha=0.6, zorder=5)

    ax.set_yticks(list(lt_y.values()))
    ax.set_yticklabels(list(lt_y.keys()))
    for lt, y in lt_y.items():
        ax.get_yticklabels()[y].set_color(MPL_COLOURS[lt])
    ax.set_xlabel("Question (global)")
    ax.set_title(
        f"LT Recommendation Timeline  |  profile: {profile or 'all'}\n"
        f"(each row = one LT,  filled = 'best recommendation' at that question,  "
        f"red dashes = change events,  yellow = seeding)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    out = os.path.join(out_dir, f"lt_recommendation_timeline{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Seeding vs free phase LT usage comparison
# ─────────────────────────────────────────────────────────────────────────────

def make_phase_bar_plot(all_steps: List[Dict], out_dir: str = "plots",
                         profile: str = ""):
    """
    Grouped bar chart: LT usage % in seeding phase vs free phase.
    Makes the effect of seeding visible at a glance.
    """
    if not HAS_PLOT: return
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

    x = np.arange(len(LEARNING_TYPES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - w / 2, [seed_pct[lt] for lt in LEARNING_TYPES], w,
                label=f"Seeding  ({len(seed_steps)} Q)", color="#ffe082",
                edgecolor="grey", linewidth=0.7, alpha=0.9)
    b2 = ax.bar(x + w / 2, [free_pct[lt] for lt in LEARNING_TYPES], w,
                label=f"Free     ({len(free_steps)} Q)", color="#90caf9",
                edgecolor="grey", linewidth=0.7, alpha=0.9)

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
        f"(seeding = first {SEEDING_QUESTIONS} Q/session forced on assigned LT,  "
        f"free = epsilon-greedy)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, f"seeding_vs_free_bar{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Mastery / Performance / Engagement per LT
# ─────────────────────────────────────────────────────────────────────────────

def make_mpe_plots(all_steps: List[Dict], out_dir: str = "plots",
                   profile: str = ""):
    """
    2×4 grid: one column per LT, three sub-rows (M, P, E) stacked.
    Shows how each metric evolves over questions answered in that LT.
    """
    if not HAS_PLOT: return
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{profile}" if profile else ""

    fig = plt.figure(figsize=(20, 10))
    metrics = [("mastery_score", "Mastery (M)", "solid"),
               ("performance",   "Performance (P)", "dashed"),
               ("engagement",    "Engagement (E)", "dotted")]

    gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.55, wspace=0.35)

    for col, lt in enumerate(LEARNING_TYPES):
        lt_steps = [s for s in all_steps if s["lt"] == lt]
        if not lt_steps:
            continue
        xs = list(range(1, len(lt_steps) + 1))
        col_colour = MPL_COLOURS[lt]

        for row, (key, label, ls) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            ys = [s[key] for s in lt_steps]
            ax.plot(xs, ys, color=col_colour, lw=1.6, ls=ls, alpha=0.85)
            if len(ys) >= 4:
                ri, rv = _rolling(ys, max(3, len(ys) // 5))
                ax.plot([xs[r] for r in ri], rv, color=col_colour, lw=2.5, alpha=0.95)
            ax.axhline(np.mean(ys), color="grey", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(lt, fontsize=11, color=col_colour, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 2:
                ax.set_xlabel("Q#", fontsize=8)

    fig.suptitle(
        f"Mastery / Performance / Engagement per LT  |  profile: {profile or 'all'}\n"
        f"(thin = per-step,  thick = rolling mean,  dashed grey = overall mean)",
        fontsize=12, y=1.01)
    out = os.path.join(out_dir, f"mpe_per_lt{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-LT / CROSS-LT PLOT WRAPPERS  (same API as before, calls new helpers)
# ─────────────────────────────────────────────────────────────────────────────

def make_per_lt_plots(results, profile, out_dir="plots"):
    if not HAS_PLOT: return
    os.makedirs(out_dir, exist_ok=True)

    # Flatten results for shared utilities
    all_flat = [s for steps in results.values() for s in steps]

    # Time-series
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True, sharey=True)
    for i, lt in enumerate(LEARNING_TYPES):
        ax = axes.flatten()[i]; col = MPL_COLOURS[lt]
        steps = results.get(lt, [])
        if not steps: ax.axis("off"); continue
        xs  = [s["question_num"] for s in steps]
        ys  = [s["reward"] for s in steps]
        c_x = [s["question_num"] for s in steps if s["is_correct"]]
        c_y = [s["reward"] for s in steps if s["is_correct"]]
        w_x = [s["question_num"] for s in steps if not s["is_correct"]]
        w_y = [s["reward"] for s in steps if not s["is_correct"]]
        ax.scatter(c_x, c_y, color=col, s=30, alpha=0.7,  marker="o", zorder=3)
        ax.scatter(w_x, w_y, color=col, s=30, alpha=0.35, marker="x", zorder=3)
        if len(xs) >= 5:
            ri, rv = _rolling(ys, max(3, len(ys) // 8))
            ax.plot([xs[r] for r in ri], rv, color=col, lw=2.5, zorder=4)
        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"{lt}  avg={np.mean(ys):+.3f}  n={len(xs)}",
                     fontsize=11, color=col, fontweight="bold")
        ax.grid(True, alpha=0.2)
    for ax in axes[:, 0]: ax.set_ylabel("Reward")
    for ax in axes[-1, :]: ax.set_xlabel("Question #")
    fig.suptitle(f"Per-LT Reward Time Series  |  Profile: {profile}", fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, f"per_lt_timeseries_{profile}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")

    # Avg reward ranking
    avgs    = {lt: np.mean([s["reward"] for s in steps])
               for lt, steps in results.items() if steps}
    lt_rank = sorted(avgs, key=avgs.get, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = [MPL_COLOURS[lt] for lt in lt_rank]
    vals = [avgs[lt] for lt in lt_rank]
    bars = ax.bar(lt_rank, vals, color=cols, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold",
                va="bottom" if v >= 0 else "top")
    ax.set_xlabel("Learning Type"); ax.set_ylabel("Average Reward")
    ax.set_title(f"LT Ranking by Average Reward  |  Profile: {profile}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    out = os.path.join(out_dir, f"per_lt_ranking_{profile}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")

    # Mastery progression
    fig, ax = plt.subplots(figsize=(16, 5))
    for lt in LEARNING_TYPES:
        steps = results.get(lt, [])
        if not steps: continue
        ax.plot([s["question_num"] for s in steps], [s["mastery_level"] for s in steps],
                color=MPL_COLOURS[lt], lw=2.0, label=lt, drawstyle="steps-post", alpha=0.85)
    ax.set_xlabel("Question #"); ax.set_ylabel("Mastery Level (1–6)")
    ax.set_yticks(range(1, 7))
    ax.set_title(f"Mastery Level Progression per LT  |  Profile: {profile}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, f"per_lt_mastery_{profile}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


def make_cross_lt_plots(result: Dict, out_dir: str = "plots"):
    if not HAS_PLOT: return
    os.makedirs(out_dir, exist_ok=True)
    matrix  = result["matrix"]
    lt_list = result["lt_list"]
    n_q     = result["n_questions"]
    corr    = np.corrcoef(matrix.T)

    # Reward heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Average reward")
    ax.set_xticks(range(8)); ax.set_xticklabels(lt_list, fontsize=11)
    ax.set_yticks(range(8)); ax.set_yticklabels(lt_list, fontsize=11)
    ax.set_xlabel("Question LT"); ax.set_ylabel("Student type (natural LT)")
    ax.set_title(f"Cross-LT Reward Matrix  ({n_q} Q per cell)\n"
                 f"Diagonal = student answers their own LT", fontsize=12)
    for i in range(8):
        for j in range(8):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > matrix.mean() + matrix.std() else "black",
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="blue", lw=2.5, zorder=5))
    plt.tight_layout()
    out = os.path.join(out_dir, "cross_lt_reward_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(8)); ax.set_xticklabels(lt_list, fontsize=11)
    ax.set_yticks(range(8)); ax.set_yticklabels(lt_list, fontsize=11)
    ax.set_title("LT Correlation Matrix  (Pearson r)", fontsize=12)
    for i in range(8):
        for j in range(8):
            r = corr[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(r) > 0.7 else "black",
                    fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    out = os.path.join(out_dir, "cross_lt_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [plot] {out}")


def make_fixed_lt_plots(all_steps, fixed_lt, profile, out_dir="plots"):
    """Reuse the enhanced make_plots + make_qvalue_plots + make_phase_bar_plot."""
    if not HAS_PLOT: return
    make_plots(all_steps, out_dir, profile=f"fixed_{fixed_lt}_{profile}")
    make_qvalue_plots(all_steps, out_dir, profile=f"fixed_{fixed_lt}_{profile}")
    make_phase_bar_plot(all_steps, out_dir, profile=f"fixed_{fixed_lt}_{profile}")
    make_mpe_plots(all_steps, out_dir, profile=f"fixed_{fixed_lt}_{profile}")


def make_story_plots(all_steps, assigned_lt, true_lt, profile, out_dir="plots"):
    """Reuse the enhanced make_plots + qvalue + timeline for story mode."""
    if not HAS_PLOT: return
    tag = f"story_{assigned_lt}_to_{true_lt}_{profile}"
    make_plots(all_steps, out_dir, profile=tag)
    make_qvalue_plots(all_steps, out_dir, profile=tag)
    make_lt_change_plot(all_steps, out_dir, profile=tag)
    make_mpe_plots(all_steps, out_dir, profile=tag)


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_json(steps, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"steps_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [export] {path}  ({len(steps)} steps)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PLS RL Simulator v4")
    parser.add_argument("--sessions",         type=int,  default=1)
    parser.add_argument("--questions",        type=int,  default=15)
    parser.add_argument("--profile",          type=str,  default="random",
                        choices=list(PROFILES.keys()))
    parser.add_argument("--all-profiles",     action="store_true")
    parser.add_argument("--fixed-lt",         type=str,  default=None,
                        choices=LEARNING_TYPES)
    parser.add_argument("--seed-questions",   type=int,  default=SEEDING_QUESTIONS,
                        help=f"Seeding phase length (default: {SEEDING_QUESTIONS})")
    parser.add_argument("--per-lt",           type=int,  default=None, metavar="N")
    parser.add_argument("--story",            action="store_true")
    parser.add_argument("--assigned-lt",      type=str,  default="PAI",
                        choices=LEARNING_TYPES)
    parser.add_argument("--true-lt",          type=str,  default="TAR",
                        choices=LEARNING_TYPES)
    parser.add_argument("--seed-sessions",    type=int,  default=2)
    parser.add_argument("--explore-sessions", type=int,  default=2)
    parser.add_argument("--converge-sessions",type=int,  default=3)
    parser.add_argument("--cross-lt",         type=int,  default=None, metavar="N")
    parser.add_argument("--isolated",         action="store_true")
    parser.add_argument("--initial-epsilon",  type=float, default=0.20)
    parser.add_argument("--seeding-lt",       type=str,  default=None,
                        choices=LEARNING_TYPES,
                        help="Force seeding on this LT for first "
                             f"{SEEDING_QUESTIONS} questions")
    parser.add_argument("--category",         type=str,  default="Penggalang",
                        choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--seed",             type=int,  default=None)
    parser.add_argument("--quiet",            action="store_true")
    parser.add_argument("--no-plots",         action="store_true")
    parser.add_argument("--out",              type=str,  default="plots")
    parser.add_argument("--export",           action="store_true")
    parser.add_argument("--ai-sim",           action="store_true")
    parser.add_argument("--api-key",          type=str,  default="")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if args.ai_sim and not api_key:
        print(f"{Y}[warn] No API key — falling back to heuristic.{X}")
        args.ai_sim = False

    verbose = not args.quiet

    print(f"\n{B}{'='*70}{X}")
    print(f"{B}  PLS RL Session Simulator  v4{X}")
    print(f"{B}  Seeding phase: {args.seed_questions} questions  |  "
          f"8 LTs  |  epsilon-greedy bandit{X}")
    print(f"{B}{'='*70}{X}")

    # ── FIXED-LT ─────────────────────────────────────────────────────────────
    if args.fixed_lt:
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
            print(f"\n{B}  Generating plots…{X}")
            make_fixed_lt_plots(steps, args.fixed_lt, args.profile, args.out)
        if args.export:
            export_json(steps, args.out, f"fixed_{args.fixed_lt}_{args.profile}")

    # ── PER-LT ───────────────────────────────────────────────────────────────
    elif args.per_lt:
        print(f"  Mode: Per-LT  |  {args.per_lt} Q each × 8 LTs  "
              f"|  profile={args.profile}")
        results = run_per_lt_simulation(
            n_questions_per_lt=args.per_lt, profile_name=args.profile,
            category=args.category, seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            print(f"\n{B}  Generating plots…{X}")
            make_per_lt_plots(results, args.profile, args.out)
            flat = [s for steps in results.values() for s in steps]
            make_mpe_plots(flat, args.out, profile=args.profile)
        if args.export:
            export_json([s for v in results.values() for s in v],
                        args.out, f"per_lt_{args.profile}")

    # ── CROSS-LT ─────────────────────────────────────────────────────────────
    elif args.cross_lt:
        print(f"  Mode: Cross-LT  |  {args.cross_lt} Q/cell  "
              f"|  8×8={args.cross_lt*64} total Q")
        result = run_cross_lt_simulation(
            n_questions=args.cross_lt, category=args.category,
            seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            print(f"\n{B}  Generating plots…{X}")
            make_cross_lt_plots(result, args.out)

    # ── STORY ────────────────────────────────────────────────────────────────
    elif args.story:
        assigned = args.assigned_lt; true_lt = args.true_lt
        profile  = args.profile if args.profile != "random" else "pai_misfit"
        n_total  = args.seed_sessions + args.explore_sessions + args.converge_sessions
        print(f"  Mode: Story  |  assigned={assigned}  true={true_lt}  "
              f"|  sessions={n_total}×{args.questions}Q")
        steps, _ = run_story_simulation(
            assigned_lt=assigned, true_lt=true_lt, profile_name=profile,
            n_seed_sessions=args.seed_sessions,
            n_explore_sessions=args.explore_sessions,
            n_converge_sessions=args.converge_sessions,
            n_questions=args.questions, seed_questions=args.seed_questions,
            category=args.category, seed=args.seed, verbose=verbose,
        )
        if not args.no_plots:
            print(f"\n{B}  Generating plots…{X}")
            make_story_plots(steps, assigned, true_lt, profile, args.out)
        if args.export:
            export_json(steps, args.out, f"story_{assigned}_{true_lt}")

    # ── NORMAL ───────────────────────────────────────────────────────────────
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
                isolated=args.isolated,
                seeding_lt=args.seeding_lt,
            )
            all_combined.extend(steps)
            if not args.no_plots:
                print(f"\n{B}  Generating plots…{X}")
                make_plots(steps, out_dir=args.out, profile=profile_name)
                make_qvalue_plots(steps, out_dir=args.out, profile=profile_name,
                                  registry=reg)
                make_lt_change_plot(steps, out_dir=args.out, profile=profile_name)
                make_phase_bar_plot(steps, out_dir=args.out, profile=profile_name)
                make_mpe_plots(steps, out_dir=args.out, profile=profile_name)
            if args.export:
                export_json(steps, args.out, profile_name)

        if args.all_profiles and not args.no_plots and len(profiles_to_run) > 1:
            print(f"\n{B}  Combined plots across all profiles…{X}")
            make_plots(all_combined, out_dir=args.out, profile="all_profiles")
            if args.export:
                export_json(all_combined, args.out, "all_profiles")

    print(f"\n{B}{'='*70}{X}")
    print(f"{G}{B}  Simulation complete.{X}")
    if not args.no_plots and HAS_PLOT:
        print(f"  Plots saved → {os.path.abspath(args.out)}/")
    print(f"{B}{'='*70}{X}\n")


if __name__ == "__main__":
    main()
