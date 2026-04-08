"""
simulate_rl/runners.py
───────────────────────
All simulation runner functions.  Each runner returns a list of step dicts
(and optionally a SessionRegistry) that can be passed to the plot functions.

Modes
-----
run_session            — one session, core loop
run_simulation         — multi-session normal mode
run_fixed_lt_session   — one session locked to a single LT then free
run_fixed_lt_simulation
run_story_simulation   — 3-phase narrative (assigned → explore → converge)
run_per_lt_simulation  — isolated per-LT benchmark
run_cross_lt_simulation— 8×8 student-type × question-LT matrix
run_correlation_simulation — PAR/TAR positive-correlation demo
run_single_line_simulation — one student, continuous reward line
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from pedagogy_selector import SessionRegistry, SEEDING_QUESTIONS
from rl_metrics import LEARNING_TYPES
from simulate_rl.profiles import (
    PROFILES, StudentSimulator, PROFILE_HOME_LT,
    G, R, Y, C, M, B, D, X,
    lt_col, reward_col,
)
from simulate_rl.terminal import (
    fmt_transition, print_qvalues, print_recommendation_change,
    print_session_summary, print_global_summary,
)


# ── Core session loop ─────────────────────────────────────────────────────────

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
    Run one session through the full RL loop.

    If seeding_lt is set, the agent forces that LT for the first
    SEEDING_QUESTIONS questions before switching to epsilon-greedy.
    """
    agent = registry.get_agent(
        session_id, category=student.category, seeding_lt=seeding_lt
    )
    steps:   List[Dict] = []
    prev_lt: Optional[str] = None

    if verbose:
        seed_info = (
            f"  seeding={seeding_lt} ({SEEDING_QUESTIONS}Q)" if seeding_lt else ""
        )
        print(f"\n{B}{'═'*72}{X}")
        print(
            f"{B}  SESSION {session_num}/{total_sessions}  │  {session_id}  │  "
            f"profile={student.profile_name}  │  {n_questions} Q{seed_info}{X}"
        )
        print(f"{B}{'═'*72}{X}")
        print(
            f"\n  {'#':<4} {'Phase':<8} {'Mode':<9} {'LT':<5} {'OK':<4}"
            f"{'n':<3} {'t_ans':>6}  "
            f"{'M':>5} {'P':>5} {'E':>5}  {'Reward':>8}  {'ε':>6}  Transition"
        )
        print(f"  {'─'*80}")

    for q_num in range(1, n_questions + 1):
        # Exploit first question of non-first sessions in persistent mode
        if exploit_first and q_num == 1 and session_num > 1:
            saved         = agent.epsilon
            agent.epsilon = 0.0
            lt, _         = agent.select_action()
            agent.epsilon = saved
            mode          = "exploit"
        else:
            lt, _ = agent.select_action()
            mode  = (
                "seeding"  if agent.step_log and agent.step_log[-1].get("phase") == "seeding"
                else "explore" if random.random() < agent.epsilon
                else "exploit"
            )

        q_before = agent.q_values()
        is_correct, n_attempt, t_ans = student.answer(lt)

        mastery_code = f"{agent.mastery_levels[lt]}{lt}"
        step = agent.record_response(
            learning_type=lt, is_correct=is_correct,
            n_attempt=n_attempt, t_answer_seconds=t_ans,
            mastery_code=mastery_code,
        )
        q_after = agent.q_values()

        record = {
            "session_id":    session_id,  "session_num":   session_num,
            "question_num":  q_num,
            "global_q":      (session_num - 1) * n_questions + q_num,
            "profile":       student.profile_name,
            "phase":         step["phase"],
            "lt":            lt,           "prev_lt":       prev_lt,
            "mode":          mode,         "is_correct":    is_correct,
            "n_attempt":     n_attempt,    "t_answer":      t_ans,
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
            "best_lt":       max(q_after, key=q_after.get),
            "lt_changed":    step["lt_changed"],
            "lt_change_detail": step.get("lt_change_detail"),
        }
        steps.append(record)

        if verbose:
            ok_sym  = f"{G}✓{X}" if is_correct else f"{R}✗{X}"
            phase_s = f"{Y}seed{X}  " if step["phase"] == "seeding" else f"{C}free{X}  "
            mode_s  = (
                f"{Y}forced {X}"  if mode == "seeding"
                else f"{C}exploit{X}" if mode == "exploit"
                else f"{D}explore{X}"
            )
            mlr_tag = f" {Y}[MLR]{X}" if step["mlr_refitted"] else ""

            print(
                f"  {q_num:<4} {phase_s} {mode_s:<16} "
                f"{lt_col(lt, lt):<5} {ok_sym:<4}"
                f"{n_attempt:<3} {t_ans:>6.0f}  "
                f"{step['mastery_score']:>5.2f} {step['performance']:>5.2f} "
                f"{step['engagement']:>5.2f}  "
                f"{reward_col(step['reward'])}  "
                f"{agent.epsilon:>6.3f}  "
                f"{fmt_transition(prev_lt, lt)}{mlr_tag}"
            )

            if step["lt_changed"]:
                print_recommendation_change(step, agent, q_before, q_after, session_num)
            if q_num % 5 == 0 or q_num == SEEDING_QUESTIONS:
                print_qvalues(q_after, agent, f"Q{q_num}")

        prev_lt = lt

    if verbose:
        print_session_summary(agent, steps)

    return steps


# ── Normal multi-session ──────────────────────────────────────────────────────

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
        registry.get_agent(
            sid, category=category, seeding_lt=seeding_lt
        ).epsilon = initial_epsilon
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
        print_global_summary(all_steps, n_sessions, n_questions)

    return all_steps, registry


# ── Fixed-LT ──────────────────────────────────────────────────────────────────

def run_fixed_lt_simulation(
    fixed_lt:       str,
    profile_name:   str,
    n_sessions:     int,
    n_questions:    int,
    seed_questions: int  = SEEDING_QUESTIONS,
    category:       str  = "Penggalang",
    seed:           Optional[int] = None,
    ai_sim:         bool = False,
    api_key:        str  = "",
    verbose:        bool = True,
) -> Tuple[List[Dict], SessionRegistry]:
    if fixed_lt not in LEARNING_TYPES:
        raise ValueError(f"fixed_lt '{fixed_lt}' not in {LEARNING_TYPES}")
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    registry   = SessionRegistry()
    session_id = f"fixed_{fixed_lt}_{profile_name}_student"
    all_steps: List[Dict] = []

    for s in range(1, n_sessions + 1):
        student = StudentSimulator(profile_name, category, ai_sim, api_key)
        steps   = run_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, verbose=verbose,
            session_num=s, total_sessions=n_sessions,
            seeding_lt=fixed_lt,
        )
        all_steps.extend(steps)

    if verbose:
        print_global_summary(all_steps, n_sessions, n_questions)

    return all_steps, registry


# ── Story mode ────────────────────────────────────────────────────────────────

def run_story_simulation(
    assigned_lt:         str,
    true_lt:             str,
    profile_name:        str = "pai_misfit",
    n_seed_sessions:     int = 2,
    n_explore_sessions:  int = 2,
    n_converge_sessions: int = 3,
    n_questions:         int = 15,
    seed_questions:      int = SEEDING_QUESTIONS,
    category:            str = "Penggalang",
    seed:                Optional[int] = None,
    verbose:             bool = True,
) -> Tuple[List[Dict], SessionRegistry]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    registry   = SessionRegistry()
    session_id = f"story_{assigned_lt}_{profile_name}"
    all_steps: List[Dict] = []
    session_num = 0
    n_total = n_seed_sessions + n_explore_sessions + n_converge_sessions

    def banner(phase: int, desc: str, s1: int, s2: int) -> None:
        if not verbose: return
        col = {1: Y, 2: M, 3: G}.get(phase, B)
        print(f"\n{col}{B}{'#'*72}{X}")
        print(f"{col}{B}  PHASE {phase}  (Sessions {s1}–{s2})  {desc}{X}")
        print(f"{col}{B}{'#'*72}{X}")

    def phase_summary(steps_p: List[Dict], name: str, agent) -> None:
        if not verbose or not steps_p: return
        rewards = [s["reward"] for s in steps_p]
        correct = sum(1 for s in steps_p if s["is_correct"])
        dist    = Counter(s["lt"] for s in steps_p)
        print(f"\n  {B}Phase {name} summary:{X}")
        print(f"  {len(steps_p)} Q  avg={np.mean(rewards):+.4f}  correct={correct}/{len(steps_p)}")
        for lt, c in dist.most_common():
            tag = " ← assigned" if lt == assigned_lt else " ← TRUE fit" if lt == true_lt else ""
            print(f"    {lt_col(lt, lt):<14} {'|'*c}  ({c}){tag}")

    # Phase 1 — seeded on assigned LT
    banner(1, f"Student forced onto {assigned_lt} — wrong fit.", 1, n_seed_sessions)
    phase1: List[Dict] = []
    for _ in range(n_seed_sessions):
        session_num += 1
        student = StudentSimulator(profile_name, category)
        steps   = run_session(
            session_id=session_id, student=student, n_questions=n_questions,
            registry=registry, verbose=verbose,
            session_num=session_num, total_sessions=n_total,
            seeding_lt=assigned_lt,
        )
        for s in steps: s["story_phase"] = "assigned"
        phase1.extend(steps); all_steps.extend(steps)
    phase_summary(phase1, "1 ASSIGNED", registry.get_agent(session_id, category=category))

    # Phase 2 — high epsilon exploration
    banner(2, f"ε=0.50 — aggressive exploration. Looking for {true_lt}.",
           n_seed_sessions + 1, n_seed_sessions + n_explore_sessions)
    registry.get_agent(session_id, category=category).epsilon = 0.50
    phase2: List[Dict] = []
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
    phase_summary(phase2, "2 EXPLORE", registry.get_agent(session_id, category=category))

    # Phase 3 — converge
    banner(3, f"ε decays normally. Agent exploits {true_lt}.",
           n_seed_sessions + n_explore_sessions + 1, n_total)
    phase3: List[Dict] = []
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
    phase_summary(phase3, "3 CONVERGE", registry.get_agent(session_id, category=category))

    if verbose:
        print_global_summary(all_steps, n_total, n_questions)

    return all_steps, registry


# ── Per-LT benchmark ──────────────────────────────────────────────────────────

def run_per_lt_simulation(
    n_questions_per_lt: int,
    profile_name:       str,
    category:           str  = "Penggalang",
    seed:               Optional[int] = None,
    verbose:            bool = True,
) -> Dict[str, List[Dict]]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    results: Dict[str, List[Dict]] = {}
    for lt in LEARNING_TYPES:
        registry = SessionRegistry()
        agent    = registry.get_agent(f"perlt_{lt}_{profile_name}", category=category)
        student  = StudentSimulator(profile_name, category)
        steps: List[Dict] = []
        if verbose:
            print(f"  [{lt}] {n_questions_per_lt} Q...", flush=True)
        for q in range(1, n_questions_per_lt + 1):
            is_correct, n_attempt, t_ans = student.answer(lt)
            mc   = f"{agent.mastery_levels[lt]}{lt}"
            step = agent.record_response(lt, is_correct, n_attempt, t_ans, mc)
            steps.append({
                "lt": lt, "question_num": q,
                "is_correct": is_correct, "n_attempt": n_attempt,
                "t_answer":      t_ans,          "t_expected":  step["t_expected"],
                "mastery_score": step["mastery_score"],
                "mastery_level": step["mastery_level"],
                "performance":   step["performance"],
                "engagement":    step["engagement"],
                "reward":        step["reward"],
                "delta_m":       step["delta_m"],
                "mlr_refitted":  step["mlr_refitted"],
                "q_value_self":  agent.q_values()[lt],
                "epsilon":       agent.epsilon,
            })
        results[lt] = steps
    return results


# ── Cross-LT matrix ───────────────────────────────────────────────────────────

def run_cross_lt_simulation(
    n_questions: int  = 200,
    category:    str  = "Penggalang",
    seed:        Optional[int] = None,
    verbose:     bool = True,
) -> Dict:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    raw: Dict[str, Dict[str, List[float]]] = {slt: {} for slt in LEARNING_TYPES}

    for s_idx, student_lt in enumerate(LEARNING_TYPES):
        profile_name = f"student_{student_lt.lower()}"
        if verbose:
            print(f"  Student type {student_lt} ({s_idx+1}/8):", flush=True)
        for q_lt in LEARNING_TYPES:
            registry = SessionRegistry()
            agent    = registry.get_agent(f"cross_{student_lt}_{q_lt}", category=category)
            student  = StudentSimulator(profile_name, category)
            rewards: List[float] = []
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
            matrix[i, j]     = float(np.mean(raw[slt][qlt]))
            std_matrix[i, j] = float(np.std(raw[slt][qlt]))

    return {
        "matrix": matrix, "std_matrix": std_matrix,
        "raw": raw, "lt_list": LEARNING_TYPES, "n_questions": n_questions,
    }


# ── Correlation simulation (PAR/TAR positive-r demo) ─────────────────────────

def run_correlation_simulation(
    n_questions: int  = 40,
    n_sessions:  int  = 3,
    category:    str  = "Penggalang",
    seed:        int  = 42,
    verbose:     bool = True,
) -> Dict:
    """
    Single PAR/TAR-strong student.  Computes Pearson-r between the per-question
    reward sequences for PAR and TAR — should come out strongly positive because
    both LTs suit this student and the agent earns above-average reward on both.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    SEP = "═" * 72
    if verbose:
        print(f"\n{B}{SEP}{X}")
        print(f"{B}  LT CORRELATION SIMULATION  |  PAR/TAR student{X}")
        print(f"{B}  {n_sessions} sessions × {n_questions} Q  |  seed={seed}{X}")
        print(f"{B}{SEP}{X}\n")
        print("  Profile: high p_correct on PAR (0.82) and TAR (0.78),")
        print("           average on all other LTs (~0.43).")
        print("  Expectation: PAR and TAR reward sequences correlate positively.\n")

    registry  = SessionRegistry()
    student   = StudentSimulator("par_tar_student", category=category)
    all_steps: List[Dict] = []

    for s in range(1, n_sessions + 1):
        agent = registry.get_agent(f"corr_par_tar_s{s}", category=category, seeding_lt="PAR")
        agent.epsilon = 0.50
        steps = run_session(
            session_id=f"corr_par_tar_s{s}", student=student,
            n_questions=n_questions, registry=registry,
            verbose=verbose, session_num=s, total_sessions=n_sessions,
            seeding_lt="PAR",
        )
        all_steps.extend(steps)

    if verbose:
        print_global_summary(all_steps, n_sessions, n_questions)

    par_rewards = [s["reward"] for s in all_steps if s["lt"] == "PAR"]
    tar_rewards = [s["reward"] for s in all_steps if s["lt"] == "TAR"]
    min_len     = min(len(par_rewards), len(tar_rewards))
    corr        = float(np.corrcoef(par_rewards[:min_len], tar_rewards[:min_len])[0, 1]) \
                  if min_len >= 2 else 0.0

    avg_par   = float(np.mean(par_rewards))   if par_rewards   else 0.0
    avg_tar   = float(np.mean(tar_rewards))   if tar_rewards   else 0.0
    all_other = [s["reward"] for s in all_steps if s["lt"] not in ("PAR", "TAR")]
    avg_other = float(np.mean(all_other)) if all_other else 0.0

    if verbose:
        col = G if corr > 0.3 else (Y if corr > 0 else R)
        print(f"\n{B}{SEP}{X}")
        print(f"{B}  CORRELATION RESULT{X}")
        print(f"{B}{SEP}{X}")
        print(f"\n  PAR: {len(par_rewards)} Q  avg={reward_col(avg_par)}")
        print(f"  TAR: {len(tar_rewards)} Q  avg={reward_col(avg_tar)}")
        print(f"  Others:              avg={reward_col(avg_other)}")
        print(f"\n  Pearson r(PAR, TAR) = {col}{B}{corr:+.4f}{X}")
        verdict = (
            f"{G}Strong positive{X}" if corr > 0.3
            else f"{Y}Weak positive{X}" if corr > 0.05
            else f"{R}No clear positive{X}"
        )
        print(f"  Verdict: {verdict} correlation")
        print(f"{B}{SEP}{X}\n")

    return {
        "steps":            all_steps,
        "par_rewards":      par_rewards,
        "tar_rewards":      tar_rewards,
        "corr_par_tar":     corr,
        "avg_reward_par":   avg_par,
        "avg_reward_tar":   avg_tar,
        "avg_reward_other": avg_other,
        "n_questions":      n_questions,
        "n_sessions":       n_sessions,
    }


# ── Single-line simulation (radar-calibrated student) ────────────────────────

def run_single_line_simulation(
    profile_name:    str   = "par_tar_student",
    n_questions:     int   = 30,
    n_sessions:      int   = 1,
    category:        str   = "Penggalang",
    seed:            int   = 42,
    verbose:         bool  = True,
    initial_epsilon: float = 0.50,
) -> List[Dict]:
    """
    One student, all sessions linked in a single registry.

    Seeded on the profile's home LT (from PROFILE_HOME_LT) for the first
    SEEDING_QUESTIONS questions so the agent builds a strong reward baseline
    before exploring.  initial_epsilon starts at 0.50 so exploration is genuine.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    seeding_lt = PROFILE_HOME_LT.get(profile_name)
    registry   = SessionRegistry()
    all_steps: List[Dict] = []

    for s in range(1, n_sessions + 1):
        agent = registry.get_agent(
            f"singleline_s{s}", category=category, seeding_lt=seeding_lt
        )
        agent.epsilon = initial_epsilon
        student = StudentSimulator(profile_name, category=category)
        steps   = run_session(
            session_id     = f"singleline_s{s}",
            student        = student,
            n_questions    = n_questions,
            registry       = registry,
            verbose        = verbose,
            session_num    = s,
            total_sessions = n_sessions,
            seeding_lt     = seeding_lt,
        )
        for step in steps:
            step["global_q"] = (s - 1) * n_questions + step["question_num"]
        all_steps.extend(steps)

    if verbose:
        print_global_summary(all_steps, n_sessions, n_questions)

    return all_steps
