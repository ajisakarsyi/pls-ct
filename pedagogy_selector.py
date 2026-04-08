"""
pedagogy_selector.py — Contextual Bandit RL Agent for PLS
==========================================================
Selects the best learning type (PAR | TAR | PAI | … )
per student session using an epsilon-greedy linear bandit.

Architecture
------------
  RLAgent          — per-session agent: bandit weights + MetricsTracker
  SessionRegistry  — singleton mapping session_id → RLAgent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEEDING PHASE  (new in this version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The first SEEDING_QUESTIONS (default 10) calls to select_action()
  are the "seeding" phase.  If the agent was given a seeding_lt
  (the LT from the student's initial cognitive code), every selection
  during seeding returns that LT unconditionally so the bandit builds
  a real reward baseline before free exploration begins.
  Epsilon still decays during seeding so it is lower when free starts.

  Phase transitions:
    "seeding"  — total_selections < SEEDING_QUESTIONS  AND  seeding_lt set
    "free"     — all other cases (normal epsilon-greedy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LT CHANGE TRACKING  (new in this version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Every time argmax Q changes, a full record is appended to
  recommendation_history:
    change_index, step_num, timestamp, phase,
    from_lt, to_lt, q_values (full snapshot),
    mastery_levels, epsilon, trigger (TD/MLR)

  Every step dict in step_log also carries:
    "phase"                   — seeding | free
    "seeding_remaining"       — questions left in seeding window
    "q_values_after"          — full Q dict after bandit update
    "current_recommendation"  — argmax Q after this step
    "lt_changed"              — True if recommendation flipped
    "lt_change_detail"        — the change record if lt_changed, else None
"""

from __future__ import annotations

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rl_metrics import (
    LEARNING_TYPES,
    MetricsTracker,
    MASTERY_LEVEL_SCORE,
    MASTERY_LABELS,
    score_to_mastery_level,
    performance_label,
    engagement_label,
    N_MAX,
)

# ─────────────────────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM              = 3          # [mastery, performance, engagement]
N_ACTIONS              = len(LEARNING_TYPES)

EPSILON_INIT           = 0.50   # cold-start: explore half the time initially
EPSILON_MIN            = 0.05
EPSILON_DECAY          = 0.97   # faster decay so we exploit sooner
LEARNING_RATE          = 0.05
MLR_REFIT_EVERY        = 10

# ── Seeding phase ─────────────────────────────────────────────────────────────
SEEDING_QUESTIONS      = 10   # force assigned LT for this many questions
SEEDING_SESSIONS       = SEEDING_QUESTIONS   # legacy alias kept for ollamaapi.py

CONSECUTIVE_TO_PROMOTE = 1      # promote after 1 correct answer on a LT
DEFAULT_CATEGORY       = "Penggalang"


# ─────────────────────────────────────────────────────────────────────────────
# PER-SESSION RL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Per-session contextual bandit with seeding phase + LT change tracking.

    Phases
    ------
    "seeding"  — first SEEDING_QUESTIONS calls to select_action():
                 always returns seeding_lt.  Epsilon still decays.
    "free"     — normal epsilon-greedy.

    LT Change Log
    -------------
    recommendation_history — one entry per argmax-Q change.
    step_log entries carry q_values_after, current_recommendation, lt_changed.
    """

    def __init__(
        self,
        session_id:        str,
        category:          str   = DEFAULT_CATEGORY,
        epsilon:           float = EPSILON_INIT,
        learning_rate:     float = LEARNING_RATE,
        seeding_lt:        Optional[str] = None,
        seeding_questions: int   = SEEDING_QUESTIONS,
    ):
        self.session_id        = session_id
        self.category          = category
        self.created_at        = datetime.now().isoformat()

        # Bandit weights  (N_ACTIONS × STATE_DIM)  — optimistic init breaks ties
        rng     = np.random.default_rng(abs(hash(session_id)) % (2**32))
        self.W  = rng.uniform(0.05, 0.15, size=(N_ACTIONS, STATE_DIM))  # optimistic init
        self.lr = learning_rate

        # Exploration
        self.epsilon       = epsilon
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        # Seeding phase
        self.seeding_lt        = seeding_lt.upper() if seeding_lt else None
        self.seeding_questions = seeding_questions
        self.total_selections  = 0   # incremented every select_action() call

        # Metrics tracker
        self.tracker = MetricsTracker(student_id=session_id, category=category)

        # Per-LT state
        self.mastery_levels:      Dict[str, int] = {lt: 1 for lt in LEARNING_TYPES}
        self.question_counters:   Dict[str, int] = {lt: 0 for lt in LEARNING_TYPES}
        self.consecutive_correct: Dict[str, int] = {lt: 0 for lt in LEARNING_TYPES}

        self.last_action_lt:  Optional[str] = None
        self.last_action_idx: Optional[int] = None

        # LT recommendation change log
        self.recommendation_history: List[Dict] = []
        self._last_recommendation:   Optional[str] = None   # previous argmax Q

        # Full step log
        self.step_log: List[Dict] = []

    # ── PHASE ────────────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        """'seeding' for the first SEEDING_QUESTIONS selections, then 'free'."""
        if self.seeding_lt and self.total_selections < self.seeding_questions:
            return "seeding"
        return "free"

    @property
    def seeding_remaining(self) -> int:
        if not self.seeding_lt:
            return 0
        return max(0, self.seeding_questions - self.total_selections)

    @property
    def seeding_progress_pct(self) -> float:
        if not self.seeding_lt or self.seeding_questions == 0:
            return 1.0
        return min(1.0, self.total_selections / self.seeding_questions)

    def set_seeding_lt(self, lt: str) -> None:
        lt = lt.upper()
        if lt in LEARNING_TYPES:
            self.seeding_lt = lt

    # ── INTERNAL ─────────────────────────────────────────────────────────────

    def _global_state(self) -> np.ndarray:
        """Mean of [M, P, E] across all LT states."""
        return np.stack([
            self.tracker.states[lt].vector() for lt in LEARNING_TYPES
        ]).mean(axis=0)

    def _current_best_lt(self) -> str:
        return LEARNING_TYPES[int(np.argmax(self.W @ self._global_state()))]

    def _record_change_if_needed(self, step_num: int, phase: str,
                                  trigger: str = "TD") -> Optional[Dict]:
        """
        If argmax Q changed since last call, append a full change record
        to recommendation_history and return it. Otherwise return None.
        """
        best = self._current_best_lt()
        if best == self._last_recommendation:
            return None

        qs = self.W @ self._global_state()
        change = {
            "change_index":   len(self.recommendation_history) + 1,
            "step_num":       step_num,
            "timestamp":      datetime.now().isoformat(),
            "phase":          phase,
            "from_lt":        self._last_recommendation,   # None on first ever change
            "to_lt":          best,
            "q_values":       {l: round(float(qs[i]), 6)
                               for i, l in enumerate(LEARNING_TYPES)},
            "mastery_levels": dict(self.mastery_levels),
            "epsilon":        round(self.epsilon, 4),
            "trigger":        trigger,   # "TD" or "MLR"
        }
        self.recommendation_history.append(change)
        self._last_recommendation = best
        return change

    # ── ACTION SELECTION ────────────────────────────────────────────────────

    def select_action(self) -> Tuple[str, int]:
        """
        Phase-aware epsilon-greedy selection.

        Seeding phase → forced seeding_lt (epsilon still decays).
        Free phase    → standard epsilon-greedy.
        """
        cur_phase = self.phase          # check BEFORE incrementing
        self.total_selections += 1

        if cur_phase == "seeding" and self.seeding_lt:
            lt  = self.seeding_lt
            idx = LEARNING_TYPES.index(lt)
        elif np.random.rand() < self.epsilon:
            idx = np.random.randint(N_ACTIONS)
            lt  = LEARNING_TYPES[idx]
        else:
            qs  = self.W @ self._global_state()
            idx = int(np.argmax(qs))
            lt  = LEARNING_TYPES[idx]

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.last_action_lt  = lt
        self.last_action_idx = idx
        return lt, idx

    # ── RESPONSE RECORDING & BANDIT UPDATE ──────────────────────────────────

    def record_response(
        self,
        learning_type:    str,
        is_correct:       bool,
        n_attempt:        int,
        t_answer_seconds: float,
        mastery_code:     Optional[str] = None,
    ) -> Dict:
        """
        Record evaluation result, update MetricsTracker and bandit W.

        Enhanced step dict includes:
          phase, seeding_remaining, total_selections,
          q_values_after, current_recommendation,
          lt_changed, lt_change_detail
        """
        lt = learning_type.upper()
        if lt not in LEARNING_TYPES:
            raise ValueError(f"Unknown LT '{lt}'. Valid: {LEARNING_TYPES}")

        cur_phase = self.phase

        # Mastery progression
        if is_correct:
            self.consecutive_correct[lt] += 1
            if self.consecutive_correct[lt] >= CONSECUTIVE_TO_PROMOTE:
                self.mastery_levels[lt] = min(6, self.mastery_levels[lt] + 1)
                self.consecutive_correct[lt] = 0
        else:
            self.consecutive_correct[lt] = 0

        if mastery_code is None:
            mastery_code = f"{self.mastery_levels[lt]}{lt}"

        q_idx = self.question_counters[lt]
        self.question_counters[lt] += 1

        prev_state = self._global_state()
        prev_best  = self._current_best_lt()

        # MetricsTracker update — pass is_correct so engagement is zeroed on wrong answers
        record = self.tracker.record_attempt(
            learning_type    = lt,
            mastery_code     = mastery_code,
            n_attempt        = n_attempt,
            t_answer_seconds = t_answer_seconds,
            question_idx     = q_idx,
            is_correct       = is_correct,
        )

        # Bandit TD update
        action_idx = LEARNING_TYPES.index(lt)
        reward     = record["reward"]
        q_pred     = float(self.W[action_idx] @ prev_state)
        td_error   = reward - q_pred
        self.W[action_idx] += self.lr * td_error * prev_state

        # Q-values snapshot AFTER update
        qs_after       = self.W @ self._global_state()
        q_values_after = {l: round(float(qs_after[i]), 6)
                          for i, l in enumerate(LEARNING_TYPES)}
        new_best    = LEARNING_TYPES[int(np.argmax(qs_after))]
        lt_changed  = (new_best != prev_best)

        # MLR periodic refit
        total        = len(self.tracker._reward_history)
        mlr_refitted = False
        mlr_weights  = None
        trigger      = "TD"
        if total > 0 and total % MLR_REFIT_EVERY == 0:
            mlr_weights  = self.tracker.fit_mlr_weights()
            mlr_refitted = True
            trigger      = "MLR"

        # Record LT change if argmax Q flipped
        step_num = len(self.step_log) + 1
        lt_change_detail = self._record_change_if_needed(step_num, cur_phase, trigger)

        step = {
            # ── identity ──────────────────────────────────────────────────
            "step_num":               step_num,
            "timestamp":              datetime.now().isoformat(),
            "session_id":             self.session_id,
            # ── phase ─────────────────────────────────────────────────────
            "phase":                  cur_phase,
            "seeding_remaining":      self.seeding_remaining,
            "total_selections":       self.total_selections,
            # ── LT ────────────────────────────────────────────────────────
            "learning_type":          lt,
            "mastery_code":           mastery_code,
            "mastery_level":          self.mastery_levels[lt],
            "mastery_label":          MASTERY_LABELS[self.mastery_levels[lt]],
            "mastery_score":          record["mastery_score"],
            # ── metrics ───────────────────────────────────────────────────
            "performance":            record["performance"],
            "performance_label":      performance_label(record["performance"]),
            "engagement":             record["engagement"],
            "engagement_label":       engagement_label(record["engagement"]),
            "t_answer":               t_answer_seconds,
            "t_expected":             record["t_expected"],
            "n_attempt":              n_attempt,
            "is_correct":             is_correct,
            "delta_m":                record["delta_m"],
            "delta_p":                record["delta_p"],
            # ── bandit ────────────────────────────────────────────────────
            "reward":                 reward,
            "td_error":               td_error,
            "q_pred":                 q_pred,
            "q_values_after":         q_values_after,
            "epsilon":                self.epsilon,
            # ── recommendation ────────────────────────────────────────────
            "current_recommendation": new_best,
            "lt_changed":             lt_changed,
            "lt_change_detail":       lt_change_detail,
            # ── MLR ───────────────────────────────────────────────────────
            "mlr_refitted":           mlr_refitted,
            "mlr_weights":            mlr_weights,
        }
        self.step_log.append(step)
        return step

    # ── Q-VALUES & RECOMMENDATION ────────────────────────────────────────────

    def q_values(self) -> Dict[str, float]:
        qs = self.W @ self._global_state()
        return {lt: float(qs[i]) for i, lt in enumerate(LEARNING_TYPES)}

    def recommend(self) -> Dict:
        qs   = self.q_values()
        best = max(qs, key=qs.__getitem__)
        per_type = {}
        for lt in LEARNING_TYPES:
            s = self.tracker.states[lt]
            per_type[lt] = {
                "q_value":             round(qs[lt], 5),
                "mastery_score":       round(s.mastery_score, 3),
                "mastery_level":       self.mastery_levels[lt],
                "mastery_label":       MASTERY_LABELS[self.mastery_levels[lt]],
                "performance":         round(s.performance, 3),
                "performance_label":   performance_label(s.performance),
                "engagement":          round(s.engagement, 3),
                "engagement_label":    engagement_label(s.engagement),
                "n_attempts_total":    self.question_counters[lt],
                "consecutive_correct": self.consecutive_correct[lt],
            }
        return {
            "session_id":     self.session_id,
            "recommended_lt": best,
            "epsilon":        round(self.epsilon, 4),
            "per_type":       per_type,
            "reward_weights": {
                "beta0":  round(self.tracker.beta0, 4),
                "alpha":  round(self.tracker.alpha, 4),
                "beta":   round(self.tracker.beta, 4),
                "gamma":  round(self.tracker.gamma, 4),
            },
        }

    def summary(self) -> Dict:
        return {
            "session_id":             self.session_id,
            "category":               self.category,
            "created_at":             self.created_at,
            "epsilon":                round(self.epsilon, 4),
            "total_steps":            len(self.step_log),
            "phase":                  self.phase,
            "seeding_lt":             self.seeding_lt,
            "seeding_questions":      self.seeding_questions,
            "seeding_remaining":      self.seeding_remaining,
            "seeding_progress_pct":   round(self.seeding_progress_pct * 100, 1),
            "total_selections":       self.total_selections,
            "total_lt_changes":       len(self.recommendation_history),
            "current_recommendation": self._current_best_lt(),
            "recommendation_history": self.recommendation_history,
            "recommendation":         self.recommend(),
            "current_metrics":        self.tracker.get_all_metrics(),
            "mastery_progression": {
                lt: {
                    "level":              self.mastery_levels[lt],
                    "label":              MASTERY_LABELS[self.mastery_levels[lt]],
                    "score":              MASTERY_LEVEL_SCORE[self.mastery_levels[lt]],
                    "streak":             self.consecutive_correct[lt],
                    "questions_answered": self.question_counters[lt],
                }
                for lt in LEARNING_TYPES
            },
            "bandit_W":   self.W.tolist(),
            "last_action": {"lt": self.last_action_lt, "idx": self.last_action_idx},
        }

    def export_log(self) -> List[Dict]:
        return self.step_log


# ─────────────────────────────────────────────────────────────────────────────
# SESSION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class SessionRegistry:
    """Manages RLAgent instances keyed by session_id."""

    def __init__(self, log_dir: Optional[str] = None):
        self._agents: Dict[str, RLAgent] = {}
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def get_agent(
        self,
        session_id: str,
        category:   str  = DEFAULT_CATEGORY,
        seeding_lt: Optional[str] = None,
    ) -> RLAgent:
        if session_id not in self._agents:
            self._agents[session_id] = RLAgent(
                session_id=session_id, category=category, seeding_lt=seeding_lt,
            )
        elif seeding_lt and self._agents[session_id].seeding_lt is None:
            self._agents[session_id].set_seeding_lt(seeding_lt)
        return self._agents[session_id]

    def has_session(self, session_id: str) -> bool:
        return session_id in self._agents

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._agents:
            del self._agents[session_id]; return True
        return False

    def all_session_ids(self) -> List[str]:
        return list(self._agents.keys())

    def global_stats(self) -> Dict:
        total_steps = sum(len(a.step_log) for a in self._agents.values())
        recommendations: Dict[str, int] = {}
        phases: Dict[str, int] = {}
        for agent in self._agents.values():
            lt = agent._current_best_lt()
            recommendations[lt] = recommendations.get(lt, 0) + 1
            p = agent.phase
            phases[p] = phases.get(p, 0) + 1
        return {
            "active_sessions":    len(self._agents),
            "total_steps":        total_steps,
            "lt_recommendations": recommendations,
            "phase_counts":       phases,
        }

    def refit_all_weights(self) -> Dict[str, Dict]:
        results = {}
        for sid, agent in self._agents.items():
            if len(agent.tracker._reward_history) >= 4:
                results[sid] = agent.tracker.fit_mlr_weights()
        return results

    def save_logs(self) -> None:
        if not self.log_dir:
            return
        for sid, agent in self._agents.items():
            path = os.path.join(self.log_dir, f"rl_log_{sid}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(agent.export_log(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[RL] ⚠️ Save failed for {sid}: {e}")


# Module-level singleton — imported by ollamaapi.py
registry = SessionRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY COMPAT  (so old call-sites still work)
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = LEARNING_TYPES


class StudentState:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._agent = registry.get_agent(session_id)

    @property
    def mastery(self) -> float:
        return float(self._agent._global_state()[0])

    @property
    def performance(self) -> float:
        return float(self._agent._global_state()[1])

    @property
    def engagement(self) -> float:
        return float(self._agent._global_state()[2])

    def vector(self) -> np.ndarray:
        return self._agent._global_state()

    def update(self, *_): pass


def select_pedagogy(student: StudentState):
    return registry.get_agent(student.session_id).select_action()


def update_after_response(student, action_idx, new_mastery, new_perf, new_eng):
    lt    = LEARNING_TYPES[action_idx]
    agent = registry.get_agent(student.session_id)
    level = score_to_mastery_level(new_mastery)
    code  = f"{level}{lt}"
    n_att = max(1, round((1.0 - new_perf) * (N_MAX - 1)) + 1)
    return agent.record_response(
        learning_type=lt, is_correct=(new_perf >= 0.75),
        n_attempt=n_att, t_answer_seconds=120.0, mastery_code=code,
    )["reward"]
