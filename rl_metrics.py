"""
rl_metrics.py — Reward & State Metrics Fetcher for PLS RL Model
================================================================
Computes per-student, per-learning-type metrics:
  • Mastery  (M)  — from level codes e.g. 1PAR, 3TGI …
  • Performance (P) — from attempt counts
  • Engagement (E) — from answer time vs expected time
  • Reward (R)    — weighted sum via MLR-learned weights

Learning types : PAR, TAR, PAI, TGR, TGI
Mastery levels : 1–6 (6 = full mastery)

Usage
-----
    from rl_metrics import MetricsTracker, compute_reward, fit_weights

    tracker = MetricsTracker(student_id="s001", category="Penggalang")
    tracker.record_attempt(
        learning_type="PAR",
        mastery_code="3PAR",        # e.g. from PLS progress data
        n_attempt=2,                 # how many tries until correct
        t_answer_seconds=85.0,       # actual seconds taken
        question_idx=0,              # for per-question Texpected lookup
    )
    metrics = tracker.get_current_metrics("PAR")
    reward  = tracker.get_reward("PAR")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

LEARNING_TYPES = ["PAR", "TAR", "PAI", "TAI", "TGR", "TGI", "PGR", "PGI"]
N_MAX = 5          # maximum attempts before the system reveals the answer

# Category → (time_limit_seconds, n_questions)
# Derived from the Tantangan table (image reference)
CATEGORY_CONFIG: Dict[str, Tuple[int, int]] = {
    "SiKecil":    (30 * 60,  8),   # SD/MI  ≤ kelas 3
    "Siaga":      (40 * 60, 12),   # SD/MI  kelas 4–6
    "Penggalang": (45 * 60, 15),   # SMP    kelas 7–9
    "Penegak":    (45 * 60, 15),   # SMA/SMK kelas 10–12
}

# ─────────────────────────────────────────────
# MASTERY SCORING
# ─────────────────────────────────────────────

# Mastery level → representative score (midpoint of each band)
MASTERY_LEVEL_SCORE: Dict[int, float] = {
    1: 0.15,   # ≤ 0.30         → Pemula
    2: 0.40,   # 0.31 – 0.50    → Pemahaman dasar
    3: 0.60,   # 0.51 – 0.70    → Pemahaman menengah
    4: 0.75,   # 0.71 – 0.80    → Pemahaman berkelanjutan
    5: 0.88,   # 0.81 – 0.94    → Near mastery
    6: 0.97,   # ≥ 0.95         → Mastery
}

MASTERY_LABELS: Dict[int, str] = {
    1: "Pemula",
    2: "Pemahaman dasar",
    3: "Pemahaman menengah",
    4: "Pemahaman berkelanjutan",
    5: "Near mastery",
    6: "Mastery",
}


def score_to_mastery_level(score: float) -> int:
    """Convert a raw mastery score (0–1) to level integer (1–6)."""
    if score <= 0.30:
        return 1
    elif score <= 0.50:
        return 2
    elif score <= 0.70:
        return 3
    elif score <= 0.80:
        return 4
    elif score <= 0.94:
        return 5
    else:
        return 6


def parse_mastery_code(code: str) -> Tuple[int, str]:
    """
    Parse a mastery code like "3PAR" or "5TGI".

    Returns
    -------
    (level: int, learning_type: str)
    """
    code = code.strip().upper()
    for lt in LEARNING_TYPES:
        if code.endswith(lt):
            level_str = code[: -len(lt)]
            level = int(level_str)
            if level not in MASTERY_LEVEL_SCORE:
                raise ValueError(f"Mastery level {level} out of range 1–6")
            return level, lt
    raise ValueError(
        f"Cannot parse mastery code '{code}'. "
        f"Expected format like '3PAR', '5TGI'. "
        f"Valid types: {LEARNING_TYPES}"
    )


def mastery_score_from_code(code: str) -> float:
    """Return the numeric mastery score [0,1] from a code like '4TAR'."""
    level, _ = parse_mastery_code(code)
    return MASTERY_LEVEL_SCORE[level]


# ─────────────────────────────────────────────
# PERFORMANCE METRIC  Pt = (N_max - N_attempt + 1) / N_max
# ─────────────────────────────────────────────
# Higher score = fewer attempts needed = better performance.
# Formula interpretation:
#   Pt = 1 − (N_attempt / N_max)  normalized so:
#   N_attempt=1 → Pt=1.0 (always correct)
#   N_attempt=N_max → Pt=0.0 (always wrong / system gave answer)
#
# NOTE: The original formula Pt = N_attempt/N_max gives HIGHER values
# for MORE attempts (i.e. worse performance). We invert it here so
# that higher score always means better, consistent with mastery & engagement.

def compute_performance(n_attempt: int, n_max: int = N_MAX) -> float:
    """
    Compute normalised performance score.

    Parameters
    ----------
    n_attempt : int  — number of tries until correct (1 = first try)
    n_max     : int  — max allowed attempts (system reveals answer after this)

    Returns
    -------
    float in [0.0, 1.0]  — higher is better
    """
    n_attempt = max(1, min(n_attempt, n_max))
    # 1 → 1.0 (best),  n_max → 0.0 (worst)
    return 1.0 - (n_attempt - 1) / (n_max - 1) if n_max > 1 else 1.0


def performance_label(pt: float) -> str:
    """Human-readable label for a performance score."""
    if pt <= 0.05:
        return "Selalu salah"
    elif pt <= 0.25:
        return "Banyak salah"
    elif pt <= 0.45:
        return "Rata-rata benar"
    elif pt <= 0.65:
        return "Mulai konsisten benar"
    elif pt <= 0.85:
        return "Konsisten benar"
    else:
        return "Selalu benar"


# ─────────────────────────────────────────────
# ENGAGEMENT METRIC  Et = Tt / T_expected
# ─────────────────────────────────────────────

def compute_expected_time(
    category: str,
    question_idx: Optional[int] = None,
    historical_times: Optional[List[float]] = None,
) -> float:
    """
    Compute T_expected for a single question.

    Priority order:
    1. Mean of `historical_times` if provided (per-question average).
    2. Uniform split: total_time / n_questions for the given category.

    Parameters
    ----------
    category         : str  — one of CATEGORY_CONFIG keys
    question_idx     : int  — index of the question (unused here; for future per-question override)
    historical_times : list — list of past answer times (seconds) for this question
    """
    if historical_times and len(historical_times) > 0:
        return float(np.mean(historical_times))

    if category not in CATEGORY_CONFIG:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid: {list(CATEGORY_CONFIG.keys())}"
        )
    total_seconds, n_questions = CATEGORY_CONFIG[category]
    return total_seconds / n_questions  # uniform distribution


def compute_engagement(
    t_answer: float,
    t_expected: float,
    clip: bool = True,
) -> float:
    """
    Et = Tt / T_expected.

    Parameters
    ----------
    t_answer  : float — actual seconds taken to answer
    t_expected: float — expected seconds per question
    clip      : bool  — if True, clamp result to [0, 1]

    Returns
    -------
    float  — higher means answered faster than expected (more engaged)
             clamped to [0, 1] when clip=True
    """
    if t_expected <= 0:
        return 0.0
    et = t_answer / t_expected
    if clip:
        et = max(0.0, min(et, 1.0))  # raw ratio ≤ 1 = on-time, > 1 = slow
    # Invert so that faster = higher engagement score
    return 1.0 - et if clip else et


def engagement_label(et: float) -> str:
    if et < 0.40:
        return "Low"
    elif et < 0.60:
        return "Moderate"
    else:
        return "High"


# ─────────────────────────────────────────────
# MLR WEIGHT FITTING  β = (XᵀX)⁻¹ Xᵀy
# ─────────────────────────────────────────────

def fit_weights(
    history: List[Dict],
    y_key: str = "reward",
) -> Tuple[float, float, float, float]:
    """
    Fit reward weights (β0, β1, β2, β3) via Multiple Linear Regression.

        y = β0 + β1·ΔP + β2·ΔM + β3·E

    Parameters
    ----------
    history : list of dicts with keys:
              'delta_p', 'delta_m', 'engagement', 'reward'
    y_key   : column to use as target (default 'reward')

    Returns
    -------
    (beta0, alpha, beta, gamma) — intercept, ΔP weight, ΔM weight, E weight
    """
    if len(history) < 4:
        # Not enough data — return sensible defaults (Chen et al. 2023)
        return 0.0, 0.025, 0.399, 0.204

    X_rows, y_vals = [], []
    for row in history:
        X_rows.append([1.0, row["delta_p"], row["delta_m"], row["engagement"]])
        y_vals.append(row[y_key])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    try:
        # β = (XᵀX)⁻¹ Xᵀy
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(betas[0]), float(betas[1]), float(betas[2]), float(betas[3])
    except np.linalg.LinAlgError:
        return 0.0, 0.025, 0.399, 0.204


# ─────────────────────────────────────────────
# REWARD FUNCTION
# ─────────────────────────────────────────────

def compute_reward(
    delta_p: float,
    delta_m: float,
    engagement: float,
    alpha: float = 0.025,
    beta: float = 0.399,
    gamma: float = 0.204,
    beta0: float = 0.0,
) -> float:
    """
    rt = β0 + α·ΔP + β·ΔM + γ·E

    Parameters
    ----------
    delta_p    : change in performance (current − previous)
    delta_m    : change in mastery score (current − previous)
    engagement : current engagement score (not a delta)
    alpha      : ΔP weight
    beta       : ΔM weight  (primary — knowledge gain)
    gamma      : E weight   (secondary — engagement bonus)
    beta0      : intercept from MLR

    Returns
    -------
    float — reward signal for the RL agent
    """
    return beta0 + alpha * delta_p + beta * delta_m + gamma * engagement


# ─────────────────────────────────────────────
# STUDENT METRICS STATE
# ─────────────────────────────────────────────

@dataclass
class LearningTypeState:
    """Tracks running metrics for one student × one learning type."""
    learning_type: str
    mastery_score: float = 0.30          # M_t
    performance: float = 0.40            # P_t
    engagement: float = 0.60            # E_t
    mastery_level: int = 1
    attempt_history: List[Dict] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        return np.array([self.mastery_score, self.performance, self.engagement])

    def update(
        self,
        new_mastery: float,
        new_perf: float,
        new_eng: float,
    ) -> Dict:
        prev = self.vector()
        self.mastery_score = new_mastery
        self.performance = new_perf
        self.engagement = new_eng
        self.mastery_level = score_to_mastery_level(new_mastery)
        current = self.vector()
        delta = current - prev
        return {
            "prev_mastery": prev[0],
            "prev_performance": prev[1],
            "prev_engagement": prev[2],
            "delta_m": float(delta[0]),
            "delta_p": float(delta[1]),
            "engagement": float(new_eng),
        }


# ─────────────────────────────────────────────
# MAIN TRACKER CLASS
# ─────────────────────────────────────────────

class MetricsTracker:
    """
    Per-student metrics tracker.

    Keeps state for all 5 learning types and provides
    methods to record attempts, fetch current metrics,
    compute rewards, and fit MLR weights from history.

    Parameters
    ----------
    student_id : str
    category   : str — 'SiKecil' | 'Siaga' | 'Penggalang' | 'Penegak'
    alpha, beta, gamma : initial reward weights (overridden by fit_weights)
    """

    def __init__(
        self,
        student_id: str,
        category: str = "Penggalang",
        alpha: float = 0.025,
        beta: float = 0.399,
        gamma: float = 0.204,
    ):
        self.student_id = student_id
        self.category = category
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.beta0 = 0.0

        # Per-type state
        self.states: Dict[str, LearningTypeState] = {
            lt: LearningTypeState(learning_type=lt)
            for lt in LEARNING_TYPES
        }

        # Historical records for MLR
        self._reward_history: List[Dict] = []

        # Per-question historical times for Texpected estimation
        self._question_times: Dict[str, List[float]] = {}  # key: "lt_qidx"

    # ── RECORD ONE ATTEMPT ──────────────────────────────────────────────

    def record_attempt(
        self,
        learning_type: str,
        mastery_code: str,
        n_attempt: int,
        t_answer_seconds: float,
        question_idx: Optional[int] = None,
        n_max: int = N_MAX,
        is_correct: bool = True,
    ) -> Dict:
        """
        Record a student's attempt and compute/update all metrics.

        Parameters
        ----------
        learning_type     : str  — e.g. "PAR"
        mastery_code      : str  — e.g. "3PAR" (from PLS progress data)
        n_attempt         : int  — 1..n_max (number of tries until correct)
        t_answer_seconds  : float— wall-clock seconds the student took
        question_idx      : int  — question index within session (optional)
        n_max             : int  — override default max attempts
        is_correct        : bool — False forces engagement=0 so wrong answers
                                   carry no engagement bonus in the reward.

        Returns
        -------
        dict with all computed metrics and reward
        """
        lt = learning_type.upper()
        if lt not in self.states:
            raise ValueError(f"Unknown learning type '{lt}'. Valid: {LEARNING_TYPES}")

        # ── Mastery
        level, _ = parse_mastery_code(mastery_code)
        m_new = MASTERY_LEVEL_SCORE[level]

        # ── Performance
        p_new = compute_performance(n_attempt, n_max)

        # ── Engagement — update historical times
        q_key = f"{lt}_{question_idx}" if question_idx is not None else lt
        if q_key not in self._question_times:
            self._question_times[q_key] = []
        t_expected = compute_expected_time(
            self.category,
            question_idx=question_idx,
            historical_times=self._question_times[q_key] if self._question_times[q_key] else None,
        )
        # Engagement is zero on wrong/exhausted answers.
        # A student who did not solve the question correctly gets no engagement bonus.
        # This prevents high-speed wrong answers from earning positive reward.
        if is_correct:
            e_new = compute_engagement(t_answer_seconds, t_expected)
        else:
            e_new = 0.0
        # Always record the time so Texpected adapts to actual answering pace
        self._question_times[q_key].append(t_answer_seconds)

        # ── Update state and get deltas
        state = self.states[lt]
        deltas = state.update(m_new, p_new, e_new)

        # ── Reward
        reward = compute_reward(
            delta_p=deltas["delta_p"],
            delta_m=deltas["delta_m"],
            engagement=e_new,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            beta0=self.beta0,
        )

        # ── Record for MLR
        record = {
            "student_id": self.student_id,
            "learning_type": lt,
            "mastery_code": mastery_code,
            "mastery_level": level,
            "mastery_score": m_new,
            "performance": p_new,
            "engagement": e_new,
            "t_answer": t_answer_seconds,
            "t_expected": t_expected,
            "n_attempt": n_attempt,
            "delta_p": deltas["delta_p"],
            "delta_m": deltas["delta_m"],
            "reward": reward,
        }
        self._reward_history.append(record)
        state.attempt_history.append(record)

        return record

    # ── CURRENT METRICS ─────────────────────────────────────────────────

    def get_current_metrics(self, learning_type: str) -> Dict:
        """Return current (M, P, E) state for a learning type."""
        lt = learning_type.upper()
        s = self.states[lt]
        return {
            "student_id": self.student_id,
            "learning_type": lt,
            "mastery_score": s.mastery_score,
            "mastery_level": s.mastery_level,
            "mastery_label": MASTERY_LABELS[s.mastery_level],
            "performance": s.performance,
            "performance_label": performance_label(s.performance),
            "engagement": s.engagement,
            "engagement_label": engagement_label(s.engagement),
            "state_vector": s.vector().tolist(),
        }

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Return current metrics for all 5 learning types."""
        return {lt: self.get_current_metrics(lt) for lt in LEARNING_TYPES}

    # ── REWARD ──────────────────────────────────────────────────────────

    def get_reward(self, learning_type: str) -> float:
        """Return the most recent reward for a given learning type."""
        lt = learning_type.upper()
        hist = self.states[lt].attempt_history
        if not hist:
            return 0.0
        return hist[-1]["reward"]

    def get_reward_history(self, learning_type: Optional[str] = None) -> List[Dict]:
        """Return reward history (all types or filtered)."""
        if learning_type is None:
            return self._reward_history
        lt = learning_type.upper()
        return [r for r in self._reward_history if r["learning_type"] == lt]

    # ── MLR WEIGHT UPDATE ────────────────────────────────────────────────

    def fit_mlr_weights(self) -> Dict:
        """
        Refit reward weights using accumulated history via MLR:
            β = (XᵀX)⁻¹ Xᵀy

        Updates self.beta0, self.alpha, self.beta, self.gamma in-place.

        Returns
        -------
        dict with fitted weights and diagnostics
        """
        beta0, alpha, beta, gamma = fit_weights(self._reward_history)
        self.beta0 = beta0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        return {
            "beta0": beta0,
            "alpha_perf": alpha,
            "beta_mastery": beta,
            "gamma_engagement": gamma,
            "n_samples": len(self._reward_history),
        }

    # ── BEST LEARNING TYPE ───────────────────────────────────────────────

    def recommend_learning_type(self) -> Dict:
        """
        Return the learning type with the highest expected reward
        based on current student state × current weights.
        (Linear scoring: α·P + β·M + γ·E per type)
        """
        scores = {}
        for lt in LEARNING_TYPES:
            s = self.states[lt]
            scores[lt] = (
                self.alpha * s.performance
                + self.beta * s.mastery_score
                + self.gamma * s.engagement
            )
        best_lt = max(scores, key=scores.__getitem__)
        return {
            "recommended": best_lt,
            "scores": scores,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        }

    # ── SUMMARY ─────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Full snapshot of all metrics, weights, and recommendation."""
        return {
            "student_id": self.student_id,
            "category": self.category,
            "weights": {"beta0": self.beta0, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
            "metrics": self.get_all_metrics(),
            "recommendation": self.recommend_learning_type(),
            "n_total_attempts": len(self._reward_history),
        }


# ─────────────────────────────────────────────
# STANDALONE HELPER: compute metrics from raw dict
# ─────────────────────────────────────────────

def compute_metrics_from_dict(data: Dict, category: str = "Penggalang") -> Dict:
    """
    Compute all metrics from a flat dict without a tracker instance.

    Input dict keys
    ---------------
    mastery_code     : str   — e.g. "4PAR"
    n_attempt        : int   — 1..5
    t_answer_seconds : float — wall-clock seconds
    prev_mastery     : float — previous mastery score (default 0.30)
    prev_performance : float — previous performance score (default 0.40)
    historical_times : list  — (optional) list of past answer seconds for Texpected
    alpha, beta, gamma       — (optional) reward weights

    Returns
    -------
    dict with mastery, performance, engagement, deltas, t_expected, reward
    """
    mastery_code = data["mastery_code"]
    n_attempt = int(data.get("n_attempt", 1))
    t_answer = float(data["t_answer_seconds"])
    prev_m = float(data.get("prev_mastery", 0.30))
    prev_p = float(data.get("prev_performance", 0.40))
    historical_times = data.get("historical_times", None)
    alpha = float(data.get("alpha", 0.025))
    beta_w = float(data.get("beta", 0.399))
    gamma = float(data.get("gamma", 0.204))

    m_new = mastery_score_from_code(mastery_code)
    p_new = compute_performance(n_attempt)
    t_exp = compute_expected_time(category, historical_times=historical_times)
    e_new = compute_engagement(t_answer, t_exp)

    delta_m = m_new - prev_m
    delta_p = p_new - prev_p
    reward = compute_reward(delta_p, delta_m, e_new, alpha, beta_w, gamma)

    level, lt = parse_mastery_code(mastery_code)
    return {
        "learning_type": lt,
        "mastery_code": mastery_code,
        "mastery_level": level,
        "mastery_label": MASTERY_LABELS[level],
        "mastery_score": m_new,
        "performance": p_new,
        "performance_label": performance_label(p_new),
        "engagement": e_new,
        "engagement_label": engagement_label(e_new),
        "t_expected": t_exp,
        "t_answer": t_answer,
        "delta_m": delta_m,
        "delta_p": delta_p,
        "reward": reward,
    }


# ─────────────────────────────────────────────
# DEMO / SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("PLS RL Metrics Tracker — Demo")
    print("=" * 60)

    tracker = MetricsTracker(student_id="s001", category="Penggalang")

    # Simulate a learning session with multiple attempts across types
    attempts = [
        # (learning_type, mastery_code, n_attempt, t_answer_seconds, question_idx)
        ("PAR", "1PAR",  3, 120.0,  0),
        ("PAR", "2PAR",  2,  95.0,  1),
        ("PAR", "2PAR",  1,  60.0,  2),
        ("TAR", "1TAR",  4, 150.0,  0),
        ("TAR", "2TAR",  3, 110.0,  1),
        ("PAI", "2PAI",  2,  80.0,  0),
        ("PAI", "3PAI",  1,  55.0,  1),
        ("TGR", "1TGR",  5, 180.0,  0),
        ("TGI", "3TGI",  1,  40.0,  0),
        ("PAR", "3PAR",  1,  50.0,  3),
        ("TAR", "3TAR",  2,  90.0,  2),
    ]

    print("\n📊 Recording attempts...\n")
    for lt, code, n_att, t_ans, q_idx in attempts:
        rec = tracker.record_attempt(lt, code, n_att, t_ans, question_idx=q_idx)
        print(
            f"  [{lt}] code={code} | attempts={n_att} | t={t_ans:.0f}s | "
            f"M={rec['mastery_score']:.2f} P={rec['performance']:.2f} "
            f"E={rec['engagement']:.2f} | R={rec['reward']:+.4f}"
        )

    # Fit MLR weights from accumulated data
    print("\n🔧 Fitting MLR weights from history...")
    weights = tracker.fit_mlr_weights()
    print(f"  β0={weights['beta0']:.4f}  α(perf)={weights['alpha_perf']:.4f}  "
          f"β(mastery)={weights['beta_mastery']:.4f}  γ(eng)={weights['gamma_engagement']:.4f}")
    print(f"  (fitted on {weights['n_samples']} samples)")

    # Full summary
    print("\n📋 Current Metrics per Learning Type:")
    for lt, m in tracker.get_all_metrics().items():
        print(
            f"  {lt:4s} — Mastery: {m['mastery_score']:.2f} [{m['mastery_label']}] | "
            f"Perf: {m['performance']:.2f} [{m['performance_label']}] | "
            f"Eng: {m['engagement']:.2f} [{m['engagement_label']}]"
        )

    # Recommendation
    rec = tracker.recommend_learning_type()
    print(f"\n🎯 Recommended learning type: {rec['recommended']}")
    print(f"   Scores: { {k: round(v,4) for k,v in rec['scores'].items()} }")

    # Single-call helper
    print("\n🧮 compute_metrics_from_dict example:")
    result = compute_metrics_from_dict({
        "mastery_code": "4PAR",
        "n_attempt": 2,
        "t_answer_seconds": 75.0,
        "prev_mastery": 0.60,
        "prev_performance": 0.80,
    }, category="Penggalang")
    print(json.dumps(result, indent=2))

    print("\n✅ Done.")
