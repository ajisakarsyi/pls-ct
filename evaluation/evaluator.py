"""
evaluation/evaluator.py
════════════════════════
Three paper-backed evaluation techniques for the PLS-CT RL system.

Technique 1 — Knowledge Tracing AUC
    Protocol: Liu et al. (2022) pyKT, NeurIPS 2022.
    Metric: AUC-ROC of mastery_score predicting is_correct on held-out last 20% steps.
    Threshold: AUC ≥ 0.72 beats BKT baseline; ≥ 0.80 approaches DKT level.

Technique 2 — Reward Decomposition Analysis
    Protocol: Septon, Amitai & Amir, AAMAS 2023; Milani et al., arXiv 2211.06665.
    Metrics: α/β/γ weight stability across MLR refits; per-component Q-value contribution.

Technique 3 — Offline Policy Evaluation (Doubly Robust)
    Protocol: Zhan, Hadad, Hirshberg & Athey, KDD 2021, arXiv 2106.02029.
    Metric: V̂_DR — unbiased estimate of what a different policy would have earned.
    Policies compared: current ε-greedy vs pure-exploit (ε=0).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from rl_metrics import LEARNING_TYPES


# ══════════════════════════════════════════════════════════════════════════
#  TECHNIQUE 1 — KNOWLEDGE TRACING AUC
# ══════════════════════════════════════════════════════════════════════════

def evaluate_kt_auc(
    steps: List[Dict],
    holdout_frac: float = 0.20,
) -> Dict:
    """
    Knowledge Tracing AUC — pyKT benchmark protocol (Liu et al., NeurIPS 2022).

    Uses mastery_score at each step as the model's prediction of p(correct).
    Evaluates on the held-out last `holdout_frac` of the step sequence.

    Parameters
    ----------
    steps        : agent.step_log  (list of step dicts)
    holdout_frac : fraction of steps to hold out (default 0.20 per pyKT protocol)

    Returns
    -------
    dict with:
        auc           — AUC-ROC score (float or None if not enough data)
        n_train       — number of training steps (80%)
        n_holdout     — number of held-out steps (20%)
        accuracy      — simple threshold accuracy (mastery >= 0.5 → predict correct)
        verdict       — string interpretation against paper benchmarks
        per_lt        — per-LT AUC breakdown
        thresholds    — benchmark values from literature
    """
    if not steps:
        return {"error": "No steps recorded.", "auc": None}

    n_holdout = max(1, int(len(steps) * holdout_frac))
    n_train   = len(steps) - n_holdout

    if n_holdout < 3:
        return {
            "error": f"Not enough steps for evaluation (need ≥ 15, have {len(steps)}).",
            "auc":   None,
            "n_train": n_train, "n_holdout": n_holdout,
        }

    held_out   = steps[n_train:]
    y_pred     = [float(s.get("mastery_score", 0.0)) for s in held_out]
    y_true     = [int(bool(s.get("is_correct", False))) for s in held_out]

    # AUC-ROC (manual implementation — no sklearn dependency required)
    auc = _auc_roc(y_true, y_pred)

    # Accuracy at 0.5 threshold
    predicted_correct = [1 if p >= 0.5 else 0 for p in y_pred]
    accuracy = sum(1 for p, t in zip(predicted_correct, y_true) if p == t) / len(y_true)

    # Per-LT AUC
    per_lt: Dict[str, Optional[float]] = {}
    for lt in LEARNING_TYPES:
        lt_steps = [s for s in held_out if s.get("learning_type") == lt]
        if len(lt_steps) < 3:
            per_lt[lt] = None
            continue
        lt_pred = [float(s.get("mastery_score", 0.0)) for s in lt_steps]
        lt_true = [int(bool(s.get("is_correct", False))) for s in lt_steps]
        per_lt[lt] = _auc_roc(lt_true, lt_pred)

    # Verdict against literature benchmarks
    thresholds = {
        "random_baseline":  0.50,
        "bkt_baseline":     0.72,   # Corbett & Anderson 1994; confirmed by Abdelrahman et al. 2023
        "dkt_target":       0.80,   # Piech et al. 2015
        "simplekt_top3":    0.84,   # Liu et al. (simpleKT) ICLR 2023
    }
    if auc is None:
        verdict = "insufficient_data"
    elif auc < 0.50:
        verdict = "below_random"
    elif auc < 0.72:
        verdict = "below_bkt_baseline"
    elif auc < 0.80:
        verdict = "above_bkt_beats_baseline"
    elif auc < 0.84:
        verdict = "approaching_dkt_level"
    else:
        verdict = "top_tier_dkt"

    return {
        "technique":      "Knowledge Tracing AUC",
        "citation":       "Liu et al. (2022) pyKT, NeurIPS 2022; Abdelrahman et al. (2023), ACM CSUR 55(11)",
        "protocol":       f"Hold-out last {int(holdout_frac*100)}% of steps as test set",
        "n_total":        len(steps),
        "n_train":        n_train,
        "n_holdout":      n_holdout,
        "auc":            round(auc, 4) if auc is not None else None,
        "accuracy":       round(accuracy, 4),
        "verdict":        verdict,
        "thresholds":     thresholds,
        "per_lt_auc":     {lt: (round(v, 4) if v is not None else None) for lt, v in per_lt.items()},
        "interpretation": _kt_interpretation(auc, thresholds),
    }


def _auc_roc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """
    Compute AUC-ROC without sklearn.
    Uses the trapezoidal rule on sorted (FPR, TPR) pairs.
    Returns None if only one class is present in y_true.
    """
    if len(set(y_true)) < 2:
        return None   # Can't compute AUC with a single class

    # Sort by descending score
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    pos   = sum(y_true)
    neg   = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None

    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    prev_score = None

    for score, label in pairs:
        if score != prev_score:
            fpr = fp / neg
            tpr = tp / pos
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr = fpr, tpr
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    # Final point
    fpr = fp / neg
    tpr = tp / pos
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return float(auc)


def _kt_interpretation(auc: Optional[float], thresholds: Dict) -> str:
    if auc is None:
        return "Insufficient data for AUC computation."
    if auc < thresholds["random_baseline"]:
        return (
            "AUC below 0.50 — mastery scores are anticorrelated with actual correctness. "
            "The mastery tracker may be inverting signal; review mastery update logic."
        )
    if auc < thresholds["bkt_baseline"]:
        return (
            f"AUC {auc:.3f} — below the BKT baseline (0.72). "
            "Mastery tracking is informative but not yet reliable. "
            "Consider increasing seeding length or adjusting the mastery promotion threshold."
        )
    if auc < thresholds["dkt_target"]:
        return (
            f"AUC {auc:.3f} — above BKT (0.72), approaching DKT target (0.80). "
            "Mastery tracking is credible. The ΔM component in the reward function "
            "is reflecting real learning gain."
        )
    return (
        f"AUC {auc:.3f} — at or above DKT level (0.80). "
        "Mastery tracking is highly reliable. ΔM is a strong signal in the reward function."
    )


# ══════════════════════════════════════════════════════════════════════════
#  TECHNIQUE 2 — REWARD DECOMPOSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def evaluate_reward_decomposition(steps: List[Dict], agent) -> Dict:
    """
    Reward Decomposition Analysis.

    Evaluates:
    1. Weight stability — do α, β, γ converge over MLR refits?
    2. Component contribution — which of ΔP, ΔM, E dominates the reward signal?
    3. Q-value decomposition — which component drives LT preference?

    Citations:
        Septon, Amitai & Amir (AAMAS 2023)
        Feit et al. arXiv:2307.04098 (2023)
        Milani et al. arXiv:2211.06665 (2022, updated 2025)

    Parameters
    ----------
    steps : agent.step_log
    agent : RLAgent instance
    """
    if not steps:
        return {"error": "No steps recorded.", "stable": None}

    # ── Extract MLR refit events ───────────────────────────────────────────
    refit_events = [s for s in steps if s.get("mlr_refitted") and s.get("mlr_weights")]
    weight_trajectory: List[Dict] = []
    for s in refit_events:
        w = s["mlr_weights"]
        weight_trajectory.append({
            "step_num": s["step_num"],
            "alpha":    round(w.get("beta0", w.get("alpha_perf",    0)), 5),
            "beta":     round(w.get("beta",  w.get("beta_mastery",  0)), 5),
            "gamma":    round(w.get("gamma", w.get("gamma_engagement", 0)), 5),
        })

    # ── Weight stability ──────────────────────────────────────────────────
    stable_alpha = stable_beta = stable_gamma = None
    cv_alpha = cv_beta = cv_gamma = None

    if len(weight_trajectory) >= 3:
        alphas = [w["alpha"] for w in weight_trajectory]
        betas  = [w["beta"]  for w in weight_trajectory]
        gammas = [w["gamma"] for w in weight_trajectory]

        def cv(xs: List[float]) -> Optional[float]:
            mean = np.mean(xs)
            if abs(mean) < 1e-9:
                return None
            return float(np.std(xs) / abs(mean))

        cv_alpha = cv(alphas)
        cv_beta  = cv(betas)
        cv_gamma = cv(gammas)

        STABLE_THRESHOLD = 0.30   # CV < 30% → converged
        stable_alpha = (cv_alpha is not None and cv_alpha < STABLE_THRESHOLD)
        stable_beta  = (cv_beta  is not None and cv_beta  < STABLE_THRESHOLD)
        stable_gamma = (cv_gamma is not None and cv_gamma < STABLE_THRESHOLD)
        overall_stable = stable_alpha and stable_beta and stable_gamma
    else:
        overall_stable = None

    # ── Current weights ───────────────────────────────────────────────────
    current_weights = {
        "alpha_delta_p": round(agent.tracker.alpha, 5),
        "beta_delta_m":  round(agent.tracker.beta,  5),
        "gamma_e":       round(agent.tracker.gamma, 5),
        "beta0":         round(agent.tracker.beta0, 5),
    }

    # ── Component contribution to actual rewards ──────────────────────────
    if steps:
        delta_p_vals = [s.get("delta_p", 0.0) for s in steps]
        delta_m_vals = [s.get("delta_m", 0.0) for s in steps]
        engage_vals  = [s.get("engagement",  0.0) for s in steps]

        alpha = agent.tracker.alpha
        beta  = agent.tracker.beta
        gamma = agent.tracker.gamma

        contrib_p = float(np.mean([alpha * abs(v) for v in delta_p_vals]))
        contrib_m = float(np.mean([beta  * abs(v) for v in delta_m_vals]))
        contrib_e = float(np.mean([gamma * v      for v in engage_vals]))
        total_contrib = contrib_p + contrib_m + contrib_e

        def pct(x): return round(x / total_contrib * 100, 1) if total_contrib > 0 else 0

        component_contribution = {
            "delta_p": {"mean_weighted": round(contrib_p, 5), "pct": pct(contrib_p)},
            "delta_m": {"mean_weighted": round(contrib_m, 5), "pct": pct(contrib_m)},
            "engagement": {"mean_weighted": round(contrib_e, 5), "pct": pct(contrib_e)},
        }
        dominant = max(
            ["delta_p", "delta_m", "engagement"],
            key=lambda k: component_contribution[k]["mean_weighted"],
        )
    else:
        component_contribution = {}
        dominant = None

    # ── Q-value decomposition per LT ──────────────────────────────────────
    # Q(s,a) ≈ α·P(a) + β·M(a) + γ·E(a)  where P, M, E are current state values
    q_decomposition: Dict[str, Dict] = {}
    for lt in LEARNING_TYPES:
        s_lt = agent.tracker.states[lt]
        q_p  = round(agent.tracker.alpha * s_lt.performance,   5)
        q_m  = round(agent.tracker.beta  * s_lt.mastery_score, 5)
        q_e  = round(agent.tracker.gamma * s_lt.engagement,    5)
        q_decomposition[lt] = {
            "q_delta_p":    q_p,
            "q_delta_m":    q_m,
            "q_engagement": q_e,
            "q_total_approx": round(q_p + q_m + q_e, 5),
        }

    # ── Verdict ───────────────────────────────────────────────────────────
    if overall_stable is None:
        stability_verdict = "insufficient_refits"
    elif overall_stable:
        stability_verdict = "converged"
    else:
        n_unstable = sum(1 for x in [stable_alpha, stable_beta, stable_gamma] if x is False)
        stability_verdict = f"{n_unstable}_weights_fluctuating"

    return {
        "technique":   "Reward Decomposition Analysis",
        "citation":    (
            "Septon, Amitai & Amir (AAMAS 2023); "
            "Feit et al. arXiv:2307.04098 (2023); "
            "Milani et al. arXiv:2211.06665 (2022)"
        ),
        "n_steps":              len(steps),
        "n_mlr_refits":         len(refit_events),
        "current_weights":      current_weights,
        "weight_trajectory":    weight_trajectory,
        "stability": {
            "verdict":        stability_verdict,
            "overall_stable": overall_stable,
            "cv_alpha":       round(cv_alpha, 4) if cv_alpha is not None else None,
            "cv_beta":        round(cv_beta,  4) if cv_beta  is not None else None,
            "cv_gamma":       round(cv_gamma, 4) if cv_gamma is not None else None,
            "stable_alpha":   stable_alpha,
            "stable_beta":    stable_beta,
            "stable_gamma":   stable_gamma,
        },
        "component_contribution": component_contribution,
        "dominant_component":     dominant,
        "q_value_decomposition":  q_decomposition,
        "interpretation":         _rd_interpretation(
            overall_stable, dominant, current_weights, len(refit_events)
        ),
    }


def _rd_interpretation(stable, dominant, weights, n_refits) -> str:
    parts = []
    if n_refits < 3:
        parts.append(
            f"Only {n_refits} MLR refit(s) recorded — need ≥ 3 to assess stability. "
            "Continue interacting to accumulate enough data."
        )
    elif stable is True:
        parts.append(
            "Weights α, β, γ have converged (CV < 30% across refits). "
            "The reward signal is well-calibrated."
        )
    elif stable is False:
        parts.append(
            "One or more weights are still fluctuating (CV ≥ 30%). "
            "The reward signal has not yet stabilised — "
            "more interactions will improve calibration."
        )

    if dominant == "delta_m":
        parts.append(
            f"ΔM (mastery) is the dominant reward component "
            f"(β={weights['beta_delta_m']:.3f}). "
            "The agent is primarily rewarding knowledge gain — as intended."
        )
    elif dominant == "engagement":
        parts.append(
            f"E (engagement) is dominating over ΔM "
            f"(γ={weights['gamma_e']:.3f} vs β={weights['beta_delta_m']:.3f}). "
            "The agent may be optimising for speed rather than mastery. "
            "Consider reducing γ or increasing β."
        )
    elif dominant == "delta_p":
        parts.append(
            f"ΔP (performance) is dominating "
            f"(α={weights['alpha_delta_p']:.3f}). "
            "The agent rewards correctness changes most strongly."
        )
    return "  ".join(parts) if parts else "Insufficient data for interpretation."


# ══════════════════════════════════════════════════════════════════════════
#  TECHNIQUE 3 — OFFLINE POLICY EVALUATION (DOUBLY ROBUST)
# ══════════════════════════════════════════════════════════════════════════

def evaluate_ope_dr(
    steps:     List[Dict],
    agent,
    n_actions: int = 8,
) -> Dict:
    """
    Offline Policy Evaluation with Doubly Robust (DR) estimator.

    Estimates what reward a pure-exploit policy (ε=0) would have earned
    on the same interactions logged by the current ε-greedy policy.

    Formula (Zhan et al., KDD 2021):
        V̂_DR = r̂(s,a) + (π_new / π_log) * (r − r̂)

    where:
        π_log(a|s) = ε/n  + (1−ε)  if a == argmax Q  (logging policy)
                   = ε/n            otherwise
        π_new(a|s) = 1.0            if a == argmax Q  (pure-exploit)
                   = 0.0            otherwise
        r̂(s,a)    = mean reward for LT `a` across all logged steps (direct model)

    Parameters
    ----------
    steps     : agent.step_log
    agent     : RLAgent instance
    n_actions : number of LTs (default 8)

    Returns
    -------
    dict with V_hat_DR, comparison to logged policy, and per-LT breakdown.
    """
    if not steps:
        return {"error": "No steps recorded.", "v_hat_dr": None}

    if len(steps) < 5:
        return {
            "error": f"Too few steps ({len(steps)}) for reliable OPE (need ≥ 5).",
            "v_hat_dr": None,
        }

    # ── Direct model: mean reward per LT ─────────────────────────────────
    lt_rewards: Dict[str, List[float]] = {lt: [] for lt in LEARNING_TYPES}
    for s in steps:
        lt_rewards[s["learning_type"]].append(s["reward"])
    r_hat: Dict[str, float] = {
        lt: float(np.mean(v)) if v else 0.0
        for lt, v in lt_rewards.items()
    }

    # ── DR estimates ──────────────────────────────────────────────────────
    v_logged  = []  # actual policy value
    v_dr_list = []  # DR-adjusted estimates
    iw_list   = []  # importance weights

    for s in steps:
        lt      = s["learning_type"]
        r       = float(s["reward"])
        epsilon = float(s.get("epsilon", 0.20))

        # Logging policy probability
        q_after = s.get("q_values_after", {})
        if q_after:
            best_lt_logged = max(q_after, key=q_after.__getitem__)
        else:
            best_lt_logged = lt  # fallback

        if lt == best_lt_logged:
            pi_log = epsilon / n_actions + (1.0 - epsilon)
        else:
            pi_log = epsilon / n_actions

        # New policy: pure-exploit
        if lt == best_lt_logged:
            pi_new = 1.0
        else:
            pi_new = 0.0

        # Clip importance weight to avoid extreme variance
        iw  = min(pi_new / max(pi_log, 1e-9), 20.0)
        rh  = r_hat[lt]
        dr  = rh + iw * (r - rh)

        v_logged.append(r)
        v_dr_list.append(dr)
        iw_list.append(iw)

    v_hat_logged = float(np.mean(v_logged))
    v_hat_dr     = float(np.mean(v_dr_list))
    v_hat_dm     = float(np.mean(list(r_hat.values())))  # direct model baseline

    gain     = v_hat_dr - v_hat_logged
    gain_pct = (gain / abs(v_hat_logged) * 100) if abs(v_hat_logged) > 1e-9 else 0.0

    # ── Clipping stats (high IW = high variance risk) ─────────────────────
    clipped = sum(1 for iw in iw_list if iw >= 20.0)
    mean_iw = float(np.mean(iw_list))

    # ── Per-LT breakdown ──────────────────────────────────────────────────
    lt_breakdown: Dict[str, Dict] = {}
    for lt in LEARNING_TYPES:
        lt_steps = [s for s in steps if s["learning_type"] == lt]
        lt_breakdown[lt] = {
            "n_logged":     len(lt_steps),
            "mean_reward":  round(r_hat[lt], 5),
            "selected_as_best_pct": round(
                sum(
                    1 for s in lt_steps
                    if s.get("q_values_after") and
                    max(s["q_values_after"], key=s["q_values_after"].get) == lt
                ) / max(len(lt_steps), 1) * 100, 1
            ),
        }

    # ── Verdict ───────────────────────────────────────────────────────────
    if gain > 0.005:
        verdict = "pure_exploit_would_earn_more"
    elif gain < -0.005:
        verdict = "exploration_is_beneficial"
    else:
        verdict = "policies_equivalent"

    return {
        "technique": "Offline Policy Evaluation — Doubly Robust",
        "citation":  (
            "Zhan, Hadad, Hirshberg & Athey (KDD 2021, arXiv:2106.02029); "
            "Kiyohara et al. (WSDM 2022); Saito & Joachims (ICML 2022)"
        ),
        "policies_compared": {
            "logging": "epsilon-greedy (current system)",
            "target":  "pure-exploit (epsilon=0, always picks argmax Q)",
        },
        "n_steps":            len(steps),
        "v_hat_logged":       round(v_hat_logged, 5),
        "v_hat_dr":           round(v_hat_dr, 5),
        "v_hat_direct_model": round(v_hat_dm, 5),
        "estimated_gain":     round(gain, 5),
        "estimated_gain_pct": round(gain_pct, 2),
        "verdict":            verdict,
        "iw_stats": {
            "mean_importance_weight": round(mean_iw, 3),
            "clipped_steps":          clipped,
            "clipped_pct":            round(clipped / len(steps) * 100, 1),
            "note": (
                "High clipping (> 20%) = high variance; "
                "estimate less reliable."
                if clipped / len(steps) > 0.20 else
                "Clipping is low; DR estimate is reliable."
            ),
        },
        "per_lt_breakdown": lt_breakdown,
        "interpretation": _ope_interpretation(verdict, gain, mean_iw, clipped, len(steps)),
    }


def _ope_interpretation(verdict, gain, mean_iw, clipped, n_steps) -> str:
    parts = []
    if verdict == "pure_exploit_would_earn_more":
        parts.append(
            f"Pure-exploit policy would earn an estimated {gain:+.4f} more reward/step. "
            "The agent may be over-exploring — consider reducing epsilon or "
            "shortening the seeding phase."
        )
    elif verdict == "exploration_is_beneficial":
        parts.append(
            f"Exploration is earning {abs(gain):.4f} more reward/step than pure-exploit. "
            "The current ε-greedy balance is appropriate; reducing epsilon would hurt performance."
        )
    else:
        parts.append(
            "Both policies estimate similar values — "
            "epsilon and exploit produce comparable outcomes at this stage."
        )

    if clipped / n_steps > 0.20:
        parts.append(
            f"Warning: {clipped}/{n_steps} importance weights were clipped. "
            "DR estimate has higher variance. "
            "More interaction data will improve estimate reliability."
        )
    return "  ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  COMBINED EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_all_evaluations(session_id: str, agent) -> Dict:
    """
    Run all three evaluation techniques on a session and return a combined report.

    Best practice recommendation (based on papers):
    - If KT AUC < 0.72: focus on improving mastery tracking before tuning RL.
    - If Reward Decomp shows γ > β: the agent is optimising engagement over mastery.
    - If OPE DR shows exploration is harmful: reduce epsilon / shorten seeding.
    """
    steps = agent.step_log

    kt_result   = evaluate_kt_auc(steps)
    rd_result   = evaluate_reward_decomposition(steps, agent)
    ope_result  = evaluate_ope_dr(steps, agent)

    # ── System-level recommendation ───────────────────────────────────────
    recommendations = []

    auc = kt_result.get("auc")
    if auc is not None and auc < 0.72:
        recommendations.append(
            "KT AUC < 0.72: mastery tracking is below the BKT baseline. "
            "Consider increasing CONSECUTIVE_TO_PROMOTE threshold or "
            "reviewing the mastery level score mapping."
        )
    elif auc is not None and auc >= 0.80:
        recommendations.append(
            "KT AUC ≥ 0.80: mastery signal is reliable. "
            "The ΔM component is accurately measuring knowledge gain."
        )

    dominant = rd_result.get("dominant_component")
    if dominant == "engagement":
        recommendations.append(
            "Engagement (E) is dominating reward. "
            "The agent may converge to fast-answering LTs rather than high-mastery ones. "
            "Increase β (mastery weight) relative to γ (engagement weight)."
        )
    if rd_result.get("stability", {}).get("overall_stable") is False:
        recommendations.append(
            "MLR weights are still fluctuating. "
            "Run more sessions before drawing conclusions from OPE."
        )

    ope_verdict = ope_result.get("verdict")
    if ope_verdict == "pure_exploit_would_earn_more":
        recommendations.append(
            "OPE shows pure-exploit beats ε-greedy. "
            "Reduce EPSILON_INIT or EPSILON_DECAY to exploit faster."
        )
    elif ope_verdict == "exploration_is_beneficial":
        recommendations.append(
            "OPE shows current exploration strategy is earning more than pure-exploit. "
            "Keep current ε schedule."
        )

    if not recommendations:
        recommendations.append(
            "All three metrics are within acceptable ranges. "
            "System is performing as expected."
        )

    return {
        "session_id":          session_id,
        "n_steps":             len(steps),
        "knowledge_tracing":   kt_result,
        "reward_decomposition": rd_result,
        "offline_policy_eval": ope_result,
        "system_recommendations": recommendations,
        "best_for_next_step": (
            "Collect at least 20+ interactions per student before trusting OPE-DR. "
            "KT AUC and Reward Decomposition are reliable from 10+ steps. "
            "Priority: fix KT AUC first (validates ΔM), then check weight stability, "
            "then use OPE to tune epsilon."
        ),
    }
