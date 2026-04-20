"""
param_sweep.py
───────────────
Offline hyperparameter sweep for CSIPBLLM RL agent.
Simulates 30 students × 20 questions each under different configs,
then evaluates with all three built-in techniques:
  - Knowledge Tracing AUC
  - Reward Decomposition
  - OPE Doubly Robust
"""

import sys, os, copy, random, statistics
sys.path.insert(0, ".")

import numpy as np
from rl_metrics import (
    MetricsTracker, compute_performance, compute_engagement,
    compute_expected_time, MASTERY_LEVEL_SCORE, N_MAX
)

PROFILES = [
    {"name": "PAR_strong",  "true_lt": "PAR", "base_p": 0.82},
    {"name": "TAR_strong",  "true_lt": "TAR", "base_p": 0.78},
    {"name": "TGR_strong",  "true_lt": "TGR", "base_p": 0.80},
    {"name": "PGI_strong",  "true_lt": "PGI", "base_p": 0.75},
    {"name": "TAI_strong",  "true_lt": "TAI", "base_p": 0.77},
    {"name": "Mixed_A",     "true_lt": "PAI", "base_p": 0.70},
    {"name": "Mixed_B",     "true_lt": "TGI", "base_p": 0.68},
    {"name": "Struggling",  "true_lt": "PAR", "base_p": 0.55},
]

LEARNING_TYPES = ["PAR", "TAR", "PAI", "TAI", "TGR", "TGI", "PGR", "PGI"]


def lt_match_factor(selected_lt: str, true_lt: str) -> float:
    if selected_lt == true_lt:
        return 1.00
    if selected_lt[0] == true_lt[0] or selected_lt[-1] == true_lt[-1]:
        return 0.75
    return 0.55


def simulate_student(
    profile: dict,
    n_questions: int,
    epsilon_init: float,
    epsilon_decay: float,
    epsilon_min: float,
    w_init_low: float,
    w_init_high: float,
    consecutive_to_promote: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n_actions = len(LEARNING_TYPES)
    state_dim = 3
    W = np_rng.uniform(w_init_low, w_init_high, size=(n_actions, state_dim))
    epsilon = epsilon_init
    mastery_levels = {lt: 1 for lt in LEARNING_TYPES}
    consecutive_correct = {lt: 0 for lt in LEARNING_TYPES}
    consecutive_wrong   = {lt: 0 for lt in LEARNING_TYPES}
    tracker = MetricsTracker(student_id=f"{profile['name']}_{seed}", category="Penggalang")

    step_log = []
    prev_perf = {lt: 0.0 for lt in LEARNING_TYPES}
    prev_mast = {lt: MASTERY_LEVEL_SCORE[1] for lt in LEARNING_TYPES}

    for q_idx in range(n_questions):
        state = np.array([
            np.mean([MASTERY_LEVEL_SCORE[mastery_levels[lt]] for lt in LEARNING_TYPES]),
            np.mean([tracker.states[lt].performance for lt in LEARNING_TYPES]),
            np.mean([tracker.states[lt].engagement  for lt in LEARNING_TYPES]),
        ], dtype=float)

        q_vals = {lt: float(W[i] @ state) for i, lt in enumerate(LEARNING_TYPES)}
        if rng.random() < epsilon:
            lt = rng.choice(LEARNING_TYPES)
        else:
            lt = max(q_vals, key=q_vals.__getitem__)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        match   = lt_match_factor(lt, profile["true_lt"])
        p_corr  = min(0.97, profile["base_p"] * match)
        is_corr = rng.random() < p_corr

        if is_corr:
            n_attempt = rng.randint(1, 3) if p_corr > 0.7 else rng.randint(1, N_MAX)
        else:
            n_attempt = N_MAX

        t_ans = rng.uniform(20, 90)
        mastery_code = f"{mastery_levels[lt]}{lt}"

        record = tracker.record_attempt(
            learning_type    = lt,
            mastery_code     = mastery_code,
            n_attempt        = n_attempt,
            t_answer_seconds = t_ans,
            question_idx     = q_idx,
            is_correct       = is_corr,
        )

        action_idx = LEARNING_TYPES.index(lt)
        reward     = record["reward"]
        td_error   = reward - float(W[action_idx] @ state)
        lr         = 0.1
        W[action_idx] += lr * td_error * state

        if is_corr:
            consecutive_correct[lt] += 1
            consecutive_wrong[lt] = 0
            if (consecutive_correct[lt] >= consecutive_to_promote
                    and mastery_levels[lt] < 6):
                mastery_levels[lt] += 1
                consecutive_correct[lt] = 0
        else:
            consecutive_correct[lt] = 0
            consecutive_wrong[lt] += 1
            if consecutive_wrong[lt] >= 2 and mastery_levels[lt] > 1:
                mastery_levels[lt] -= 1
                consecutive_wrong[lt] = 0

        step_log.append({
            "step_num":       q_idx + 1,
            "learning_type":  lt,
            "is_correct":     is_corr,
            "n_attempt":      n_attempt,
            "mastery_level":  mastery_levels[lt],
            "mastery_score":  record["mastery_score"],
            "performance":    record["performance"],
            "engagement":     record["engagement"],
            "reward":         reward,
            "delta_p":        record["delta_p"],
            "delta_m":        record["delta_m"],
            "epsilon":        epsilon,
            "q_values_after": dict(q_vals),
            "phase":          "free",
            "mlr_refitted":   False,
            "mlr_weights":    None,
        })

    return {"step_log": step_log, "tracker": tracker}


# ── Evaluators ───────────────────────────────────────────────────────────

def auc_roc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    pos = sum(y_true); neg = len(y_true) - pos
    if pos == 0 or neg == 0: return None
    tp = fp = prev_fpr = prev_tpr = auc = 0.0
    prev_score = None
    for score, label in pairs:
        if score != prev_score:
            fpr = fp/neg; tpr = tp/pos
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr, prev_score = fpr, tpr, score
        tp += label == 1; fp += label != 1
    fpr = fp/neg; tpr = tp/pos
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return float(auc)


def eval_kt_auc(step_log):
    n = len(step_log)
    if n < 10: return None
    holdout = step_log[int(n * 0.8):]
    y_true = [int(s["is_correct"]) for s in holdout]
    y_pred = [float(s["mastery_score"]) for s in holdout]
    return auc_roc(y_true, y_pred)


def eval_reward_stability(step_log):
    rewards = [s["reward"] for s in step_log]
    if not rewards or statistics.mean(rewards) == 0: return None
    try:
        return statistics.stdev(rewards) / abs(statistics.mean(rewards))
    except statistics.StatisticsError:
        return None


def eval_exploration_efficiency(step_log, true_lt):
    for i, s in enumerate(step_log):
        if i >= 4:
            last5 = [step_log[j]["learning_type"] for j in range(i-4, i+1)]
            if all(lt == true_lt for lt in last5):
                return i + 1
    return len(step_log)


def eval_ope_dr(step_log):
    if len(step_log) < 5: return None
    lt_rewards = {}
    for s in step_log:
        lt_rewards.setdefault(s["learning_type"], []).append(s["reward"])
    r_hat = {lt: np.mean(v) for lt, v in lt_rewards.items()}

    v_log = []; v_dr = []
    for s in step_log:
        lt = s["learning_type"]; r = s["reward"]; eps = s["epsilon"]
        q = s.get("q_values_after", {})
        best = max(q, key=q.get) if q else lt
        pi_log = eps/8 + (1-eps) if lt == best else eps/8
        pi_new = 1.0 if lt == best else 0.0
        iw = min(pi_new / max(pi_log, 1e-9), 20.0)
        rh = r_hat.get(lt, 0.0)
        v_log.append(r)
        v_dr.append(rh + iw * (r - rh))
    return float(np.mean(v_dr)) - float(np.mean(v_log))


# ── Parameter grid ────────────────────────────────────────────────────────

CONFIGS = [
    # name, eps_init, eps_decay, eps_min, w_low, w_high, consec
    ("Low Exploration",              0.20, 0.995, 0.05, 0.001, 0.01, 2),
    ("Low-Med Exploration",          0.30, 0.98,  0.05, 0.01,  0.05, 1),
    ("Medium Exploration, High W Init", 0.40, 0.97, 0.05, 0.05, 0.15, 1),
    ("Current",                      0.50, 0.97,  0.05, 0.05,  0.15, 1),
    ("High Exploration A",           0.60, 0.96,  0.05, 0.05,  0.15, 1),
    ("High Exploration B",           0.70, 0.95,  0.05, 0.05,  0.20, 1),
    ("Very High Exploration",        0.80, 0.94,  0.05, 0.05,  0.20, 1),
    ("Fast Decay",                   0.50, 0.90,  0.05, 0.05,  0.15, 1),
    ("Slow Decay",                   0.50, 0.99,  0.05, 0.05,  0.15, 1),
    ("High W Init",                  0.50, 0.97,  0.05, 0.10,  0.20, 1),
]

N_STUDENTS   = 30
N_QUESTIONS  = 20
BASE_SEED    = 42

print(f"Running {len(CONFIGS)} configs × {N_STUDENTS} students × {N_QUESTIONS} Q ...\n")

results = []
for cfg in CONFIGS:
    name, eps_i, eps_d, eps_m, w_lo, w_hi, consec = cfg

    aucs, stabs, opes, conv_qs = [], [], [], []

    for s_idx in range(N_STUDENTS):
        profile = PROFILES[s_idx % len(PROFILES)]
        seed    = BASE_SEED + s_idx * 100

        sim = simulate_student(
            profile, N_QUESTIONS,
            epsilon_init=eps_i, epsilon_decay=eps_d, epsilon_min=eps_m,
            w_init_low=w_lo, w_init_high=w_hi,
            consecutive_to_promote=consec,
            seed=seed,
        )
        log = sim["step_log"]

        a = eval_kt_auc(log)
        if a is not None: aucs.append(a)

        st = eval_reward_stability(log)
        if st is not None: stabs.append(st)

        ope = eval_ope_dr(log)
        if ope is not None: opes.append(ope)

        cq = eval_exploration_efficiency(log, profile["true_lt"])
        conv_qs.append(cq)

    results.append({
        "name":      name,
        "eps_init":  eps_i,
        "eps_decay": eps_d,
        "eps_min":   eps_m,
        "w_low":     w_lo,
        "w_high":    w_hi,
        "consec":    consec,
        "kt_auc":    statistics.mean(aucs)   if aucs   else 0,
        "reward_cv": statistics.mean(stabs)  if stabs  else 99,
        "ope_gain":  statistics.mean(opes)   if opes   else 0,
        "conv_q":    statistics.mean(conv_qs),
    })
    print(f"  {name:<38} KT_AUC={results[-1]['kt_auc']:.4f}  "
          f"RewardCV={results[-1]['reward_cv']:.3f}  "
          f"OPE_gain={results[-1]['ope_gain']:+.4f}")

# ── Per-metric ranking (no composite score) ───────────────────────────────
SEP = "=" * 72

print("\n" + SEP)
print("RESULTS — Raw metrics per configuration")
print(SEP)
print(f"  {'Configuration':<38} {'KT AUC':>8}  {'Reward CV':>10}  {'OPE Gain':>10}")
print(f"  {'-'*38} {'-'*8}  {'-'*10}  {'-'*10}")
for r in results:
    print(f"  {r['name']:<38} {r['kt_auc']:>8.4f}  {r['reward_cv']:>10.3f}  {r['ope_gain']:>+10.4f}")

print("\n" + SEP)
print("PER-METRIC WINNERS")
print(SEP)

by_kt  = sorted(results, key=lambda x: -x["kt_auc"])
by_cv  = sorted(results, key=lambda x:  x["reward_cv"])
by_ope = sorted(results, key=lambda x: -x["ope_gain"])
by_cq  = sorted(results, key=lambda x:  x["conv_q"])

print("\n  KT AUC  (higher = better mastery tracking):")
for i, r in enumerate(by_kt[:3]):
    print(f"    #{i+1}  {r['name']:<38}  {r['kt_auc']:.4f}")

print("\n  Reward CV  (lower = more stable reward signal):")
for i, r in enumerate(by_cv[:3]):
    cv_note = "  ← catastrophic" if r["reward_cv"] > 5 else ""
    print(f"    #{i+1}  {r['name']:<38}  {r['reward_cv']:.3f}{cv_note}")

print("\n  OPE Gain  (positive = exploration is helping):")
for i, r in enumerate(by_ope[:3]):
    print(f"    #{i+1}  {r['name']:<38}  {r['ope_gain']:+.4f}")

print("\n  Convergence Speed  (fewer Q = faster):")
for i, r in enumerate(by_cq[:3]):
    print(f"    #{i+1}  {r['name']:<38}  {r['conv_q']:.1f} Q")

# Top-2 finish tally
print("\n" + SEP)
print("TOP-2 FINISHES PER CONFIGURATION")
print(SEP)
top2 = {}
for r in by_kt[:2]:  top2[r["name"]] = top2.get(r["name"], []) + ["KT AUC"]
for r in by_cv[:2]:  top2[r["name"]] = top2.get(r["name"], []) + ["Reward CV"]
for r in by_ope[:2]: top2[r["name"]] = top2.get(r["name"], []) + ["OPE Gain"]
for r in by_cq[:2]:  top2[r["name"]] = top2.get(r["name"], []) + ["Convergence"]
for name, wins in sorted(top2.items(), key=lambda x: -len(x[1])):
    print(f"  {name:<38}  top-2 in: {', '.join(wins)}")

best_name = max(top2, key=lambda k: len(top2[k]))
best = next(r for r in results if r["name"] == best_name)
print("\n" + SEP)
print(f"BEST OVERALL: {best['name']}")
print(f"  EPSILON_INIT           = {best['eps_init']}")
print(f"  EPSILON_DECAY          = {best['eps_decay']}")
print(f"  EPSILON_MIN            = {best['eps_min']}")
print(f"  W init range           = ({best['w_low']}, {best['w_high']})")
print(f"  CONSECUTIVE_TO_PROMOTE = {best['consec']}")
print(SEP)