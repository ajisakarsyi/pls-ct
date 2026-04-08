"""
simulate_rl/profiles.py
────────────────────────
Student profiles (p_correct + speed per LT) and the StudentSimulator class.

Each profile encodes a realistic student archetype.  p_correct drives the
reward signal; speed drives engagement (faster = more engaged).

Radar-calibrated profiles (e.g. student_tgi) derive p_correct directly from
the cognitive radar chart ring distances — adjacent LTs on the radar get
nearly the same p_correct as the home LT; opposite ones get the lowest.
"""

from __future__ import annotations

import json
import random
from typing import Dict, Tuple

import numpy as np

from rl_metrics import LEARNING_TYPES, CATEGORY_CONFIG

# ── ANSI terminal colours ─────────────────────────────────────────────────────
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
SEED_SHADE = "#fffde7"


def lt_col(lt: str, text: str) -> str:
    return LT_COLOURS.get(lt, "") + text + X

def reward_col(r: float) -> str:
    if r >  0.10: return f"{G}{r:+.4f}{X}"
    if r >= 0:    return f"{Y}{r:+.4f}{X}"
    return f"{R}{r:+.4f}{X}"

def bar(v: float, w: int = 16) -> str:
    filled = max(0, min(w, round(v * w)))
    return "█" * filled + "░" * (w - filled)


# ── Base profiles ─────────────────────────────────────────────────────────────

PROFILES: Dict[str, Dict] = {
    "random": {
        "label": "Random Student",
        "desc":  "Equal ability across all LTs — forces maximum exploration.",
        "lt":    {lt: {"p_correct": 0.50, "speed": 0.70} for lt in LEARNING_TYPES},
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
        "desc":  "Starts weak, steadily improves across all LTs within each session.",
        "lt":    {lt: {"p_correct": 0.28, "speed": 0.95} for lt in LEARNING_TYPES},
        "noise": 0.15,
        "improvement_rate": 0.018,
    },
    "volatile": {
        "label": "Volatile Student",
        "desc":  "High variance — tests agent robustness to noise.",
        "lt":    {lt: {"p_correct": 0.52, "speed": 0.72} for lt in LEARNING_TYPES},
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
    # ── Radar-calibrated profiles ─────────────────────────────────────────────
    # p_correct mirrors radar chart ring distances from the home LT.
    # Ring 6 = home (best), ring 1 = no shared cognitive dimensions (worst).
    "student_tgi": {
        "label": "TGI-type student (radar-calibrated)",
        "desc": (
            "p_correct mirrors TGI radar rings: "
            "TGI=6 > PAI=5 (near-equal) > TGR=4 > PGI=3 > PGR=2 > TAI=2 > TAR=2 > PAR=1"
        ),
        "lt": {
            "TGI": {"p_correct": 0.85, "speed": 0.38},  # ring 6 — home LT
            "PAI": {"p_correct": 0.81, "speed": 0.41},  # ring 5 — shares G+I, nearly as good
            "TGR": {"p_correct": 0.67, "speed": 0.57},  # ring 4 — shares T+G
            "PGI": {"p_correct": 0.54, "speed": 0.68},  # ring 3 — shares G+I
            "PGR": {"p_correct": 0.44, "speed": 0.78},  # ring 2-3
            "TAI": {"p_correct": 0.34, "speed": 0.88},  # ring 2 — shares T+I only
            "TAR": {"p_correct": 0.31, "speed": 0.91},  # ring 2
            "PAR": {"p_correct": 0.22, "speed": 0.98},  # ring 1 — no shared dims
        },
        "noise": 0.07,
    },
    "par_tar_student": {
        "label": "PAR/TAR student",
        "desc":  "High ability on both PAR and TAR; average on remaining LTs.",
        "lt": {
            "PAR": {"p_correct": 0.82, "speed": 0.40},
            "TAR": {"p_correct": 0.78, "speed": 0.43},
            "PAI": {"p_correct": 0.45, "speed": 0.80},
            "TAI": {"p_correct": 0.42, "speed": 0.82},
            "TGR": {"p_correct": 0.44, "speed": 0.81},
            "TGI": {"p_correct": 0.41, "speed": 0.84},
            "PGR": {"p_correct": 0.46, "speed": 0.79},
            "PGI": {"p_correct": 0.43, "speed": 0.82},
        },
        "noise": 0.10,
    },
}


# ── Auto-generate structural per-LT profiles (student_PAR … student_PGI) ─────

def _shared_dims(lt1: str, lt2: str) -> int:
    return sum(a == b for a, b in zip(lt1, lt2))

for _home_lt in LEARNING_TYPES:
    _cfg: Dict = {}
    for lt in LEARNING_TYPES:
        shared = _shared_dims(_home_lt, lt)
        if   shared == 3: p, spd = 0.78, 0.40
        elif shared == 2: p, spd = 0.63, 0.55
        elif shared == 1: p, spd = 0.52, 0.70
        else:             p, spd = 0.42, 0.82
        _cfg[lt] = {"p_correct": p, "speed": spd}
    PROFILES[f"student_{_home_lt.lower()}"] = {
        "label": f"{_home_lt}-type student",
        "desc":  f"Natural fit is {_home_lt}; p_correct scales by shared cognitive dimensions.",
        "lt":    _cfg,
        "noise": 0.12,
    }


# ── Home LT mapping for automatic seeding ────────────────────────────────────

PROFILE_HOME_LT: Dict[str, str] = {
    "par_tar_student":  "PAR",
    "student_tgi":      "TGI",
    "tgi_lean":         "TGI",
    "par_lean":         "PAR",
    "theoretical_lean": "TAR",
    **{f"student_{lt.lower()}": lt for lt in LEARNING_TYPES},
}


# ── Radar strip ordering (for make_single_line_plot) ─────────────────────────
# Maps profile name → [(LT, ring_level), ...] ordered best→worst

RADAR_RING_ORDERS: Dict[str, list] = {
    "student_tgi": [
        ("TGI", 6), ("PAI", 5), ("TGR", 4), ("PGI", 3),
        ("PGR", 2), ("TAI", 2), ("TAR", 2), ("PAR", 1),
    ],
}

# Default clockwise radar axis order (used when no ring order defined)
RADAR_CLOCKWISE = ["TGI", "PAR", "TGR", "TAI", "TAR", "PGI", "PGR", "PAI"]


# ── Student Simulator ─────────────────────────────────────────────────────────

class StudentSimulator:
    """
    Simulates a student's answer behaviour based on a profile.

    Parameters
    ----------
    profile_name : str   — key in PROFILES
    category     : str   — one of CATEGORY_CONFIG keys
    ai_sim       : bool  — use Claude API to adjust p_correct (requires api_key)
    api_key      : str   — Anthropic API key for ai_sim mode
    """

    def __init__(
        self,
        profile_name: str,
        category:     str  = "Penggalang",
        ai_sim:       bool = False,
        api_key:      str  = "",
    ):
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Valid: {sorted(PROFILES.keys())}"
            )
        self.profile_name = profile_name
        self.profile      = PROFILES[profile_name]
        self.category     = category
        self.ai_sim       = ai_sim
        self.api_key      = api_key
        self.question_num = 0

        total_s, n_q = CATEGORY_CONFIG[category]
        self.t_expected  = total_s / n_q

    def answer(self, lt: str) -> Tuple[bool, int, float]:
        """
        Simulate one answer for a given LT.

        Returns
        -------
        (is_correct, n_attempt, t_answer_seconds)
        """
        self.question_num += 1
        lt_profile = self.profile["lt"][lt]
        p          = float(lt_profile["p_correct"])
        noise_std  = float(self.profile.get("noise", 0.18))

        # Improving profile: p grows within each session
        if "improvement_rate" in self.profile:
            p = min(0.88, p + self.profile["improvement_rate"] * self.question_num)

        p = float(np.clip(p + random.gauss(0, noise_std), 0.05, 0.95))

        if self.ai_sim and self.api_key:
            p = self._ai_adjust_p(lt, p)

        is_correct = random.random() < p
        n_attempt  = (
            (1 if random.random() > (1 - p) * 0.5 else 2)
            if is_correct else random.randint(2, 5)
        )

        speed = float(lt_profile["speed"])
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
                )}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }, method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return max(0.05, min(0.95,
                    float(json.loads(resp.read())["content"][0]["text"].strip())
                ))
        except Exception:
            return base_p
