"""
app/api/routes/rl.py
─────────────────────
All RL-specific endpoints:

  GET  /rl/recommend/{session_id}   — greedy recommendation, no side effects
  GET  /rl/select/{session_id}      — epsilon-greedy select (has side effect)
  GET  /rl/phase/{session_id}       — seeding vs free phase status
  GET  /rl/changes/{session_id}     — full recommendation-change log
  GET  /rl/summary/{session_id}     — end-of-session analytics
  GET  /rl/log/{session_id}         — step-by-step reward/M/P/E log
  GET  /rl/sessions                 — list all active session IDs
  GET  /rl/plots/{session_id}       — all 7 plots as base64 PNGs
  GET  /rl/plots/{session_id}/{name}— single plot as PNG file
  POST /rl/refit                    — force MLR weight refit
  DELETE /rl/session/{session_id}   — delete a session

  (legacy aliases kept for backward compat)
  GET  /metrics/{session_id}
  GET  /pedagogy/{session_id}
"""

import base64
import io
import time
from collections import Counter
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from app.services.rl import (
    rl_registry, session_question_start,
    parse_cognitive, build_cognitive_code,
    print_session_summary, generate_plots,
    MASTERY_LABELS, MASTERY_LEVEL_SCORE, RL_LTS, EPSILON_MIN,
)

router = APIRouter(tags=["RL"])


# ── Recommend (read-only) ─────────────────────────────────────────────────

@router.get("/rl/recommend/{session_id}",
            summary="Rekomendasi LT terbaik — greedy, tanpa side effect")
def rl_recommend(session_id: str, category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    rec   = agent.recommend()
    best  = rec["recommended_lt"]
    level = agent.mastery_levels[best]
    return {
        "session_id":            session_id,
        "recommended_lt":        best,
        "recommended_cognitive": build_cognitive_code(level, best),
        "epsilon":               round(agent.epsilon, 4),
        "phase":                 agent.phase,
        "seeding_lt":            agent.seeding_lt,
        "seeding_remaining":     agent.seeding_remaining,
        "seeding_progress_pct":  round(agent.seeding_progress_pct, 3),
        "per_type": {
            lt: {
                **info,
                "mastery_level": agent.mastery_levels[lt],
                "cognitive":     build_cognitive_code(agent.mastery_levels[lt], lt),
            }
            for lt, info in rec["per_type"].items()
        },
        "reward_weights": rec["reward_weights"],
    }


# ── Select (with side effect — advances epsilon) ──────────────────────────

@router.get("/rl/select/{session_id}",
            summary="Pilih LT via epsilon-greedy (ada side effect)")
def rl_select(session_id: str, category: str = "Penggalang"):
    agent                          = rl_registry.get_agent(session_id, category=category)
    lt, _                          = agent.select_action()
    level                          = agent.mastery_levels[lt]
    session_question_start[session_id] = time.time()
    return {
        "session_id":        session_id,
        "selected_lt":       lt,
        "cognitive":         build_cognitive_code(level, lt),
        "mastery_level":     level,
        "mastery_label":     MASTERY_LABELS.get(level, ""),
        "epsilon":           round(agent.epsilon, 4),
        "mode":              "exploit" if agent.epsilon <= EPSILON_MIN + 0.01 else "explore/exploit",
        "phase":             agent.phase,
        "seeding_lt":        agent.seeding_lt,
        "seeding_remaining": agent.seeding_remaining,
        "q_values":          {l: round(q, 5) for l, q in agent.q_values().items()},
    }


# ── Phase status ──────────────────────────────────────────────────────────

@router.get("/rl/phase/{session_id}",
            summary="Status fase RL: seeding vs free, progress, seeding LT")
def rl_phase(session_id: str, category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    qs    = agent.q_values()
    best  = max(qs, key=qs.__getitem__)
    return {
        "session_id":            session_id,
        "phase":                 agent.phase,
        "seeding_lt":            agent.seeding_lt,
        "seeding_sessions":      agent.seeding_questions,
        "seeding_completed":     agent.total_selections >= agent.seeding_questions,
        "seeding_remaining":     agent.seeding_remaining,
        "seeding_progress_pct":  round(agent.seeding_progress_pct * 100, 1),
        "global_question_count": agent.total_selections,
        "current_recommendation": best,
        "recommendation_changes": len(agent.recommendation_history),
        "last_change": (
            agent.recommendation_history[-1] if agent.recommendation_history else None
        ),
        "message": (
            f"🌱 Seeding phase: {agent.seeding_remaining} questions left "
            f"(using {agent.seeding_lt})"
            if agent.phase == "seeding"
            else f"🆓 Free phase: RL selecting freely (current best: {best})"
        ),
    }


# ── Recommendation change log ─────────────────────────────────────────────

@router.get("/rl/changes/{session_id}",
            summary="Log semua perubahan rekomendasi LT")
def rl_changes(session_id: str, category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    return {
        "session_id":    session_id,
        "total_changes": len(agent.recommendation_history),
        "current_rec":   agent._last_recommendation,
        "phase":         agent.phase,
        "changes":       agent.recommendation_history,
    }


# ── Full summary ──────────────────────────────────────────────────────────

@router.get("/rl/summary/{session_id}",
            summary="Ringkasan lengkap sesi RL: reward, LT usage, phase breakdown, Q-values")
def rl_summary(session_id: str, category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    steps = agent.step_log

    if not steps:
        return {
            "session_id":  session_id,
            "total_steps": 0,
            "message":     "No steps recorded yet for this session.",
        }

    rewards    = [s["reward"] for s in steps]
    seed_steps = [s for s in steps if s.get("phase") == "seeding"]
    free_steps = [s for s in steps if s.get("phase") == "free"]
    correct    = sum(1 for s in steps if s.get("is_correct"))

    lt_counts  = Counter(s["learning_type"] for s in steps)
    lt_rewards = {
        lt: [s["reward"] for s in steps if s["learning_type"] == lt]
        for lt in RL_LTS
    }
    lt_usage = []
    for lt in sorted(RL_LTS, key=lambda x: -np.mean(lt_rewards[x]) if lt_rewards[x] else 0):
        r = lt_rewards[lt]
        lt_usage.append({
            "lt":            lt,
            "questions":     lt_counts.get(lt, 0),
            "avg_reward":    round(float(np.mean(r)), 5) if r else None,
            "sum_reward":    round(float(sum(r)),  5)    if r else None,
            "correct_pct":   round(
                sum(1 for s in steps if s["learning_type"] == lt and s.get("is_correct"))
                / lt_counts[lt] * 100, 1
            ) if lt_counts.get(lt) else None,
            "mastery_level": agent.mastery_levels[lt],
            "mastery_label": MASTERY_LABELS.get(agent.mastery_levels[lt], ""),
        })

    def _phase_block(phase_steps):
        if not phase_steps:
            return None
        r = [s["reward"] for s in phase_steps]
        return {
            "questions":       len(phase_steps),
            "total_reward":    round(float(sum(r)), 5),
            "mean_reward":     round(float(np.mean(r)), 5),
            "correct_pct":     round(
                sum(1 for s in phase_steps if s.get("is_correct")) / len(phase_steps) * 100, 1
            ),
            "lt_distribution": {
                lt: {"count": c, "pct": round(c / len(phase_steps) * 100, 1)}
                for lt, c in Counter(s["learning_type"] for s in phase_steps).most_common()
            },
        }

    print_session_summary(session_id, agent, steps)

    return {
        "session_id":             session_id,
        "total_steps":            len(steps),
        "total_reward":           round(float(sum(rewards)), 5),
        "mean_reward":            round(float(np.mean(rewards)), 5),
        "correct_pct":            round(correct / len(steps) * 100, 1),
        "epsilon":                round(agent.epsilon, 4),
        "phase":                  agent.phase,
        "seeding_lt":             agent.seeding_lt,
        "current_recommendation": agent._current_best_lt(),
        "total_lt_changes":       len(agent.recommendation_history),
        "seeding":                _phase_block(seed_steps),
        "free":                   _phase_block(free_steps),
        "lt_usage":               lt_usage,
        "lt_changes":             agent.recommendation_history,
        "q_values":               {lt: round(q, 6) for lt, q in agent.q_values().items()},
        "reward_weights": {
            "alpha": round(agent.tracker.alpha, 4),
            "beta":  round(agent.tracker.beta,  4),
            "gamma": round(agent.tracker.gamma, 4),
            "beta0": round(agent.tracker.beta0, 4),
        },
        "mastery": {
            lt: {
                "level": agent.mastery_levels[lt],
                "label": MASTERY_LABELS.get(agent.mastery_levels[lt], ""),
                "score": MASTERY_LEVEL_SCORE.get(agent.mastery_levels[lt], 0),
            }
            for lt in RL_LTS
        },
        "current_metrics": agent.tracker.get_all_metrics(),
    }


# ── Step log ──────────────────────────────────────────────────────────────

@router.get("/rl/log/{session_id}",
            summary="Log step-by-step RL (reward, M, P, E, Q-values tiap soal)")
def rl_log(session_id: str, category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    return {
        "session_id": session_id,
        "n_steps":    len(agent.step_log),
        "steps":      agent.step_log,
    }


# ── Sessions list ─────────────────────────────────────────────────────────

@router.get("/rl/sessions", summary="Daftar semua session ID aktif")
def rl_sessions():
    return {
        "sessions": list(rl_registry._agents.keys()),
        "count":    len(rl_registry._agents),
    }


# ── Force MLR refit ───────────────────────────────────────────────────────

@router.post("/rl/refit", summary="Paksa refit bobot MLR dari step log")
def rl_refit(session_id: str = "default", category: str = "Penggalang"):
    agent = rl_registry.get_agent(session_id, category=category)
    try:
        agent.tracker.fit()
        return {
            "session_id": session_id,
            "refitted":   True,
            "alpha": round(agent.tracker.alpha, 4),
            "beta":  round(agent.tracker.beta,  4),
            "gamma": round(agent.tracker.gamma, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Delete session ────────────────────────────────────────────────────────

@router.delete("/rl/session/{session_id}",
               summary="Hapus sesi RL (reset Q-table dan step log)")
def rl_delete_session(session_id: str):
    if session_id in rl_registry._agents:
        del rl_registry._agents[session_id]
        return {"deleted": True, "session_id": session_id}
    return {"deleted": False, "session_id": session_id, "message": "Session not found"}


# ── Plots (all 7 as base64) ───────────────────────────────────────────────

@router.get("/rl/plots/{session_id}",
            tags=["RL Plots"],
            summary="Generate semua 7 plot RL sebagai base64 PNG")
def rl_plots(session_id: str):
    agent = rl_registry.get_agent(session_id)
    if len(agent.step_log) < 2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not enough steps to generate plots (need ≥ 2).",
        )
    plots = generate_plots(session_id)
    if not plots:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="matplotlib not installed — pip install matplotlib",
        )
    return {"session_id": session_id, "plots": plots}


@router.get("/rl/plots/{session_id}/{plot_name}",
            tags=["RL Plots"],
            summary="Download satu plot RL sebagai file PNG")
def rl_single_plot(session_id: str, plot_name: str):
    plots = generate_plots(session_id)
    if plot_name not in plots:
        available = list(plots.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{plot_name}' not found. Available: {available}",
        )
    return Response(
        content     = base64.b64decode(plots[plot_name]),
        media_type  = "image/png",
        headers     = {"Content-Disposition": f'attachment; filename="{session_id}_{plot_name}.png"'},
    )


# ── Legacy aliases ────────────────────────────────────────────────────────

@router.get("/metrics/{session_id}", include_in_schema=False)
def legacy_metrics(session_id: str, category: str = "Penggalang"):
    return rl_log(session_id, category)


@router.get("/pedagogy/{session_id}", include_in_schema=False)
def legacy_pedagogy(session_id: str, category: str = "Penggalang"):
    return rl_recommend(session_id, category)
