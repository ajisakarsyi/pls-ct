"""
simulate_rl — PLS RL Session Simulator  (v5)
=============================================
Package exposing the public simulation API.

Import examples
---------------
    from simulate_rl import run_simulation, make_plots, PROFILES
    from simulate_rl.profiles import StudentSimulator
    from simulate_rl.runners import run_single_line_simulation
    from simulate_rl.plots import make_single_line_plot

Run from CLI
------------
    python -m simulate_rl --sessions 3 --questions 20 --profile tgi_lean
    python -m simulate_rl --single-line --profile student_tgi --questions 40
    python -m simulate_rl --correlation
"""

# ── Public re-exports ─────────────────────────────────────────────────────────

from simulate_rl.profiles import (       # noqa: F401
    PROFILES,
    PROFILE_HOME_LT,
    RADAR_RING_ORDERS,
    RADAR_CLOCKWISE,
    StudentSimulator,
    lt_col, reward_col, bar,
    MPL_COLOURS, LT_COLOURS,
    G, R, Y, C, M, B, D, X,
)

from simulate_rl.terminal import (       # noqa: F401
    fmt_transition,
    print_qvalues,
    print_recommendation_change,
    print_session_summary,
    print_global_summary,
)

from simulate_rl.runners import (        # noqa: F401
    run_session,
    run_simulation,
    run_fixed_lt_simulation,
    run_story_simulation,
    run_per_lt_simulation,
    run_cross_lt_simulation,
    run_correlation_simulation,
    run_single_line_simulation,
)

from simulate_rl.plots import (          # noqa: F401
    HAS_PLOT,
    make_plots,
    make_qvalue_plots,
    make_lt_change_plot,
    make_phase_bar_plot,
    make_mpe_plots,
    make_per_lt_plots,
    make_cross_lt_plots,
    make_fixed_lt_plots,
    make_story_plots,
    make_correlation_plots,
    make_single_line_plot,
)

__version__ = "5.0.0"
__all__ = [
    # profiles
    "PROFILES", "PROFILE_HOME_LT", "StudentSimulator",
    # runners
    "run_session", "run_simulation", "run_fixed_lt_simulation",
    "run_story_simulation", "run_per_lt_simulation", "run_cross_lt_simulation",
    "run_correlation_simulation", "run_single_line_simulation",
    # plots
    "HAS_PLOT", "make_plots", "make_qvalue_plots", "make_lt_change_plot",
    "make_phase_bar_plot", "make_mpe_plots", "make_per_lt_plots",
    "make_cross_lt_plots", "make_fixed_lt_plots", "make_story_plots",
    "make_correlation_plots", "make_single_line_plot",
]
