"""Single source of truth for highway fitness / soft-success speed scales.

Import these constants everywhere (metrics, prompts, proxy feedback). Do not
redefine or hand-pick parallel literals downstream.
"""

from __future__ import annotations

# Nominal traffic / cruise reference (m/s) for *diagnostic* fitness /
# soft_success logging only — must NOT appear in LLM prompts or seed code.
V_TARGET = 25.0

# Below this, survival terms are gated down (m/s).
V_MIN = 0.4 * V_TARGET  # 10.0

# Hard disqualify crawl / parked hack (m/s).
V_FLOOR = 0.15 * V_TARGET  # 3.75

# Sigmoid steepness for the recentered speed gate spanning [V_MIN, V_TARGET].
K_GATE = 6.0

# Lag penalty kicks in when mean_speed is below this (m/s).
LAG_SPEED_MPS = 18.0

# Progress scale for the fitness progress term (~20 m/s * 40 s episode).
REF_PROGRESS_M = 800.0

# soft_success also requires this much forward progress (m); diagnostic only.
SOFT_PROGRESS_M = 400.0

# Survival = SR − CRASH_WEIGHT·CR − OFF_ROAD_WEIGHT·TR. A crash must cost more
# than a survived episode earns, else crash-prone slow policies score positive.
CRASH_WEIGHT = 2.0
OFF_ROAD_WEIGHT = 1.0

# PPO discount. Stage I Score1 ranks dataset trajectories by the same
# discounted return, so the two must not diverge.
PPO_GAMMA = 0.99
