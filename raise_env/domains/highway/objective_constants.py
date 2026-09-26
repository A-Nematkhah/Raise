"""Single source of truth for highway fitness / soft-success speed scales.

Import these constants everywhere (metrics, prompts, proxy feedback). Do not
redefine or hand-pick parallel literals downstream.
"""

from __future__ import annotations

# Nominal traffic / cruise target (m/s). Soft-success and fitness speed terms
# are expressed relative to this value.
V_TARGET = 25.0

# Below this, survival terms are gated down (m/s).
V_MIN = 0.4 * V_TARGET  # 10.0

# Hard disqualify crawl / parked hack (m/s).
V_FLOOR = 0.15 * V_TARGET  # 3.75

# Sigmoid steepness for the recentered speed gate spanning [V_MIN, V_TARGET].
K_GATE = 6.0

# Lag penalty kicks in when mean_speed is below this (m/s).
LAG_SPEED_MPS = 18.0
