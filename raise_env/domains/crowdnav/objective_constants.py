"""Single source of truth for CrowdNav selection / training discount constants."""

from __future__ import annotations

# PPO discount (per policy step, dt = 0.25 s). Stage I Score1 ranks dataset
# trajectories by the same discounted return, so the two must not diverge.
PPO_GAMMA = 0.99

# Upstream evaluation discount: crowd_nav/configs/policy.config [rl] gamma.
# Explorer.run_k_episodes weighs step t's reward by
# gamma ** (t * time_step * v_pref) (discount per metre at preferred speed).
CROWDNAV_EVAL_GAMMA = 0.9
