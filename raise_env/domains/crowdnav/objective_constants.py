"""Single source of truth for CrowdNav++ fitness and PPO discount constants."""

from __future__ import annotations

# Upstream fitness is the mean sum of native rewards per evaluation episode.
CROWDNAV_FITNESS_OBJECTIVE = "mean_native_episode_return"

# The upstream evaluator sums native rewards per episode without discounting;
# 1.0 is a neutral factor kept for tools that express this as a gamma.
CROWDNAV_EVAL_GAMMA = 1.0

# PPO discount per policy step is a training parameter, independent of the
# official mean upstream episode-return selection objective.
PPO_GAMMA = 0.99
