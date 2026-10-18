"""Single source of truth for CrowdNav++ fitness and PPO discount constants."""

from __future__ import annotations

# Selection fitness is the mean of +1 success, -1 collision, and -0.5 timeout.
CROWDNAV_FITNESS_OBJECTIVE = "mean_outcome_score"

# Outcome scores are not discounted.
CROWDNAV_EVAL_GAMMA = 1.0

# PPO discount per policy step is a training parameter, independent of the
# official outcome-score selection objective.
PPO_GAMMA = 0.99
