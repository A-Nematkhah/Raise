"""
Pluggable reward search types for CrowdNav++.

Termination / safety (done, episode_info) stay env-owned. Candidate reward
code may only read RewardState and return a scalar float.
"""

from crowd_nav.reward_search.state import (
    HumanObservable,
    LegacyReward,
    RewardFunction,
    RewardState,
    RobotRewardState,
)

__all__ = [
    "HumanObservable",
    "LegacyReward",
    "RewardFunction",
    "RewardState",
    "RobotRewardState",
]

