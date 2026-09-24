"""Backward-compatible re-exports for frozen CrowdNav / Score1 import paths.

Canonical RAISE core: ``raise_core``.
Canonical CrowdNav domain: ``domains.crowdnav``.
"""

from __future__ import annotations

from raise_core import *  # noqa: F401,F403
from raise_core import __all__ as _core_all

# CrowdNav-owned surfaces historically imported from this package:
from domains.crowdnav.state import (  # noqa: F401
    HumanObservable,
    LegacyReward,
    RewardFunction,
    RewardState,
    RobotRewardState,
    build_reward_state,
)
from domains.crowdnav.dataset import (  # noqa: F401
    TrajectoryRecord,
    load_stage1_dataset,
)
from domains.crowdnav import regime as regime  # noqa: F401
from domains.crowdnav import reporting as reporting  # noqa: F401
from domains.crowdnav import dsrnn_baseline as dsrnn_baseline  # noqa: F401
from domains.crowdnav import prompts as prompts  # noqa: F401

__all__ = list(_core_all) + [
    "HumanObservable",
    "LegacyReward",
    "RewardFunction",
    "RewardState",
    "RobotRewardState",
    "build_reward_state",
    "TrajectoryRecord",
    "load_stage1_dataset",
    "regime",
    "reporting",
    "dsrnn_baseline",
    "prompts",
]
