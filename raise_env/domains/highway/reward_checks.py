"""Highway reward AST extras (speed-setpoint reachability)."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.runtime import SandboxedReward
from raise_core.sandbox.validator import RewardValidator

from domains.highway.action_config import (
    check_target_speed_literals,
    warn_inline_speed_literals,
)

logger = logging.getLogger(__name__)


class HighwayRewardValidator(RewardValidator):
    """RewardValidator + reject unreachable target_speed literals for action_mode."""

    def validate_code(self, code: str) -> SandboxedReward:
        from raise_core.llm import normalize_to_compute_reward

        normalized = normalize_to_compute_reward(str(code))
        check_target_speed_literals(normalized)
        for warning in warn_inline_speed_literals(normalized):
            logger.warning("highway reward speed check: %s", warning)
        return super().validate_code(code)


def make_highway_validator(
    *,
    config: Optional[SandboxConfig] = None,
    smoke_states: Optional[Sequence[object]] = None,
) -> HighwayRewardValidator:
    return HighwayRewardValidator(config=config, smoke_states=smoke_states)
