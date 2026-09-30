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
    """
    Highway validator: speed-setpoint checks + EUREKA component returns.

    Enables ``allow_components=True`` so ``compute_reward`` may return
    ``(float, dict[str, float])`` (bare float still accepted via shim).
    CrowdNav validators never set this flag.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        smoke_states: Optional[Sequence[object]] = None,
    ) -> None:
        super().__init__(
            config=config,
            smoke_states=smoke_states,
            allow_components=True,
        )

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
