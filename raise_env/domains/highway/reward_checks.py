"""Highway reward AST extras (speed-setpoint reachability)."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Sequence

from raise_core.llm import normalize_to_compute_reward
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.runtime import SandboxedReward
from raise_core.sandbox.validator import RewardValidator

from domains.highway.action_config import (
    check_target_speed_literals,
    warn_inline_speed_literals,
)
from domains.highway.state import default_smoke_states

logger = logging.getLogger(__name__)

# LLM typography that breaks the Python tokenizer outside strings
# (e.g. ``25\u202fm`` → "invalid decimal literal", ``\u2011`` → "invalid character").
_UNICODE_PUNCT_TO_ASCII = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u200b": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def normalize_unicode_punctuation(code: str) -> str:
    """Map look-alike Unicode dashes/spaces/quotes to ASCII."""
    return str(code).translate(_UNICODE_PUNCT_TO_ASCII)


class HighwayRewardValidator(RewardValidator):
    """
    Highway validator: speed-setpoint checks + EUREKA component returns.

    Enables ``allow_components=True`` so ``compute_reward`` may return
    ``(float, dict[str, float])`` (bare float still accepted via shim).
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        smoke_states: Optional[Sequence[object]] = None,
    ) -> None:
        cfg = config if config is not None else SandboxConfig()
        if not cfg.restrict_top_level:
            cfg = replace(cfg, restrict_top_level=True)
        super().__init__(
            config=cfg,
            smoke_states=(
                smoke_states if smoke_states is not None else default_smoke_states()
            ),
            allow_components=True,
        )

    def validate_code(self, code: str) -> SandboxedReward:
        code = normalize_unicode_punctuation(str(code))
        normalized = normalize_to_compute_reward(code)
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
