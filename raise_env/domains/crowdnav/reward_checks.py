"""CrowdNav reward validator (component returns + LLM typography cleanup)."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.runtime import SandboxedReward
from raise_core.sandbox.validator import RewardValidator

from domains.crowdnav.sandbox_fields import crowdnav_sandbox_config
from domains.crowdnav.state import default_smoke_states

# LLM typography that breaks the Python tokenizer outside strings
# (e.g. ``0.2\u202fm`` → "invalid decimal literal", ``\u2011`` → "invalid character").
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


class CrowdNavRewardValidator(RewardValidator):
    """
    Validator for ``compute_reward(state, memory)`` over ``CrowdNavRewardState``.

    Enables ``allow_components=True`` so a reward may return
    ``(float, dict[str, float])`` (a bare float is still accepted via shim).
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        smoke_states: Optional[Sequence[object]] = None,
    ) -> None:
        cfg = config if config is not None else crowdnav_sandbox_config()
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
        return super().validate_code(normalize_unicode_punctuation(str(code)))


def make_crowdnav_validator(
    *,
    config: Optional[SandboxConfig] = None,
    smoke_states: Optional[Sequence[object]] = None,
) -> CrowdNavRewardValidator:
    return CrowdNavRewardValidator(config=config, smoke_states=smoke_states)
