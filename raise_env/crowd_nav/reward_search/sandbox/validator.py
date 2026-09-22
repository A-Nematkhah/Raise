"""
Reward validation pipeline.

Ported from mobile_robot_env.rewards.sandbox.validator; CrowdNav operates on
source strings (no RewardCandidate pool required for Phase 0/1 unit tests).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from crowd_nav.reward_search.sandbox.ast_policy import (
    check_interface,
    check_structure,
    parse_reward_code,
)
from crowd_nav.reward_search.sandbox.config import SandboxConfig
from crowd_nav.reward_search.sandbox.errors import RewardSandboxError
from crowd_nav.reward_search.sandbox.runtime import (
    SandboxedReward,
    compile_compute_reward,
    default_smoke_states,
    smoke_test_compute,
)
from crowd_nav.reward_search.state import RewardState


class RewardValidator:
    """Validate LLM (or other) reward source and return a SandboxedReward if it passes."""

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        smoke_states: Optional[Sequence[RewardState]] = None,
    ) -> None:
        self.config = config if config is not None else SandboxConfig()
        self.smoke_states = (
            tuple(smoke_states) if smoke_states is not None else default_smoke_states()
        )

    def validate_code(self, code: str) -> SandboxedReward:
        """
        AST + interface + restricted exec + smoke test.

        Raises RewardSandboxError on any failure.
        """
        if not code or not str(code).strip():
            raise RewardSandboxError("no source code to validate")

        # Normalize aliases / missing memory arg before AST checks so prompt
        # examples and LLM outputs share one sandbox contract.
        from crowd_nav.reward_search.llm import normalize_to_compute_reward

        code = normalize_to_compute_reward(str(code))

        tree = parse_reward_code(code, self.config)
        check_structure(tree, self.config)
        check_interface(tree, self.config)
        compute_fn = compile_compute_reward(code, self.config)
        smoke_test_compute(compute_fn, self.smoke_states, self.config)
        return SandboxedReward(compute_fn, self.config, source_code=code)

    def try_validate(self, code: str) -> Tuple[Optional[SandboxedReward], Optional[str]]:
        """Return (reward, None) on success or (None, reason) on failure."""
        try:
            return self.validate_code(code), None
        except RewardSandboxError as exc:
            return None, str(exc)
