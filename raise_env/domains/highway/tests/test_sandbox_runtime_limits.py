"""Runtime limits that the 12-state smoke test alone cannot enforce."""

from __future__ import annotations

import pytest

from domains.highway.reward_checks import make_highway_validator
from domains.highway.sandbox_fields import highway_sandbox_config
from domains.highway.state import default_smoke_states, fingerprint_smoke_states
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import SandboxedReward, compile_compute_reward

_FAST = "def compute_reward(state, memory):\n    return float(state.speed)\n"


def _reward(code: str, **cfg_overrides) -> SandboxedReward:
    cfg = highway_sandbox_config(**cfg_overrides)
    return SandboxedReward(
        compile_compute_reward(code, cfg), cfg, source_code=code, allow_components=True
    )


def test_smoke_rejects_state_dependent_slow_loop():
    code = (
        "def compute_reward(state, memory):\n"
        "    total = 0.0\n"
        "    for i in range(int(abs(state.ego.x)) * 20000):\n"
        "        total = total + 0.0\n"
        "    return total\n"
    )
    validator = make_highway_validator(smoke_states=fingerprint_smoke_states())
    reward_fn, err = validator.try_validate(code)
    assert reward_fn is None
    assert "too slow" in (err or "")


def test_compute_rejects_a_single_call_over_budget():
    reward = _reward(_FAST, max_call_seconds=0.0)
    with pytest.raises(RewardSandboxError, match="too slow"):
        reward.compute(default_smoke_states()[0])


def test_compute_rejects_slow_rolling_mean():
    reward = _reward(_FAST, max_mean_call_seconds=0.0, mean_call_window=3)
    state = default_smoke_states()[0]
    reward.compute(state)
    reward.compute(state)
    with pytest.raises(RewardSandboxError, match="mean compute"):
        reward.compute(state)


def test_fast_reward_stays_within_default_budget():
    reward = _reward(_FAST)
    for state in default_smoke_states() * 50:
        reward.compute(state)
