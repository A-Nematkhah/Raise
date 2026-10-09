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


def test_smoke_rejects_huge_reward_scale():
    code = "def compute_reward(state, memory):\n    return 1e15 * float(state.speed)\n"
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is None
    assert "magnitude" in (err or "")


def test_large_but_bounded_terminal_penalty_is_accepted():
    code = (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n"
        "        return -1000.0, {'crash': -1000.0}\n"
        "    return float(state.progress), {'progress': float(state.progress)}\n"
    )
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is not None, err


def test_compute_rejects_out_of_bound_reward_at_train_time():
    code = "def compute_reward(state, memory):\n    return float(state.ego.x)\n"
    reward = _reward(code, max_abs_reward=50.0)
    states = fingerprint_smoke_states()
    assert any(abs(s.ego.x) > 50.0 for s in states)
    with pytest.raises(RewardSandboxError, match="magnitude"):
        for state in states:
            reward.compute(state)


def test_fast_reward_stays_within_default_budget():
    reward = _reward(_FAST)
    for state in default_smoke_states() * 50:
        reward.compute(state)
