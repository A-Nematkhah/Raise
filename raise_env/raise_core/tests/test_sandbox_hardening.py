"""Shared sandbox hardening: hostile parse inputs and child-first execution."""

from __future__ import annotations

import time

from domains.crowdnav.state import default_smoke_states
from raise_core.llm import split_reward_function_sources, strip_redundant_scalar_subscripts
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.validator import RewardValidator

_DEEP_EXPR = "1" + "+1" * 9000


def _validator(**kw) -> RewardValidator:
    return RewardValidator(smoke_states=default_smoke_states(), **kw)


def test_strip_scalar_subscripts_survives_hostile_input():
    deep = f"def compute_reward(state, memory):\n    return {_DEEP_EXPR}\n"
    assert strip_redundant_scalar_subscripts(deep) == deep.strip()
    nul = "def compute_reward(state, memory):\n    return 1\x00\n"
    assert strip_redundant_scalar_subscripts(nul) == nul.strip()
    huge = "x = 1\n" * 20_000
    assert strip_redundant_scalar_subscripts(huge) == huge.strip()


def test_strip_scalar_subscripts_targets_reward_state_fields_only():
    code = (
        "def compute_reward(state, memory):\n"
        "    a = state.robot.px[0] + state.progress[0] + state.robot.v_pref[-1]\n"
        "    b = state.humans[0].px[0]\n"
        "    c = state.action[0]\n"
        "    return a + b + c\n"
    )
    out = strip_redundant_scalar_subscripts(code)
    assert "state.robot.px +" in out and "state.progress +" in out
    assert "state.robot.v_pref\n" in out
    assert "state.humans[0].px\n" in out
    assert "state.action[0]" in out


def test_split_sources_survives_hostile_input():
    completion = f"```python\ndef compute_reward(state, memory):\n    return {_DEEP_EXPR}\n```"
    pieces = split_reward_function_sources(completion)
    assert len(pieces) == 1


def test_validator_rejects_deep_nesting_instead_of_crashing():
    reward, reason = _validator().try_validate(
        f"def compute_reward(state, memory):\n    return {_DEEP_EXPR}\n"
    )
    assert reward is None
    assert reason


def test_validator_rejects_null_bytes():
    reward, reason = _validator().try_validate(
        "def compute_reward(state, memory):\n    return 1.0\x00\n"
    )
    assert reward is None
    assert reason


def test_module_level_hang_is_killed_in_child_not_run_in_parent():
    code = (
        "X = sum(range(10 ** 12))\n"
        "def compute_reward(state, memory):\n"
        "    return 0.0\n"
    )
    validator = _validator(config=SandboxConfig(timeout_seconds=3.0))
    t0 = time.monotonic()
    reward, reason = validator.try_validate(code)
    assert reward is None
    assert "timeout" in reason
    assert time.monotonic() - t0 < 30.0


def test_default_config_keeps_legacy_top_level_semantics():
    code = (
        "SCALE = max(1.0, 2.0)\n"
        "def compute_reward(state, memory):\n"
        "    return 0.1 * SCALE\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reason is None
    assert reward is not None
