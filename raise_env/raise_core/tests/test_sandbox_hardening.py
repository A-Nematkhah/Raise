"""Shared sandbox hardening: hostile parse inputs and child-first execution."""

from __future__ import annotations

import time

import pytest

from domains.crowdnav.sandbox_fields import crowdnav_sandbox_config
from domains.crowdnav.state import default_smoke_states
from raise_core.llm import split_reward_function_sources
from raise_core.source_normalize import (
    normalize_to_compute_reward_with_report,
    strip_redundant_scalar_subscripts,
)
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.validator import RewardValidator

_DEEP_EXPR = "1" + "+1" * 9000


def _validator(**kw) -> RewardValidator:
    """Domain allowlist by default; pass allow_unrestricted for legacy probes."""
    if "config" not in kw and not kw.get("allow_unrestricted"):
        kw.setdefault("config", crowdnav_sandbox_config())
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


def test_module_level_huge_range_is_rejected_fast():
    code = (
        "X = sum(range(10 ** 12))\n"
        "def compute_reward(state, memory):\n"
        "    return 0.0\n"
    )
    # Unrestricted top-level + allowlist; guarded range fails in the child.
    validator = _validator(
        config=SandboxConfig(
            timeout_seconds=3.0,
            allowed_attributes=crowdnav_sandbox_config().allowed_attributes,
        )
    )
    t0 = time.monotonic()
    reward, reason = validator.try_validate(code)
    assert reward is None
    assert reason and ("range" in reason.lower() or "timeout" in reason.lower())
    assert time.monotonic() - t0 < 30.0


def test_default_config_keeps_legacy_top_level_semantics():
    code = (
        "SCALE = max(1.0, 2.0)\n"
        "def compute_reward(state, memory):\n"
        "    return 0.1 * SCALE\n"
    )
    reward, reason = _validator(
        config=SandboxConfig(
            allowed_attributes=crowdnav_sandbox_config().allowed_attributes,
        )
    ).try_validate(code)
    assert reason is None
    assert reward is not None


def test_refuse_unrestricted_without_flag():
    with pytest.raises(ValueError, match="allow_unrestricted"):
        RewardValidator(
            config=SandboxConfig(allowed_attributes=None),
            smoke_states=default_smoke_states()[:1],
        )


def test_allow_unrestricted_opt_in():
    v = RewardValidator(
        config=SandboxConfig(allowed_attributes=None),
        smoke_states=default_smoke_states()[:1],
        allow_unrestricted=True,
    )
    reward, reason = v.try_validate(
        "def compute_reward(state, memory):\n    return 0.0\n"
    )
    assert reason is None
    assert reward is not None


def test_guarded_range_rejects_huge_length():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(sum(range(10 ** 12)) * 0)\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reward is None
    assert reason and "range" in reason.lower()


def test_guarded_pow_rejects_tower():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(9 ** 9 ** 9)\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reward is None
    assert reason and ("pow" in reason.lower() or "exponent" in reason.lower())


def test_bounded_memory_rejects_too_many_keys():
    code = (
        "def compute_reward(state, memory):\n"
        "    for i in range(80):\n"
        "        memory[i] = 0\n"
        "    return 0.0\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reward is None
    assert reason and "memory" in reason.lower()


def test_bounded_memory_rejects_huge_list_value():
    # Build the list without ``*`` or a single large ``range`` (those have
    # their own guards). Nested small ranges stay under max_range_length.
    code = (
        "def compute_reward(state, memory):\n"
        "    xs = []\n"
        "    for _ in range(101):\n"
        "        for _j in range(100):\n"
        "            xs.append(0)\n"
        "    memory['x'] = xs\n"
        "    return 0.0\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reward is None
    assert reason and "memory" in reason.lower()


def test_format_dunder_rejected_by_allowlist():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(len('{0.__class__}'.format(state)) * 0)\n"
    )
    reward, reason = _validator().try_validate(code)
    assert reward is None
    assert reason


def test_normalize_report_annotated_signature_and_helper():
    code = (
        "def helper(x):\n"
        "    return x\n"
        "def cal_reward(state: object):\n"
        "    return float(helper(state.progress))\n"
    )
    out, repairs = normalize_to_compute_reward_with_report(code)
    assert "def compute_reward(state, memory)" in out
    assert "def helper" in out
    assert "renamed_function" in repairs
    assert "added_memory_arg" in repairs
