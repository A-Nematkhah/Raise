"""Regression probes for shift / factorial / sequence-repeat / pow guards."""

from __future__ import annotations

import sys

import pytest

from domains.crowdnav.state import default_smoke_states
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.validator import RewardValidator


def _v() -> RewardValidator:
    return RewardValidator(
        smoke_states=default_smoke_states(),
        config=SandboxConfig(
            allowed_attributes=(
                "progress",
                "robot",
                "humans",
                "min_separation",
                "px",
                "py",
                "radius",
            ),
            timeout_seconds=5.0,
        ),
    )


def test_rejects_huge_left_shift():
    code = "def compute_reward(state, memory):\n    return float(1 << 100)\n"
    reward, reason = _v().try_validate(code)
    assert reward is None
    assert reason and "shift" in reason.lower()


def test_rejects_non_constant_left_shift():
    code = (
        "def compute_reward(state, memory):\n"
        "    n = int(state.progress) + 10\n"
        "    return float(1 << n)\n"
    )
    reward, reason = _v().try_validate(code)
    assert reward is None
    assert reason and "shift" in reason.lower()


def test_rejects_huge_factorial():
    code = "def compute_reward(state, memory):\n    return float(math.factorial(5000))\n"
    reward, reason = _v().try_validate(code)
    assert reward is None
    assert reason and "factorial" in reason.lower()


def test_rejects_huge_sequence_repeat():
    code = "def compute_reward(state, memory):\n    x = [0] * 100_000\n    return float(len(x))\n"
    reward, reason = _v().try_validate(code)
    assert reward is None
    assert reason and ("sequence" in reason.lower() or "repeat" in reason.lower())


def test_builtin_pow_follows_safe_pow_rules():
    code = "def compute_reward(state, memory):\n    return float(pow(9, 99))\n"
    reward, reason = _v().try_validate(code)
    assert reward is None
    assert reason and ("pow" in reason.lower() or "exponent" in reason.lower())


def test_normal_arithmetic_still_ok():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(state.progress) * 0.5 + (1 << 3) * 0.01\n"
    )
    reward, reason = _v().try_validate(code)
    assert reason is None
    assert reward is not None


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT only on POSIX")
def test_posix_rlimit_allows_normal_reward_with_torch_numpy():
    """Smoke child applies rlimit after compile; torch/numpy already imported."""
    import numpy  # noqa: F401
    import torch  # noqa: F401

    code = "def compute_reward(state, memory):\n    return float(state.progress)\n"
    reward, reason = _v().try_validate(code)
    assert reason is None and reward is not None


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT only on POSIX")
def test_posix_rlimit_rejects_memory_hog():
    # Allocate far beyond the relative budget in a loop of bytearrays.
    code = (
        "def compute_reward(state, memory):\n"
        "    chunks = []\n"
        "    for _ in range(64):\n"
        "        chunks.append(bytearray(64 * 1024 * 1024))\n"
        "    return float(len(chunks))\n"
    )
    cfg = SandboxConfig(
        allowed_attributes=("progress",),
        smoke_rlimit_as_bytes=64 * 1024 * 1024,  # 64 MiB budget on top of RSS
        timeout_seconds=10.0,
        smoke_max_call_seconds=5.0,
    )
    validator = RewardValidator(smoke_states=default_smoke_states()[:1], config=cfg)
    reward, reason = validator.try_validate(code)
    assert reward is None
    assert reason  # killed / sandbox error / worker exit
