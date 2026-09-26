"""Sandbox timeout must kill the worker process (not leak a daemon thread)."""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import (
    default_smoke_states,
    run_with_timeout,
    smoke_test_compute,
)
from raise_core.sandbox.validator import RewardValidator


def _sleep_forever() -> int:
    time.sleep(60.0)
    return 1


def test_run_with_timeout_kills_process():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RewardSandboxError, match="timeout"):
        run_with_timeout(_sleep_forever, 0.4)
    # Give the OS a beat to reap.
    time.sleep(0.2)
    after = {p.pid for p in mp.active_children()}
    leaked = after - before
    assert not leaked, f"timed-out workers still alive: {leaked}"


def test_smoke_test_with_source_kills_on_timeout():
    # Nested for-loop busy work is AST-legal (while is forbidden).
    code = """
def compute_reward(state, memory):
    s = 0.0
    for i in range(10**9):
        s = s + 1.0
    return float(s)
"""
    cfg = SandboxConfig(timeout_seconds=0.5)
    with pytest.raises(RewardSandboxError, match="timeout"):
        smoke_test_compute(
            (lambda s, m: 0.0),
            default_smoke_states()[:1],
            cfg,
            source_code=code,
        )


def test_validator_still_accepts_fast_reward():
    code = """
def compute_reward(state, memory):
    return float(state.progress) if hasattr(state, 'progress') else 0.0
"""
    # CrowdNav RewardState has no progress — use dmin path.
    code = """
def compute_reward(state, memory):
    return float(state.dmin)
"""
    rw = RewardValidator().validate_code(code)
    assert rw is not None
