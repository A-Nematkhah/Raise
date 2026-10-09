"""Sandbox smoke timeout must kill the worker process (not leak a daemon thread)."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import replace

import pytest

from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import smoke_test_compute

_HANG = """
def compute_reward(state, memory):
    s = 0.0
    for i in range(10**9):
        s = s + 1.0
    return float(s)
"""


def test_smoke_timeout_kills_the_worker():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RewardSandboxError, match="exceeded timeout"):
        smoke_test_compute(
            _HANG, default_smoke_states()[:1], SandboxConfig(timeout_seconds=0.5)
        )
    time.sleep(0.2)
    leaked = {p.pid for p in mp.active_children()} - before
    assert not leaked, f"timed-out workers still alive: {leaked}"


def test_smoke_requires_source_code():
    with pytest.raises(RewardSandboxError, match="source_code"):
        smoke_test_compute("", default_smoke_states()[:1], SandboxConfig())


def test_smoke_resets_memory_when_time_does_not_advance():
    code = (
        "def compute_reward(state, memory):\n"
        "    prev_t = memory.get('t')\n"
        "    memory['t'] = float(state.global_time)\n"
        "    if prev_t is None:\n"
        "        return 0.0\n"
        "    return 1.0 / (float(state.global_time) - prev_t)\n"
    )
    base = default_smoke_states()[0]
    states = tuple(replace(base, global_time=t) for t in (5.0, 5.2, 5.2, 1.0))
    smoke_test_compute(code, states, SandboxConfig())


def test_validator_still_accepts_fast_reward():
    code = """
def compute_reward(state, memory):
    return float(state.progress)
"""
    rw = make_highway_validator().validate_code(code)
    assert rw is not None
