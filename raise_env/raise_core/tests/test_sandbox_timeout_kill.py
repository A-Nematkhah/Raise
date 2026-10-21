"""Sandbox smoke timeout must kill the worker process (not leak a daemon thread)."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import replace

import pytest

from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.state import default_smoke_states
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import smoke_test_compute

# ``while`` is rejected by the AST policy, but smoke_test_compute compiles
# directly — use it so the child actually hangs past timeout_seconds.
# (A huge ``range(...)`` is rejected immediately by the guarded range helper.)
_HANG = """
def compute_reward(state, memory):
    s = 0.0
    while True:
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


class _SlowStartWorker:
    """Worker that takes ``delay`` s to start, then answers immediately."""

    def __init__(self, delay: float, result=("ok", None)):
        self._t0 = time.monotonic()
        self._delay = delay
        self._result = result

    def is_set(self) -> bool:
        return time.monotonic() - self._t0 >= self._delay

    def get(self, timeout: float):
        from queue import Empty

        if not self.is_set():
            time.sleep(timeout)
            raise Empty
        return self._result


class _AliveProc:
    exitcode = None

    def is_alive(self) -> bool:
        return True


def test_slow_worker_startup_is_not_charged_to_the_candidate():
    from raise_core.sandbox.runtime import _await_smoke_result

    w = _SlowStartWorker(delay=0.4)
    status, _ = _await_smoke_result(_AliveProc(), w, 0.2, startup_timeout_seconds=5.0, ready=w)
    assert status == "ok"


def test_worker_that_never_starts_reports_startup_timeout():
    from raise_core.sandbox.runtime import _await_smoke_result

    w = _SlowStartWorker(delay=60.0)
    with pytest.raises(RewardSandboxError, match="did not start"):
        _await_smoke_result(_AliveProc(), w, 0.2, startup_timeout_seconds=0.3, ready=w)


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
    rw = make_crowdnav_validator().validate_code(code)
    assert rw is not None
