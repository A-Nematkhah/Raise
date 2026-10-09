"""Highway sandbox hardening: top-level restriction, Score1 timeout, PPO watchdog."""

from __future__ import annotations

import threading
import time

import pytest

from domains.highway.reward_checks import make_highway_validator
from domains.highway.score1_worker import Score1TimeoutWorker
from domains.highway.stage1 import (
    save_highway_dataset,
    score_highway_dataset,
    smoke_trajectories,
)
from domains.highway.watchdog import StallWatchdog, beat
from raise_core.sandbox.runtime import SandboxedReward, compile_compute_reward
from domains.highway.sandbox_fields import highway_sandbox_config

_BODY = (
    "def compute_reward(state, memory):\n"
    "    if state.collision:\n"
    "        return -1.0\n"
    "    return 0.05 * state.ego.speed\n"
)


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        '"""Docstring."""\n',
        "TARGET = 25.0\n",
        "PAIR = (1, -1.5)\nNESTED = ((0, 1.0), (1, math.pi))\n",
        "BASE = 2.0\nSCALED = BASE * 3 + -1\n",
    ],
)
def test_highway_accepts_constant_top_level(prefix):
    reward, reason = make_highway_validator().try_validate(prefix + _BODY)
    assert reason is None, reason
    assert reward is not None


@pytest.mark.parametrize(
    "prefix",
    [
        "X = sum(range(5))\n",
        "F = lambda s: 1.0\n",
        "max(1, 2)\n",
        "TABLE = [0.0] * 4\nY = TABLE[0]\n",
        "W = {'speed': 1.0, 'crash': -2}\n",
        "HIST = []\n",
        "SEEN = {1, 2}\n",
        "PAIR = (1, [2])\n",
    ],
)
def test_highway_rejects_top_level_work(prefix):
    reward, reason = make_highway_validator().try_validate(prefix + _BODY)
    assert reward is None
    assert reason and "top-level" in reason


def test_mutable_global_cannot_leak_across_episodes():
    code = (
        "HIST = []\n"
        "def compute_reward(state, memory):\n"
        "    HIST.append(1.0)\n"
        "    return float(len(HIST))\n"
    )
    reward, reason = make_highway_validator().try_validate(code)
    assert reward is None
    assert "memory dict" in reason


def test_highway_validator_defaults_to_highway_smoke_states():
    validator = make_highway_validator()
    assert validator.config.restrict_top_level
    reward, reason = validator.try_validate(_BODY)
    assert reason is None, reason


def _reward(code: str) -> SandboxedReward:
    cfg = highway_sandbox_config()
    return SandboxedReward(
        compile_compute_reward(code, cfg), cfg, source_code=code, allow_components=True
    )


def test_score1_worker_matches_in_process_and_kills_hangs(tmp_path):
    trajs = smoke_trajectories()
    save_highway_dataset(str(tmp_path), trajs)
    worker = Score1TimeoutWorker(str(tmp_path), trajs, timeout_seconds=3.0)
    try:
        good = _reward(_BODY)
        expected = score_highway_dataset(good, trajs)
        got = worker(good, candidate_id="good")
        assert got.score == pytest.approx(expected.score)
        assert got.rejected == expected.rejected

        hang = _reward(
            "def compute_reward(state, memory):\n"
            "    total = 0.0\n"
            "    for i in range(10 ** 12):\n"
            "        total += 0.0\n"
            "    return total\n"
        )
        t0 = time.monotonic()
        res = worker(hang, candidate_id="hang")
        assert res.rejected and res.reject_reason == "score1_timeout"
        assert time.monotonic() - t0 < 30.0
        assert worker.restarts == 1

        again = worker(good, candidate_id="good2")
        assert again.score == pytest.approx(expected.score)
    finally:
        worker.close()


def test_score1_worker_runs_sourceless_callables_in_process(tmp_path):
    trajs = smoke_trajectories()

    class _Plain:
        def reset(self):
            pass

        def compute(self, state):
            return float(state.speed)

    worker = Score1TimeoutWorker(str(tmp_path / "missing"), trajs, timeout_seconds=1.0)
    res = worker(_Plain(), candidate_id="plain")
    assert res.score == pytest.approx(score_highway_dataset(_Plain(), trajs).score)
    assert worker._proc is None


def _spin_forever() -> None:
    while True:
        pass


def test_watchdog_interrupts_stalled_main_thread():
    with pytest.raises(RuntimeError, match="stalled"):
        with StallWatchdog(0.5, label="spin"):
            _spin_forever()


def test_watchdog_interrupts_stalled_worker_thread():
    errors = []

    def _work():
        try:
            with StallWatchdog(0.5, label="thread"):
                _spin_forever()
        except RuntimeError as exc:
            errors.append(exc)

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive()
    assert errors and "stalled" in str(errors[0])


def test_watchdog_quiet_while_beating_and_disabled_at_zero():
    with StallWatchdog(0.5, label="busy") as dog:
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            beat()
            time.sleep(0.01)
    assert not dog.fired

    with StallWatchdog(0.0, label="off") as off:
        time.sleep(0.2)
    assert not off.fired
