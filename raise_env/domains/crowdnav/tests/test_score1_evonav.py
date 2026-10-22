"""EvoNav Score_1 unit tests (hand-built trajectories; no real dataset required)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Tuple

import numpy as np
import pytest

from domains.crowdnav.score1 import (
    Score1Config,
    Score1Evaluator,
    RuleRanker,
    frame_outcome_features,
    rule_score_at_frame,
    score_crowdnav_dataset,
)
from domains.crowdnav.stage1 import CrowdNavTrajectoryRecord
from domains.crowdnav.state import CrowdNavRewardState, HumanState, RobotState
from raise_core.rules import _rankdata, spearman_correlation


def _st(
    i: int,
    *,
    py: float,
    goal_y: float = 4.0,
    collision: bool = False,
    reached: bool = False,
    timeout: bool = False,
    min_sep: float = 1.0,
) -> CrowdNavRewardState:
    return CrowdNavRewardState(
        robot=RobotState(
            px=0.0,
            py=py,
            vx=0.0,
            vy=0.5,
            gx=0.0,
            gy=goal_y,
            theta=math.pi / 2,
            radius=0.3,
            v_pref=1.0,
            kinematics="holonomic",
        ),
        humans=(
            HumanState(px=2.0, py=py, vx=0.0, vy=0.0, radius=0.3, visible=True),
        ),
        action=(0.0, 0.5),
        collision=collision,
        reached_goal=reached,
        timeout=timeout,
        min_separation=min_sep,
        discomfort_dist=0.2,
        goal_distance=abs(goal_y - py),
        progress=0.5,
        time_step=0.25,
        global_time=0.25 * (i + 1),
        time_limit=25.0,
        min_separation_post=min_sep,
    )


def _traj(
    tid: str,
    label: str,
    frames: List[CrowdNavRewardState],
    *,
    scenario: str = "s0",
) -> CrowdNavTrajectoryRecord:
    return CrowdNavTrajectoryRecord(
        trajectory_id=tid,
        scenario_id=scenario,
        seed=0,
        states=tuple(frames),
        label=label,
        behavior="test",
    )


def _tiny_dataset() -> List[CrowdNavTrajectoryRecord]:
    """One scenario: short success, long success, early collision (carry-forward)."""
    short = [
        _st(0, py=-2.0),
        _st(1, py=0.0),
        _st(2, py=4.0, reached=True),
    ]
    long = [
        _st(0, py=-3.0),
        _st(1, py=-1.0),
        _st(2, py=1.0),
        _st(3, py=3.0),
        _st(4, py=4.0, reached=True),
    ]
    crash = [
        _st(0, py=-2.0, min_sep=0.5),
        _st(1, py=-1.0, min_sep=0.05, collision=True),
    ]
    return [
        _traj("short", "success", short),
        _traj("long", "success", long),
        _traj("crash", "collision", crash),
    ]


class _RuleMatchingReward:
    """Per-frame reward = Δ rule_score so cumsum ranks match rule ranks."""

    def __init__(self, trajs: List[CrowdNavTrajectoryRecord], sign: float = 1.0) -> None:
        self._tables = {}
        for traj in trajs:
            f_max = max(t.length for t in trajs if t.scenario_id == traj.scenario_id)
            rules = [rule_score_at_frame(traj, f) for f in range(f_max)]
            steps = [rules[0]] + [rules[i] - rules[i - 1] for i in range(1, len(rules))]
            # Truncate to actual length; evaluate uses carry-forward on cumsum.
            self._tables[id(traj)] = [sign * s for s in steps[: traj.length]]
            self._by_tid = getattr(self, "_by_tid", {})
            self._by_tid[traj.trajectory_id] = self._tables[id(traj)]
        self._idx = {}

    def reset(self) -> None:
        self._idx = {}

    def compute(self, state: CrowdNavRewardState) -> float:
        # Match by scanning — tests bind traj via wrapper below.
        raise RuntimeError("use BoundReward")


class _BoundReward:
    def __init__(self, steps_by_tid: dict, trajs: List[CrowdNavTrajectoryRecord]) -> None:
        self.steps_by_tid = steps_by_tid
        self.trajs = trajs
        self._active: List[float] = []
        self._i = 0

    def reset(self) -> None:
        self._active = []
        self._i = 0

    def bind(self, traj: CrowdNavTrajectoryRecord) -> None:
        self._active = list(self.steps_by_tid[traj.trajectory_id])
        self._i = 0

    def compute(self, state: CrowdNavRewardState) -> float:
        if self._i >= len(self._active):
            return 0.0
        v = self._active[self._i]
        self._i += 1
        return float(v)


def _ideal_fn(trajs: List[CrowdNavTrajectoryRecord], sign: float = 1.0) -> _BoundReward:
    steps = {}
    f_max = max(t.length for t in trajs)
    for traj in trajs:
        rules = [rule_score_at_frame(traj, f) for f in range(f_max)]
        deltas = [rules[0]] + [rules[i] - rules[i - 1] for i in range(1, traj.length)]
        steps[traj.trajectory_id] = [sign * d for d in deltas]
    return _BoundReward(steps, trajs)


class _SeqReward:
    """Returns precomputed step rewards; resets pointer on ``reset``."""

    def __init__(self, sequences: dict) -> None:
        self.sequences = sequences
        self._tid = None
        self._i = 0
        self._cur: List[float] = []

    def set_trajectory(self, traj: CrowdNavTrajectoryRecord) -> None:
        self._tid = traj.trajectory_id
        self._cur = list(self.sequences[traj.trajectory_id])
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def compute(self, state: CrowdNavRewardState) -> float:
        if self._i >= len(self._cur):
            return float(self._cur[-1]) if self._cur else 0.0
        v = self._cur[self._i]
        self._i += 1
        return float(v)


def _patch_step_rewards(monkeypatch, sequences: dict) -> None:
    """Make Score_1 use known per-trajectory step rewards."""
    import domains.crowdnav.score1 as mod

    def fake(reward_fn, traj):
        if hasattr(reward_fn, "set_trajectory"):
            reward_fn.set_trajectory(traj)
        if hasattr(reward_fn, "reset"):
            reward_fn.reset()
        seq = sequences[traj.trajectory_id]
        return np.asarray(seq, dtype=np.float64)

    monkeypatch.setattr(mod, "_step_rewards", fake)


def test_frame_outcome_adapter_and_carry_forward():
    trajs = _tiny_dataset()
    crash = trajs[2]
    cls, path, dist = frame_outcome_features(crash, 0)
    assert cls == "other"
    cls2, _, _ = frame_outcome_features(crash, 5)  # past end → carry-forward
    assert cls2 == "fail"
    assert rule_score_at_frame(trajs[0], 2) > rule_score_at_frame(trajs[2], 1)


def test_ideal_inverted_constant(monkeypatch):
    trajs = _tiny_dataset()
    f_max = max(t.length for t in trajs)
    ideal_seq = {}
    inv_seq = {}
    for traj in trajs:
        rules = [rule_score_at_frame(traj, f) for f in range(f_max)]
        deltas = [rules[0]]
        for i in range(1, traj.length):
            deltas.append(rules[i] - rules[i - 1])
        ideal_seq[traj.trajectory_id] = deltas
        inv_seq[traj.trajectory_id] = [-d for d in deltas]
    const_seq = {t.trajectory_id: [0.0] * t.length for t in trajs}

    _patch_step_rewards(monkeypatch, ideal_seq)
    ideal = score_crowdnav_dataset(object(), trajs)
    assert ideal.score == pytest.approx(1.0, abs=1e-9)
    assert ideal.n_nan_frames == 0

    _patch_step_rewards(monkeypatch, inv_seq)
    inv = score_crowdnav_dataset(object(), trajs)
    assert inv.score == pytest.approx(-1.0, abs=1e-9)

    _patch_step_rewards(monkeypatch, const_seq)
    const = score_crowdnav_dataset(
        object(), trajs, config=Score1Config(nan_policy="zero")
    )
    assert const.score == pytest.approx(0.0, abs=1e-12)
    assert const.n_nan_frames == const.n_frames > 0


def test_exception_reward_is_rejected():
    class Boom:
        def reset(self):
            return None

        def compute(self, state):
            raise RuntimeError("boom")

    result = score_crowdnav_dataset(Boom(), _tiny_dataset())
    assert result.rejected
    assert result.score == float("-inf")
    assert result.error and "boom" in result.error


def test_nan_reward_is_rejected():
    class Bad:
        def reset(self):
            return None

        def compute(self, state):
            return float("nan")

    result = score_crowdnav_dataset(Bad(), _tiny_dataset())
    assert result.rejected
    assert result.score == float("-inf")


def test_vectorized_matches_naive_reference(monkeypatch):
    trajs = _tiny_dataset()
    f_max = max(t.length for t in trajs)
    seq = {}
    for traj in trajs:
        rules = [rule_score_at_frame(traj, f) for f in range(f_max)]
        seq[traj.trajectory_id] = [
            rules[0] if i == 0 else rules[i] - rules[i - 1] for i in range(traj.length)
        ]
    _patch_step_rewards(monkeypatch, seq)
    fast = score_crowdnav_dataset(object(), trajs)

    # Naive double loop over scenarios/frames.
    groups = {"s0": trajs}
    rhos = []
    for sid, group in groups.items():
        fj = max(t.length for t in group)
        for f in range(fj):
            rule_scores = [rule_score_at_frame(t, f) for t in group]
            # rebuild cum from seq
            cums = []
            for t in group:
                steps = seq[t.trajectory_id]
                c = np.cumsum(steps)
                if f < len(c):
                    cums.append(c[f])
                else:
                    cums.append(c[-1])
            rho = spearman_correlation(_rankdata(rule_scores), _rankdata(cums))
            rhos.append(0.0 if not math.isfinite(rho) else float(rho))
    naive = float(np.mean(rhos))
    assert fast.score == pytest.approx(naive, abs=1e-9)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        return
    # Optional oracle on a frame with non-constant ranks.
    rs = [rule_score_at_frame(t, 1) for t in trajs]
    cs = []
    for t in trajs:
        steps = seq[t.trajectory_id]
        c = np.cumsum(steps)
        cs.append(float(c[min(1, len(c) - 1)]))
    ours = spearman_correlation(rs, cs)
    oracle = spearmanr(rs, cs).correlation
    if math.isfinite(ours) and math.isfinite(float(oracle)):
        assert ours == pytest.approx(float(oracle), abs=1e-9)


def test_rule_ranker_cache_shapes():
    trajs = _tiny_dataset()
    ranker = RuleRanker.build(trajs)
    assert ranker.rule_ranks.shape[0] == 1
    assert ranker.n_traj[0] == 3
    assert ranker.valid_mask[0].sum() == max(t.length for t in trajs)


def test_evaluate_population_sorted():
    trajs = _tiny_dataset()
    ev = Score1Evaluator(trajs)

    class Const:
        def __init__(self, v):
            self.v = v

        def reset(self):
            return None

        def compute(self, state):
            return self.v

    ranked = ev.evaluate_population([Const(1.0), Const(0.0)], n_workers=1)
    assert len(ranked) == 2
    assert ranked[0][1].score >= ranked[1][1].score
