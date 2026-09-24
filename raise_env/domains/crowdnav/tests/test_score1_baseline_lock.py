"""
Regression tests for Score1 padding / nav-length / degeneracy fixes
(baseline-pre-amfrs lock).
"""

from __future__ import annotations

from typing import List, Tuple

from domains.crowdnav.dataset import TrajectoryRecord
from raise_core.rules import TrajectoryCategory, rule_preference_score
from raise_core.scoring import (
    Score1Result,
    _cumulative_reward,
    _scenario_frame_correlations,
    score1_for_dataset,
)
from domains.crowdnav.state import (
    HumanObservable,
    RewardFunction,
    RewardState,
    RobotRewardState,
)


def _robot(*, px: float, py: float, gx: float = 0.0, gy: float = 0.0) -> RobotRewardState:
    return RobotRewardState(
        px=px, py=py, vx=0.0, vy=0.0, radius=0.3, gx=gx, gy=gy, v_pref=1.0
    )


def _state(
    *,
    px: float,
    py: float,
    gx: float = 0.0,
    gy: float = 0.0,
    global_time: float = 1.0,
    collision: bool = False,
    reaching_goal: bool = False,
    timeout: bool = False,
) -> RewardState:
    return RewardState(
        robot=_robot(px=px, py=py, gx=gx, gy=gy),
        humans=(HumanObservable(10.0, 10.0, 0.0, 0.0, 0.3),),
        dmin=5.0,
        discomfort_dist=0.25,
        collision=collision,
        reaching_goal=reaching_goal,
        timeout=timeout,
        action=None,
        time_step=0.25,
        global_time=global_time,
        time_limit=50.0,
    )


def _traj(
    scenario_id: str,
    tid: str,
    label: str,
    frames: List[Tuple[float, float, float]],
) -> TrajectoryRecord:
    states = tuple(
        _state(
            px=px,
            py=py,
            global_time=gt,
            reaching_goal=(label == "success" and i == len(frames) - 1),
            collision=(label == "collision" and i == len(frames) - 1),
            timeout=(label == "timeout" and i == len(frames) - 1),
        )
        for i, (px, py, gt) in enumerate(frames)
    )
    return TrajectoryRecord(
        trajectory_id=tid,
        scenario_id=scenario_id,
        seed=0,
        states=states,
        label=label,
        behavior="synthetic",
    )


class _TerminalSpikeReward(RewardFunction):
    """Large nonzero reward on the terminal state only — catches pad accumulation."""

    def __init__(self, spike: float = 100.0) -> None:
        self.spike = float(spike)
        self.compute_calls = 0

    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        self.compute_calls += 1
        if state.reaching_goal or state.collision or state.timeout:
            return self.spike
        return 0.0


class _ElapsedStepsReward(RewardFunction):
    """
    Reward keyed only on how many times compute() has been called this episode.

    Used to detect final-length leak: if rule Success ranks used final nav_length
    for every frame, frame-wise rule ranks among Success trajs would be constant
    while this reward's cumulative still varies — Spearman would still work, but
    rule_preference_score itself would not vary across frames for Success.

    We assert rule scores vary with f via rule_preference_score directly, and
    that cumulative freezes after real length so pad frames do not keep
    incrementing this counter-based reward.
    """

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def compute(self, state: RewardState) -> float:
        del state
        self._step += 1
        return float(self._step)


class _ConstantReward(RewardFunction):
    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        del state
        return 1.0


def test_cumulative_reward_freezes_on_padding():
    """Bug #1: padded frames must not call compute() or grow the cumulative."""
    short = _traj(
        "s0",
        "short",
        "success",
        [(2.0, 0.0, 0.25), (0.1, 0.0, 0.5)],  # length 2, last is goal
    )
    long_len = 5  # pad short traj out to frame index 4
    reward = _TerminalSpikeReward(spike=100.0)
    totals = _cumulative_reward(reward, short, max_frame=long_len - 1)

    assert len(totals) == long_len
    # Real frames: step0 → 0, step1 (goal) → 100; pads repeat 100.
    assert totals[0] == 0.0
    assert totals[1] == 100.0
    assert totals[2] == 100.0
    assert totals[3] == 100.0
    assert totals[4] == 100.0
    # Only real frames invoke compute (2 calls), not padded ones.
    assert reward.compute_calls == 2


def test_success_rule_nav_length_capped_at_traj_length():
    """Figure 3: after short Success ends, nav_length stays short (no f+1 growth on pads)."""
    short = _traj(
        "s0",
        "short",
        "success",
        [(3.0, 0.0, 0.25), (1.5, 0.0, 0.5), (0.1, 0.0, 0.75)],
    )
    long = _traj(
        "s0",
        "long",
        "success",
        [
            (4.0, 0.0, 0.25),
            (3.0, 0.0, 0.5),
            (2.0, 0.0, 0.75),
            (1.0, 0.0, 1.0),
            (0.1, 0.0, 1.25),
        ],
    )
    assert short.length == 3
    assert long.length == 5

    # At late frame index, short is capped at length 3; long still grows with f+1.
    f_late = 4
    short_nav = float(min(f_late + 1, short.length))
    long_nav = float(min(f_late + 1, long.length))
    assert short_nav == 3.0
    assert long_nav == 5.0
    short_rule = rule_preference_score(
        TrajectoryCategory.SUCCESS, nav_length=short_nav, dist_goal=0.0
    )
    long_rule = rule_preference_score(
        TrajectoryCategory.SUCCESS, nav_length=long_nav, dist_goal=0.0
    )
    assert short_rule > long_rule

    # Early frames: both still running → same steps-so-far (no final-length leak).
    f_early = 1
    assert min(f_early + 1, short.length) == min(f_early + 1, long.length) == 2

    reward = _ElapsedStepsReward()
    corrs, n_pairs, n_deg = _scenario_frame_correlations([short, long], reward)
    assert n_pairs >= 1
    r2 = _ElapsedStepsReward()
    totals = _cumulative_reward(r2, short, max_frame=4)
    assert r2._step == short.length
    assert totals[short.length - 1] == totals[-1]


def test_score1_prefers_reward_that_ranks_short_success_above_long():
    """
    Two rewards agree on Success≻timeout≻collision but disagree on short vs long Success.
    With min(f+1, len), Score1 must separate them (plain f+1 cannot).
    """

    class _ShortSuccessPrefer(RewardFunction):
        """Higher cumulative on short Success; lower on long Success."""

        def reset(self) -> None:
            return None

        def compute(self, state: RewardState) -> float:
            if state.collision:
                return -100.0
            if state.timeout:
                return -10.0
            if state.reaching_goal:
                # Prefer finishing early: bonus inverse to time.
                return 50.0 - float(state.global_time)
            return 0.1

    class _LongSuccessPrefer(RewardFunction):
        def reset(self) -> None:
            return None

        def compute(self, state: RewardState) -> float:
            if state.collision:
                return -100.0
            if state.timeout:
                return -10.0
            if state.reaching_goal:
                return 10.0 + float(state.global_time)
            return 0.1

    sid = "s_fig3"
    short = _traj(
        sid,
        "short",
        "success",
        [(2.0, 0.0, 0.25), (0.1, 0.0, 0.5)],
    )
    long = _traj(
        sid,
        "long",
        "success",
        [
            (5.0, 0.0, 0.25),
            (4.0, 0.0, 0.5),
            (3.0, 0.0, 0.75),
            (2.0, 0.0, 1.0),
            (0.1, 0.0, 1.25),
        ],
    )
    timeout = _traj(
        sid,
        "to",
        "timeout",
        [(3.0, 0.0, 0.25), (3.0, 0.0, 0.5), (3.0, 0.0, 0.75)],
    )
    collide = _traj(
        sid,
        "col",
        "collision",
        [(1.0, 0.0, 0.25), (0.5, 0.0, 0.5)],
    )
    ds = {sid: [short, long, timeout, collide]}
    s_short = float(score1_for_dataset(ds, _ShortSuccessPrefer()))
    s_long = float(score1_for_dataset(ds, _LongSuccessPrefer()))
    assert s_short > s_long


def test_degenerate_fraction_flagged_for_constant_reward():
    """Constant reward → high degenerate_fraction and hard-reject (-inf)."""
    sid = "s_deg"
    # Same-category Success trajs so rule ranks can still vary via nav-so-far,
    # but constant reward → Spearman NaN on reward side.
    t_a = _traj(sid, "a", "success", [(2.0, 0.0, 0.25), (0.1, 0.0, 0.5)])
    t_b = _traj(
        sid,
        "b",
        "success",
        [(3.0, 0.0, 0.25), (2.0, 0.0, 0.5), (1.0, 0.0, 0.75), (0.1, 0.0, 1.0)],
    )
    ds = {sid: [t_a, t_b]}
    result = score1_for_dataset(ds, _ConstantReward())
    assert isinstance(result, Score1Result)
    assert result.n_pairs >= 1
    assert result.degenerate_fraction >= 0.5
    assert result.rejected is True
    assert result.score == float("-inf")
    corrs, n_pairs, n_deg = _scenario_frame_correlations([t_a, t_b], _ConstantReward())
    assert corrs == []
    assert n_pairs >= 1
    assert n_deg == n_pairs
