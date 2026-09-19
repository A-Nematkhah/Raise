"""Holdout split + Score1 hard-reject / train–holdout gap gates."""

from __future__ import annotations

from typing import List, Tuple

from crowd_nav.reward_search.dataset import TrajectoryRecord, split_stage1_dataset
from crowd_nav.reward_search.scoring import make_score1_fn, score1_for_dataset
from crowd_nav.reward_search.state import (
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
        behavior="test",
    )


def _scenario(sid: str) -> list:
    """One Success short, Success long, timeout, collision — scoreable."""
    return [
        _traj(sid, f"{sid}_ok_s", "success", [(1.0, 0.0, 0.25), (0.0, 0.0, 0.5)]),
        _traj(
            sid,
            f"{sid}_ok_l",
            "success",
            [(3.0, 0.0, 0.25), (2.0, 0.0, 0.5), (1.0, 0.0, 0.75), (0.0, 0.0, 1.0)],
        ),
        _traj(sid, f"{sid}_to", "timeout", [(2.0, 0.0, 0.25), (2.0, 0.0, 0.5)]),
        _traj(sid, f"{sid}_col", "collision", [(1.5, 0.0, 0.25), (1.5, 0.0, 0.5)]),
    ]


class _ConstantReward(RewardFunction):
    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        del state
        return 1.0


class _CategoryAlignedReward(RewardFunction):
    """Higher reward on success frames, lower on collision — roughly rule-aligned."""

    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        if state.collision:
            return -10.0
        if state.reaching_goal:
            return 10.0
        if state.timeout:
            return -1.0
        # Prefer closer to goal.
        dx = state.robot.px - state.robot.gx
        dy = state.robot.py - state.robot.gy
        return -((dx * dx + dy * dy) ** 0.5)


def test_split_stage1_dataset_by_scenario():
    ds = {f"s{i}": _scenario(f"s{i}") for i in range(10)}
    train, hold = split_stage1_dataset(ds, holdout_fraction=0.3, seed=425)
    assert len(train) + len(hold) == 10
    assert len(hold) >= 1
    assert len(train) >= 1
    assert set(train.keys()).isdisjoint(hold.keys())
    # Deterministic.
    train2, hold2 = split_stage1_dataset(ds, holdout_fraction=0.3, seed=425)
    assert set(train.keys()) == set(train2.keys())
    assert set(hold.keys()) == set(hold2.keys())


def test_split_single_scenario_no_holdout():
    ds = {"only": _scenario("only")}
    train, hold = split_stage1_dataset(ds, holdout_fraction=0.3, seed=1)
    assert hold == {}
    assert set(train.keys()) == {"only"}


def test_constant_reward_hard_rejected():
    ds = {"s0": _scenario("s0")}
    result = score1_for_dataset(ds, _ConstantReward())
    assert result.rejected is True
    assert result.score == float("-inf")
    assert result.degenerate_fraction >= 0.5


def test_make_score1_fn_reports_holdout_and_keeps_aligned():
    ds = {f"s{i}": _scenario(f"s{i}") for i in range(6)}
    fn = make_score1_fn(ds, holdout_fraction=0.3, split_seed=7, train_holdout_gap=1.5)
    out = fn(_CategoryAlignedReward(), candidate_id="c0")
    assert out.holdout_score is not None
    assert out.train_score is not None
    assert out.rejected is False
    assert float(out.score) > -1.0


def test_train_holdout_gap_rejects():
    """Gap gate fires when train − holdout exceeds the configured margin."""
    ds = {f"s{i}": _scenario(f"s{i}") for i in range(6)}
    fn = make_score1_fn(
        ds, holdout_fraction=0.3, split_seed=7, train_holdout_gap=-1.0
    )
    out = fn(_CategoryAlignedReward(), candidate_id="c0")
    assert out.train_score is not None
    assert out.holdout_score is not None
    assert out.rejected is True
    assert out.reject_reason == "train_holdout_gap"
    assert out.score == float("-inf")


def test_scenario_scores_and_failure_hints():
    ds = {f"s{i}": _scenario(f"s{i}") for i in range(4)}
    result = score1_for_dataset(ds, _CategoryAlignedReward(), reject_degenerate_at=1.01)
    assert result.scenario_scores is not None
    assert len(result.scenario_scores) >= 1
    from crowd_nav.reward_search.scoring import format_score1_failure_hints

    hints = format_score1_failure_hints(result)
    assert "worst_scenarios" in hints


def test_evolver_reflection_includes_diagnostics():
    from crowd_nav.reward_search.evolver import RewardCandidate, StageIEvolver
    from crowd_nav.reward_search.llm import ScriptedLLMClient

    client = ScriptedLLMClient(["unused"])
    evolver = StageIEvolver(client, score_fn=lambda *a, **k: 0.0)
    ranked = [
        RewardCandidate(
            candidate_id="best",
            code="def compute_reward(state, memory):\n    return 1.0\n",
            score=0.9,
            metadata={},
        ),
        RewardCandidate(
            candidate_id="weak",
            code="def compute_reward(state, memory):\n    return 0.0\n",
            score=0.1,
            metadata={
                "score1_failure_hints": "worst_scenarios[s0:-0.200]; raise collision"
            },
        ),
    ]
    note = evolver._build_reflection(ranked, generation=0)
    assert "Diagnostics" in note
    assert "worst_scenarios" in note


def test_mutation_weakness_includes_parent_diagnostics(monkeypatch):
    from crowd_nav.reward_search.evolver import RewardCandidate, StageIEvolver
    from crowd_nav.reward_search.llm import ScriptedLLMClient

    captured = {}

    def fake_mutation(code, reflection, **kwargs):
        captured["reflection"] = reflection
        return "PROMPT"

    monkeypatch.setattr(
        "crowd_nav.reward_search.evolver.format_d2_mutation", fake_mutation
    )
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(state.reaching_goal) - float(state.collision)\n"
    )
    client = ScriptedLLMClient(
        [f"```python\n{code}\n```"]
    )
    evolver = StageIEvolver(client, score_fn=lambda *a, **k: 0.0)
    parent = RewardCandidate(
        candidate_id="weak",
        code=code,
        score=0.2,
        valid=True,
        metadata={"score1_failure_hints": "worst_scenarios[sA:0.1]"},
    )
    # Bypass full validate path by stubbing _llm_raw / _completion_to_code chain
    evolver._try_mutate(parent, "global note", phase="mutation", attempt=1)
    assert "worst_scenarios[sA:0.1]" in captured.get("reflection", "")
