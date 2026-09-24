"""
Stage I Score1 tests: synthetic dataset (fast) + optional slow collect.

Run fast::
  pytest crowd_nav/reward_search/tests/test_stage1_real_dataset.py -q -m \"not slow\"

Slow collect (skipped by default unless explicitly selected)::
  pytest crowd_nav/reward_search/tests/test_stage1_real_dataset.py -m slow
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from domains.crowdnav.dataset import (
    TrajectoryRecord,
    load_stage1_dataset,
    reward_state_from_dict,
    reward_state_to_dict,
    save_stage1_dataset,
)
from raise_core.rules import (
    TrajectoryCategory,
    _rankdata,
    rule_preference_score,
    spearman_correlation,
)
from raise_core.scoring import make_score1_fn, score1_for_dataset
from domains.crowdnav.state import HumanObservable, RewardFunction, RewardState, RobotRewardState


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
    # frames: (px, py, global_time)
    states = tuple(
        _state(px=px, py=py, global_time=gt, reaching_goal=(label == "success" and i == len(frames) - 1),
               collision=(label == "collision" and i == len(frames) - 1),
               timeout=(label == "timeout" and i == len(frames) - 1))
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


class _AlignedReward(RewardFunction):
    """Higher cumulative reward for Success / closer-to-goal / shorter success."""

    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        if state.collision:
            return -20.0
        if state.reaching_goal:
            return 10.0
        dist = ((state.robot.px - state.robot.gx) ** 2 + (state.robot.py - state.robot.gy) ** 2) ** 0.5
        return float(-dist)


class _AntiAlignedReward(RewardFunction):
    """Inverts preferences relative to the analytical rules."""

    def __init__(self) -> None:
        self._inner = _AlignedReward()

    def reset(self) -> None:
        self._inner.reset()

    def compute(self, state: RewardState) -> float:
        return -self._inner.compute(state)


def _three_scenario_dataset():
    """
    Three scenarios, each with ≥2 trajs spanning Success/Other/Fail.
    """
    ds = {}
    for s in range(3):
        sid = f"scenario_{s:03d}"
        # Success short (closer, fewer steps)
        t_ok_short = _traj(
            sid, f"{sid}_ok_s", "success",
            [(4.0, 0.0, 0.25), (2.0, 0.0, 0.5), (0.2, 0.0, 0.75)],
        )
        # Success long
        t_ok_long = _traj(
            sid, f"{sid}_ok_l", "success",
            [(5.0, 0.0, 0.25), (4.0, 0.0, 1.0), (3.0, 0.0, 2.0), (1.0, 0.0, 3.0), (0.2, 0.0, 4.0)],
        )
        # Timeout close to goal
        t_to_near = _traj(
            sid, f"{sid}_to_n", "timeout",
            [(1.5, 0.0, 0.25), (1.0, 0.0, 1.0), (0.8, 0.0, 50.0)],
        )
        # Timeout far
        t_to_far = _traj(
            sid, f"{sid}_to_f", "timeout",
            [(6.0, 0.0, 0.25), (5.5, 0.0, 1.0), (5.0, 0.0, 50.0)],
        )
        # Collision near goal
        t_col_near = _traj(
            sid, f"{sid}_col_n", "collision",
            [(1.2, 0.0, 0.25), (0.9, 0.0, 0.5)],
        )
        # Collision far
        t_col_far = _traj(
            sid, f"{sid}_col_f", "collision",
            [(7.0, 0.0, 0.25), (6.5, 0.0, 0.5)],
        )
        ds[sid] = [t_ok_short, t_ok_long, t_to_near, t_to_far, t_col_near, t_col_far]
    return ds


def test_rankdata_and_spearman_basic():
    ranks = _rankdata([3.0, 1.0, 2.0])
    assert list(ranks) == [3.0, 1.0, 2.0]
    # Perfect agreement
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_rule_preference_category_order():
    s_short = rule_preference_score(TrajectoryCategory.SUCCESS, nav_length=1.0, dist_goal=9.0)
    s_long = rule_preference_score(TrajectoryCategory.SUCCESS, nav_length=5.0, dist_goal=0.1)
    o_near = rule_preference_score(TrajectoryCategory.OTHER, nav_length=1.0, dist_goal=1.0)
    f_far = rule_preference_score(TrajectoryCategory.FAIL, nav_length=1.0, dist_goal=9.0)
    assert s_short > s_long > o_near > f_far


def test_score1_aligned_beats_antialigned():
    ds = _three_scenario_dataset()
    aligned = score1_for_dataset(ds, _AlignedReward())
    anti = score1_for_dataset(ds, _AntiAlignedReward())
    assert -1.0 <= aligned.score <= 1.0
    assert -1.0 <= anti.score <= 1.0
    assert aligned.score > anti.score
    assert 0.0 <= aligned.degenerate_fraction <= 1.0
    assert aligned.n_pairs >= 1


def test_score1_raises_when_nothing_scoreable():
    sid = "scenario_empty"
    only_one = [
        _traj(sid, "only", "success", [(1.0, 0.0, 0.25), (0.2, 0.0, 0.5)]),
    ]
    with pytest.raises(ValueError, match="no scoreable"):
        score1_for_dataset({sid: only_one}, _AlignedReward())


def test_make_score1_fn_and_roundtrip(tmp_path):
    ds = _three_scenario_dataset()
    out = tmp_path / "stage1"
    save_stage1_dataset(ds, str(out), fmt="jsonl")
    loaded = load_stage1_dataset(str(out))
    assert set(loaded) == set(ds)
    fn = make_score1_fn(loaded)
    result = fn(_AlignedReward(), candidate_id="c0")
    assert -1.0 <= float(result) <= 1.0
    assert "degenerate_fraction" in result.as_dict()
    # State JSON round-trip
    s0 = ds["scenario_000"][0].states[0]
    assert reward_state_from_dict(reward_state_to_dict(s0)).robot.px == s0.robot.px


def test_npz_save_creates_dir_when_missing(tmp_path):
    ds = _three_scenario_dataset()
    out = tmp_path / "missing_dir"
    assert not out.exists()
    save_stage1_dataset(ds, str(out), fmt="npz")
    assert (out / "stage1_dataset.npz").is_file()
    loaded = load_stage1_dataset(str(out))
    assert len(loaded) == 3


@pytest.mark.slow
def test_collect_stage1_dataset_smoke():
    """Real simulator collect with M=3, N_traj=4 — opt-in via -m slow."""
    import importlib.util
    from pathlib import Path

    # .../crowd_nav/reward_search/tests/this_file.py → raise_env/
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "collect_stage1_dataset.py"
    assert script.is_file(), script

    spec = importlib.util.spec_from_file_location("collect_stage1_dataset", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Write under the project on D: — system temp is often on a full C: drive.
    out = root / "data" / "_pytest_stage1_collect"
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    try:
        mod.collect_dataset(
            n_scenarios=3,
            n_traj=4,
            base_seed=425,
            out_dir=str(out),
            noise_stds=(0.25, 0.55),
            human_num=5,
            fmt="npz",
        )
        loaded = load_stage1_dataset(str(out))
        assert len(loaded) == 3
        for trajs in loaded.values():
            assert len(trajs) >= 2
            for t in trajs:
                assert t.length >= 1
                assert t.label in ("success", "collision", "timeout")
        score = score1_for_dataset(loaded, _AlignedReward())
        assert -1.0 <= score.score <= 1.0
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            pytest.skip(f"disk full during collect: {exc}")
        raise
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
