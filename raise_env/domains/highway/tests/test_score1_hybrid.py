"""Highway Score1: discounted safety preference, driving quality, sanity rejects."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.stage1 import (
    DEFAULT_STAGE1_DATASET,
    HighwayTrajectoryRecord,
    load_highway_dataset,
    make_smoke_score_fn,
    score_highway_dataset,
    trajectory_role,
)
from domains.highway.state import EgoVehicle, HighwayRewardState, NearbyVehicle
from raise_core.domains import load_domain, make_validator_for_domain


def _frames(*, speed: float, progress: float, n: int = 8, crash_at: int | None = None):
    out = []
    x = 0.0
    for i in range(n):
        x += progress
        collision = crash_at is not None and i >= crash_at
        out.append(
            HighwayRewardState(
                ego=EgoVehicle(
                    x=x,
                    y=0.0,
                    vx=speed,
                    vy=0.0,
                    heading=0.0,
                    speed=speed,
                    lane_index=1.0,
                    on_road=True,
                ),
                others=(NearbyVehicle(x=30.0, y=0.0, vx=speed, vy=0.0, heading=0.0),),
                collision=collision,
                off_road=False,
                timeout=False,
                action=1,
                time_step=0.2,
                global_time=0.2 * (i + 1),
                time_limit=40.0,
                progress=0.0 if collision else progress,
                speed=speed,
            )
        )
    return out


def _toy_dataset():
    trajs = []
    # Fast successes
    for i in range(4):
        frames = _frames(speed=22.0, progress=4.5, n=10)
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"fast_{i}",
                scenario_id="fast",
                seed=i,
                states=tuple(frames),
                label="success",
                behavior="safe_fast",
                metadata={"mean_speed": 22.0, "mean_progress": 45.0, "overtakes": 0},
            )
        )
    # Safe cruise (~20–21) successes — distinct from crawl and from overtake
    for i in range(3):
        frames = _frames(speed=20.5, progress=3.8, n=10)
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"cruise_{i}",
                scenario_id="safe_cruise",
                seed=50 + i,
                states=tuple(frames),
                label="success",
                behavior="safe_cruise",
                metadata={
                    "mean_speed": 20.5,
                    "mean_progress": 38.0,
                    "overtakes": 0,
                    "safe_cruise_band": True,
                },
            )
        )
    # Genuine overtake successes
    for i in range(3):
        frames = _frames(speed=22.5, progress=5.0, n=10)
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"ovt_{i}",
                scenario_id="overtake",
                seed=80 + i,
                states=tuple(frames),
                label="success",
                behavior="overtake",
                metadata={
                    "mean_speed": 22.5,
                    "mean_progress": 50.0,
                    "overtakes": float(1 + i),
                    "genuine_overtake": True,
                },
            )
        )
    # Crawl successes
    for i in range(4):
        frames = _frames(speed=6.0, progress=1.0, n=10)
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"crawl_{i}",
                scenario_id="crawl",
                seed=100 + i,
                states=tuple(frames),
                label="success",
                behavior="crawl",
                metadata={"mean_speed": 6.0, "mean_progress": 10.0, "overtakes": 0},
            )
        )
    # Collisions
    for i in range(4):
        frames = _frames(speed=25.0, progress=3.0, n=6, crash_at=4)
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"crash_{i}",
                scenario_id="crash",
                seed=200 + i,
                states=tuple(frames),
                label="collision",
                behavior="aggressive",
                metadata={"mean_speed": 25.0, "mean_progress": 12.0, "overtakes": 0},
            )
        )
    # Timeouts
    for i in range(3):
        frames = _frames(speed=10.0, progress=2.0, n=8)
        # mark last as timeout-ish via label only
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"to_{i}",
                scenario_id="to",
                seed=300 + i,
                states=tuple(frames),
                label="timeout",
                behavior="synth_timeout",
                metadata={"mean_speed": 10.0, "mean_progress": 16.0, "overtakes": 0},
            )
        )
    return trajs


def test_hybrid_prefers_seed_over_crawl_reward():
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    seed_fn = validator.validate_code(D5_SEED_FUNCTION)

    crawl_code = """
def compute_reward(state, memory):
    if state.collision:
        return -1.0
    if state.off_road:
        return -1.0
    # Prefers low speed: inverse speed shaping
    return float(5.0 - 0.2 * state.speed + 0.01 * state.progress)
"""
    crawl_fn = validator.validate_code(crawl_code)
    trajs = _toy_dataset()
    s_seed = score_highway_dataset(seed_fn, trajs)
    s_crawl = score_highway_dataset(crawl_fn, trajs)
    assert float(s_seed.score) > float(s_crawl.score) + 0.05
    assert s_seed.scenario_scores is not None
    for key in ("safety", "quality", "step_spearman", "throughput", "overtake_alignment"):
        assert key in s_seed.scenario_scores


def test_overtake_aware_beats_passive_crawl_on_toy():
    """Acceptance: overtake-aware reward ranks above passive safe crawl."""
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    trajs = _toy_dataset()
    passive = validator.validate_code(
        """
def compute_reward(state, memory):
    if state.collision:
        return -50.0
    if state.off_road:
        return -20.0
    # Survival + modest speed around 20; no credit for passing.
    return float(1.0 - 0.2 * abs(state.speed - 20.0) + 0.01 * state.progress)
"""
    )
    aware = validator.validate_code(
        """
def compute_reward(state, memory):
    if state.collision:
        return -50.0
    if state.off_road:
        return -20.0
    r = 0.08 * state.progress + 0.02 * min(state.speed, 24.0)
    prev = memory.get("prev_others")
    cur = list(state.others or ())
    if prev is not None:
        for o in cur:
            for px, py in prev:
                if abs(float(o.y) - float(py)) < 2.0 and float(px) > 2.0 and float(o.x) <= 0.0:
                    r += 5.0
                    break
    stored = []
    for o in cur:
        stored.append((float(o.x), float(o.y)))
    memory["prev_others"] = stored
    for o in cur:
        # Closing gap on a lead vehicle (ego-relative vx < 0 while still ahead).
        if abs(float(o.y)) < 2.5 and float(o.x) > 0.0 and float(o.vx) < -0.5:
            r += 0.2
    return float(r)
"""
    )
    s_passive = score_highway_dataset(passive, trajs)
    s_aware = score_highway_dataset(aware, trajs)
    assert float(s_aware.score) > float(s_passive.score)
    assert float(s_aware.scenario_scores["overtake_alignment"]) > float(
        s_passive.scenario_scores["overtake_alignment"]
    )


def _fn(body: str):
    validator = make_validator_for_domain(load_domain("highway"))
    return validator.validate_code(f"def compute_reward(state, memory):\n    {body}\n")


def test_collision_loving_reward_is_rejected():
    crash_fn = _fn("return 50.0 if state.collision else 0.1 * state.progress")
    res = score_highway_dataset(crash_fn, _toy_dataset())
    assert res.rejected and res.reject_reason == "prefers_crash"
    assert res.scenario_scores["safety"] < 0.0


def test_safety_uses_discounted_returns():
    """A crash bonus below the undiscounted survival tail but above the discounted one."""
    trajs = []
    for i in range(3):
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"ok_{i}", scenario_id="ok", seed=i,
                states=tuple(_frames(speed=22.0, progress=4.5, n=400)), label="success",
            )
        )
        trajs.append(
            HighwayTrajectoryRecord(
                trajectory_id=f"crash_{i}", scenario_id="crash", seed=i,
                states=tuple(_frames(speed=25.0, progress=3.0, n=11, crash_at=10)), label="collision",
            )
        )
    # Undiscounted: 400 > 10 + 120. Discounted (γ=0.99): ≈98 < ≈9.6 + 120·0.90.
    res = score_highway_dataset(_fn("return 120.0 if state.collision else 1.0"), trajs)
    assert res.scenario_scores["safety"] == -1.0
    assert res.reject_reason == "prefers_crash"


def test_constant_and_zero_rewards_carry_no_quality():
    trajs = _toy_dataset()
    zero = score_highway_dataset(_fn("return 0.0"), trajs)
    assert zero.rejected and zero.reject_reason == "no_signal"
    alive = score_highway_dataset(_fn("return 1.0"), trajs)
    assert not alive.rejected
    assert alive.scenario_scores["quality"] == 0.0
    assert alive.degenerate_fraction == 1.0
    assert alive.score == pytest.approx(0.5 * alive.scenario_scores["safety"])
    seed = score_highway_dataset(make_validator_for_domain(load_domain("highway")).validate_code(D5_SEED_FUNCTION), trajs)
    assert seed.score > alive.score


def test_role_metadata_overrides_behavior_inference():
    trajs = _toy_dataset()
    assert {trajectory_role(t) for t in trajs if t.behavior == "safe_cruise"} == {"neutral"}
    assert {trajectory_role(t) for t in trajs if t.behavior == "crawl"} == {"lag"}
    assert {trajectory_role(t) for t in trajs if t.label != "success"} == {"failure"}
    relabeled = [
        replace(t, metadata={**t.metadata, "role": "positive"}) if t.behavior == "crawl" else t
        for t in trajs
    ]
    slow = _fn("return -1.0 if state.collision else 2.0 - state.speed / 25.0")
    assert score_highway_dataset(slow, trajs).scenario_scores["crawl_penalty"] > 0.0
    assert score_highway_dataset(slow, relabeled).scenario_scores["crawl_penalty"] == 0.0


def test_hybrid_spread_on_toy_rewards():
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    trajs = _toy_dataset()
    codes = [
        D5_SEED_FUNCTION,
        """
def compute_reward(state, memory):
    if state.collision: return -50.0
    return float(state.progress)
""",
        """
def compute_reward(state, memory):
    if state.collision: return 1.0
    return float(-state.speed)
""",
    ]
    scores = []
    for code in codes:
        fn = validator.validate_code(code)
        scores.append(float(score_highway_dataset(fn, trajs).score))
    assert max(scores) - min(scores) >= 0.15


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DEFAULT_STAGE1_DATASET, "trajectories.jsonl")),
    reason="Stage I dataset not collected (gitignored)",
)
def test_score1_properties_on_real_dataset():
    trajs = load_highway_dataset(DEFAULT_STAGE1_DATASET)

    def score(body: str):
        return score_highway_dataset(_fn(body), trajs)

    assert score("return 0.0").reject_reason == "no_signal"
    assert score("return 10.0 if state.collision else 0.1").reject_reason == "prefers_crash"
    assert score("return -1.0").reject_reason == "prefers_crash"
    alive = score("return 1.0")
    slow = score("return -100.0 if state.collision else 2.0 - state.speed / 25.0")
    good = score(
        "if state.collision or state.off_road:\n        return -50.0\n"
        "    return 0.05 * state.progress + 1.0 - abs(state.speed - 24.0) / 10.0"
    )
    seed = score_highway_dataset(
        make_validator_for_domain(load_domain("highway")).validate_code(D5_SEED_FUNCTION), trajs
    )
    assert not (alive.rejected or slow.rejected or good.rejected or seed.rejected)
    assert slow.score < 0.0 < alive.score < good.score
    assert seed.score > alive.score


def test_smoke_score_still_finite():
    score_fn = make_smoke_score_fn()
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    reward = validator.validate_code(D5_SEED_FUNCTION)
    result = score_fn(reward, candidate_id="seed")
    assert float(result.score) == float(result.score)
