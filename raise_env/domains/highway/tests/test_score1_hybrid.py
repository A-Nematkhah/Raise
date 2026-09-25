"""Hybrid highway Score1 discriminates crawl vs cruise rewards."""

from __future__ import annotations

from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.stage1 import (
    HighwayTrajectoryRecord,
    make_smoke_score_fn,
    score_highway_dataset,
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
                metadata={"mean_speed": 22.0, "mean_progress": 45.0},
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
                metadata={"mean_speed": 6.0, "mean_progress": 10.0},
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
                metadata={"mean_speed": 25.0, "mean_progress": 12.0},
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
                metadata={"mean_speed": 10.0, "mean_progress": 16.0},
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
    assert "preference_auc" in s_seed.scenario_scores
    assert "collision_decoy_penalty" in s_seed.scenario_scores


def test_hybrid_penalizes_collision_loving_reward():
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    seed_fn = validator.validate_code(D5_SEED_FUNCTION)
    crash_code = """
def compute_reward(state, memory):
    if state.collision:
        return 50.0
    if state.off_road:
        return -1.0
    return float(0.1 * state.progress)
"""
    crash_fn = validator.validate_code(crash_code)
    trajs = _toy_dataset()
    # Tag collisions as decoy_crash so collision_decoy_penalty fires clearly.
    tagged = []
    for t in trajs:
        if t.label == "collision":
            tagged.append(
                HighwayTrajectoryRecord(
                    trajectory_id=t.trajectory_id,
                    scenario_id=t.scenario_id,
                    seed=t.seed,
                    states=t.states,
                    label=t.label,
                    behavior="decoy_crash",
                    metadata=t.metadata,
                )
            )
        else:
            tagged.append(t)
    s_seed = score_highway_dataset(seed_fn, tagged)
    s_crash = score_highway_dataset(crash_fn, tagged)
    assert float(s_seed.score) > float(s_crash.score) + 0.1
    assert float(s_crash.scenario_scores["collision_decoy_penalty"]) > 0.0


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


def test_smoke_score_still_finite():
    score_fn = make_smoke_score_fn()
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    reward = validator.validate_code(D5_SEED_FUNCTION)
    result = score_fn(reward, candidate_id="seed")
    assert float(result.score) == float(result.score)
