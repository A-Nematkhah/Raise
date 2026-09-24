"""HighwayRewardState — sandbox-visible fields for highway-fast-v0 rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class NearbyVehicle:
    """Relative kinematics of another vehicle (ego-centric)."""

    x: float
    y: float
    vx: float
    vy: float
    heading: float


@dataclass(frozen=True)
class EgoVehicle:
    """Ego vehicle fields a candidate reward may read."""

    x: float
    y: float
    vx: float
    vy: float
    heading: float
    speed: float
    lane_index: float
    on_road: bool


@dataclass(frozen=True)
class HighwayRewardState:
    """
    Exact fields a highway reward may read for one env step.

    Termination flags are environment-owned; candidates must not invent
    done logic beyond reading these booleans.
    """

    ego: EgoVehicle
    others: Tuple[NearbyVehicle, ...]
    collision: bool
    off_road: bool
    timeout: bool
    action: Any
    time_step: float
    global_time: float
    time_limit: float
    # Dense shaping helpers (precomputed by the wrapper).
    progress: float  # forward displacement since previous frame (m)
    speed: float  # ego speed duplicate for convenience


def default_smoke_states() -> tuple[HighwayRewardState, ...]:
    """Deterministic snapshots for RewardValidator smoke tests."""

    def _state(
        *,
        x: float = 0.0,
        speed: float = 20.0,
        collision: bool = False,
        off_road: bool = False,
        timeout: bool = False,
        progress: float = 1.0,
        others: Tuple[NearbyVehicle, ...] = (),
        action: int = 1,
        global_time: float = 1.0,
        lane_index: float = 1.0,
    ) -> HighwayRewardState:
        return HighwayRewardState(
            ego=EgoVehicle(
                x=x,
                y=0.0,
                vx=speed,
                vy=0.0,
                heading=0.0,
                speed=speed,
                lane_index=lane_index,
                on_road=not off_road,
            ),
            others=others,
            collision=collision,
            off_road=off_road,
            timeout=timeout,
            action=action,
            time_step=0.2,
            global_time=global_time,
            time_limit=40.0,
            progress=progress,
            speed=speed,
        )

    nearby = (
        NearbyVehicle(x=25.0, y=0.0, vx=18.0, vy=0.0, heading=0.0),
        NearbyVehicle(x=10.0, y=4.0, vx=22.0, vy=0.0, heading=0.0),
    )
    close_call = (
        NearbyVehicle(x=8.0, y=0.0, vx=20.0, vy=0.0, heading=0.0),
    )
    return (
        # Cruise / clear traffic
        _state(progress=2.0, speed=25.0, others=nearby),
        # Terminal outcomes
        _state(collision=True, progress=0.0, speed=5.0, global_time=5.0),
        _state(off_road=True, progress=0.5, speed=15.0, global_time=8.0),
        _state(timeout=True, progress=0.1, speed=10.0, global_time=40.0),
        # Idle / stop
        _state(progress=0.0, speed=0.0, action=0, global_time=0.2),
        # Crawl survivor (Score1 / surrogate must distinguish from cruise)
        _state(progress=0.3, speed=6.0, others=nearby, global_time=12.0),
        # High-speed cruise
        _state(progress=3.5, speed=30.0, others=nearby, global_time=3.0, x=90.0),
        # Near-miss dense front
        _state(progress=1.2, speed=22.0, others=close_call, global_time=4.0),
        # Lane shift context
        _state(
            progress=1.5,
            speed=20.0,
            others=nearby,
            lane_index=0.0,
            action=2,
            global_time=6.0,
        ),
    )


def fingerprint_smoke_states(
    *,
    dataset_path: str | None = None,
    max_dataset_frames: int = 6,
) -> tuple[HighwayRewardState, ...]:
    """
    Behavior-fingerprint states: smoke suite + optional Stage I traj frames.

    Extra frames (crawl / cruise / crash) make surrogate fingerprints more
    discriminative without changing FEATURE_SCHEMA_VERSION.
    """
    base = list(default_smoke_states())
    if not dataset_path:
        try:
            from domains.highway.stage1 import DEFAULT_STAGE1_DATASET

            dataset_path = DEFAULT_STAGE1_DATASET
        except Exception:  # noqa: BLE001
            dataset_path = None
    if not dataset_path:
        return tuple(base)

    try:
        from domains.highway.stage1 import load_highway_dataset
    except Exception:  # noqa: BLE001
        return tuple(base)

    try:
        trajs = load_highway_dataset(dataset_path)
    except Exception:  # noqa: BLE001
        return tuple(base)

    # Prefer one mid-frame from crawl success, fast success, and a collision.
    picked: list[HighwayRewardState] = []
    buckets = {"crawl": None, "fast": None, "collision": None}
    for traj in trajs:
        if not traj.states:
            continue
        mid = traj.states[len(traj.states) // 2]
        beh = str(traj.behavior or "").lower()
        lab = str(traj.label or "").lower()
        spd = float(getattr(mid, "speed", 0.0) or 0.0)
        if buckets["crawl"] is None and (
            "crawl" in beh or (lab == "success" and spd < 12.0)
        ):
            buckets["crawl"] = mid
        elif buckets["fast"] is None and lab == "success" and spd >= 18.0:
            buckets["fast"] = mid
        elif buckets["collision"] is None and lab == "collision":
            buckets["collision"] = mid
        if all(buckets.values()):
            break
    for st in buckets.values():
        if st is not None:
            picked.append(st)
        if len(picked) >= max_dataset_frames:
            break
    return tuple(base + picked)
