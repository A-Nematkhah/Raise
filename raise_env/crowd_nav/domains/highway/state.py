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
    ) -> HighwayRewardState:
        return HighwayRewardState(
            ego=EgoVehicle(
                x=x,
                y=0.0,
                vx=speed,
                vy=0.0,
                heading=0.0,
                speed=speed,
                lane_index=1.0,
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
    return (
        _state(progress=2.0, speed=25.0, others=nearby),
        _state(collision=True, progress=0.0, speed=5.0, global_time=5.0),
        _state(off_road=True, progress=0.5, speed=15.0, global_time=8.0),
        _state(timeout=True, progress=0.1, speed=10.0, global_time=40.0),
        _state(progress=0.0, speed=0.0, action=0, global_time=0.2),
    )
