"""CrowdNavRewardState — the sandbox-visible view of one CrowdNav step."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from domains.crowdnav.settings import CrowdNavSettings, get_settings

# Upstream crowd_nav/configs/env.config defaults (smoke states only; the env
# reads the live values from its config).
SMOKE_TIME_STEP = 0.25
SMOKE_TIME_LIMIT = 25.0
SMOKE_DISCOMFORT_DIST = 0.2
SMOKE_RADIUS = 0.3
SMOKE_V_PREF = 1.0
SMOKE_CIRCLE_RADIUS = 4.0


@dataclass(frozen=True)
class RobotState:
    """Robot full state after the step (upstream ``FullState``, world frame)."""

    px: float
    py: float
    vx: float
    vy: float
    gx: float
    gy: float
    theta: float
    radius: float
    v_pref: float


@dataclass(frozen=True)
class HumanState:
    """One human's observable state after the step (upstream ``ObservableState``)."""

    px: float
    py: float
    vx: float
    vy: float
    radius: float


@dataclass(frozen=True)
class CrowdNavRewardState:
    """
    Exact fields a CrowdNav reward may read for one env step.

    Outcome flags mirror the upstream ``info`` of ``CrowdSim.step`` and are
    mutually exclusive in its priority order (timeout > collision > goal).
    ``min_separation`` is the upstream ``dmin``: the smallest boundary-to-
    boundary distance between robot and any human along this step's relative
    motion (negative on collision).
    """

    robot: RobotState
    humans: Tuple[HumanState, ...]
    action: Tuple[float, float]
    collision: bool
    reached_goal: bool
    timeout: bool
    min_separation: float
    discomfort_dist: float
    goal_distance: float
    progress: float
    time_step: float
    global_time: float
    time_limit: float


def holonomic_to_unicycle(
    vx: float, vy: float, theta_prev: float
) -> Tuple[Tuple[float, float], float]:
    """``(v, r)`` and the new heading for a holonomic move, as seen by a unicycle reward."""
    speed = math.hypot(vx, vy)
    if speed < 1e-9:
        return (0.0, 0.0), float(theta_prev)
    heading = math.atan2(vy, vx)
    r = (heading - theta_prev + math.pi) % (2.0 * math.pi) - math.pi
    return (float(speed), float(r)), float(heading % (2.0 * math.pi))


def with_kinematics(
    states: Tuple[CrowdNavRewardState, ...], settings: CrowdNavSettings
) -> Tuple[CrowdNavRewardState, ...]:
    """
    Re-express holonomic ``(vx, vy)`` actions of a recorded trajectory for ``settings``.

    The Stage I dataset is collected with holonomic robots; a unicycle reward
    reads ``state.action`` as ``(v, r)`` and ``state.robot.theta`` as the
    heading, so both are derived from the recorded motion.
    """
    if settings.kinematics == "holonomic":
        return tuple(states)
    out = []
    theta = math.pi / 2.0
    for st in states:
        action, theta = holonomic_to_unicycle(st.action[0], st.action[1], theta)
        out.append(replace(st, action=action, robot=replace(st.robot, theta=theta)))
    return tuple(out)


def smoke_action(vx: float, vy: float) -> Tuple[float, float]:
    """``state.action`` for a smoke state under the active kinematics."""
    if get_settings().kinematics == "holonomic":
        return (float(vx), float(vy))
    action, _ = holonomic_to_unicycle(vx, vy, math.pi / 2.0)
    return action


def default_smoke_states() -> Tuple[CrowdNavRewardState, ...]:
    """Deterministic snapshots for RewardValidator smoke tests."""
    goal = (0.0, SMOKE_CIRCLE_RADIUS)

    def _state(
        *,
        py: float = -2.0,
        px: float = 0.0,
        vx: float = 0.0,
        vy: float = 1.0,
        humans: Tuple[HumanState, ...] = (),
        collision: bool = False,
        reached_goal: bool = False,
        timeout: bool = False,
        min_separation: float = 1.5,
        progress: float = 0.25,
        global_time: float = 2.0,
    ) -> CrowdNavRewardState:
        dist = math.hypot(goal[0] - px, goal[1] - py)
        return CrowdNavRewardState(
            robot=RobotState(
                px=px,
                py=py,
                vx=vx,
                vy=vy,
                gx=goal[0],
                gy=goal[1],
                theta=math.pi / 2.0,
                radius=SMOKE_RADIUS,
                v_pref=SMOKE_V_PREF,
            ),
            humans=humans,
            action=smoke_action(vx, vy),
            collision=collision,
            reached_goal=reached_goal,
            timeout=timeout,
            min_separation=min_separation,
            discomfort_dist=SMOKE_DISCOMFORT_DIST,
            goal_distance=dist,
            progress=progress,
            time_step=SMOKE_TIME_STEP,
            global_time=global_time,
            time_limit=SMOKE_TIME_LIMIT,
        )

    crowd = (
        HumanState(px=1.5, py=0.0, vx=-0.8, vy=0.3, radius=SMOKE_RADIUS),
        HumanState(px=-2.0, py=1.0, vx=0.7, vy=-0.5, radius=SMOKE_RADIUS),
        HumanState(px=0.5, py=3.0, vx=0.0, vy=-1.0, radius=SMOKE_RADIUS),
    )
    close = (HumanState(px=0.4, py=-1.4, vx=-0.5, vy=0.0, radius=SMOKE_RADIUS),)
    return (
        # Heading to the goal through a crowd
        _state(humans=crowd),
        # Terminal outcomes
        _state(collision=True, min_separation=-0.05, humans=close, py=-1.0, global_time=3.0),
        _state(reached_goal=True, py=3.8, progress=0.25, humans=crowd, global_time=9.0),
        _state(timeout=True, vx=0.0, vy=0.0, progress=0.0, humans=crowd, global_time=24.0),
        # Stopped (freezing robot)
        _state(vx=0.0, vy=0.0, progress=0.0, humans=crowd, global_time=5.0),
        # Discomfort zone
        _state(humans=close, min_separation=0.1, py=-1.6, global_time=2.5),
        # Moving away from the goal
        _state(vx=0.0, vy=-1.0, progress=-0.25, py=-3.0, humans=crowd, global_time=1.0),
        # Sideways dodge at speed
        _state(vx=0.9, vy=0.2, progress=0.05, px=0.8, py=-0.5, humans=crowd, global_time=4.0),
        # Start of an episode, empty surroundings
        _state(py=-3.75, progress=0.25, global_time=0.25),
    )


def fingerprint_smoke_states(
    *,
    dataset_path: str | None = None,
    max_dataset_frames: int = 6,
) -> Tuple[CrowdNavRewardState, ...]:
    """
    Behaviour-fingerprint states: smoke suite + optional Stage I trajectory frames.

    Real frames (a success mid-episode, a goal frame, a collision frame and a
    timeout frame) make the sandbox smoke test more discriminative.
    """
    base = list(default_smoke_states())
    # stage1 imports this module (state dataclasses), so import lazily.
    from domains.crowdnav.stage1 import load_crowdnav_dataset, resolve_dataset_path

    try:
        trajs = load_crowdnav_dataset(resolve_dataset_path(dataset_path))
    except Exception:  # noqa: BLE001 — missing / unreadable dataset keeps the smoke suite
        return tuple(base)

    buckets = {"success_mid": None, "goal": None, "collision": None, "timeout": None}
    settings = get_settings()
    for traj in trajs:
        states = with_kinematics(traj.states, settings)
        if traj.label == "success":
            if buckets["success_mid"] is None:
                buckets["success_mid"] = states[len(states) // 2]
            if buckets["goal"] is None:
                buckets["goal"] = states[-1]
        elif traj.label in ("collision", "timeout") and buckets[traj.label] is None:
            buckets[traj.label] = states[-1]
        if all(v is not None for v in buckets.values()):
            break
    picked = [st for st in buckets.values() if st is not None][: int(max_dataset_frames)]
    return tuple(base + picked)


__all__ = [
    "CrowdNavRewardState",
    "HumanState",
    "RobotState",
    "default_smoke_states",
    "fingerprint_smoke_states",
    "holonomic_to_unicycle",
    "smoke_action",
    "with_kinematics",
]
