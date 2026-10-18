"""CrowdNavRewardState — the sandbox-visible view of one CrowdNav step."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from domains.crowdnav.settings import CrowdNavSettings, get_settings


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
    kinematics: str = "unicycle"


@dataclass(frozen=True)
class HumanState:
    """One human's observable state after the step (upstream ``ObservableState``)."""

    px: float
    py: float
    vx: float
    vy: float
    radius: float
    visible: bool = False


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
    danger: bool = False
    success_reward: float = 1.0
    collision_penalty: float = -0.25
    discomfort_penalty_factor: float = 0.5
    potential_delta: float = 0.0
    potential_factor: float = 2.0


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
        out.append(
            replace(
                st,
                action=action,
                robot=replace(st.robot, theta=theta, kinematics="unicycle"),
                potential_factor=3.0,
            )
        )
    return tuple(out)


def smoke_action(vx: float, vy: float) -> Tuple[float, float]:
    """``state.action`` for a smoke state under the active kinematics."""
    if get_settings().kinematics == "holonomic":
        return (float(vx), float(vy))
    action, _ = holonomic_to_unicycle(vx, vy, math.pi / 2.0)
    return action


def default_smoke_states() -> Tuple[CrowdNavRewardState, ...]:
    """Deterministic snapshots for RewardValidator smoke tests."""
    from domains.crowdnav.env_wrapper import load_env_config

    config = load_env_config(get_settings().env_config_source)
    circle_radius = float(config.sim.circle_radius)
    time_step = float(config.env.time_step)
    time_limit = float(config.env.time_limit)
    discomfort_dist = float(config.reward.discomfort_dist)
    robot_radius = float(config.robot.radius)
    robot_v_pref = float(config.robot.v_pref)
    human_radius = float(config.humans.radius)
    human_v_pref = float(config.humans.v_pref)
    safe_separation = robot_radius + human_radius + discomfort_dist + 0.5
    goal = (0.0, circle_radius)
    near_goal_y = circle_radius * 0.95
    start_y = -circle_radius * 0.5
    away_y = -circle_radius * 0.75

    def _state(
        *,
        py: float = start_y,
        px: float = 0.0,
        vx: float = 0.0,
        vy: float = robot_v_pref,
        humans: Tuple[HumanState, ...] = (),
        collision: bool = False,
        reached_goal: bool = False,
        timeout: bool = False,
        min_separation: float = safe_separation,
        progress: float = time_step,
        global_time: float = time_limit * 0.1,
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
                radius=robot_radius,
                v_pref=robot_v_pref,
                kinematics=get_settings().kinematics,
            ),
            humans=humans,
            action=smoke_action(vx, vy),
            collision=collision,
            reached_goal=reached_goal,
            timeout=timeout,
            min_separation=min_separation,
            discomfort_dist=discomfort_dist,
            goal_distance=dist,
            progress=progress,
            time_step=time_step,
            global_time=global_time,
            time_limit=time_limit,
            danger=min_separation < discomfort_dist,
            discomfort_penalty_factor=float(config.reward.discomfort_penalty_factor),
            success_reward=float(config.reward.success_reward),
            collision_penalty=float(config.reward.collision_penalty),
            potential_delta=float(progress),
            potential_factor=3.0 if get_settings().kinematics == "unicycle" else 2.0,
        )

    crowd = (
        HumanState(
            px=circle_radius * 0.375,
            py=0.0,
            vx=-0.8 * human_v_pref,
            vy=0.3 * human_v_pref,
            radius=human_radius,
            visible=True,
        ),
        HumanState(
            px=-circle_radius * 0.5,
            py=circle_radius * 0.25,
            vx=0.7 * human_v_pref,
            vy=-0.5 * human_v_pref,
            radius=human_radius,
        ),
        HumanState(
            px=circle_radius * 0.125,
            py=circle_radius * 0.75,
            vx=0.0,
            vy=-human_v_pref,
            radius=human_radius,
            visible=True,
        ),
    )
    close = (
        HumanState(
            px=robot_radius + human_radius + discomfort_dist / 2.0,
            py=start_y,
            vx=-0.5 * human_v_pref,
            vy=0.0,
            radius=human_radius,
            visible=True,
        ),
    )
    return (
        # Heading to the goal through a crowd
        _state(humans=crowd, min_separation=discomfort_dist * 2.0),
        # Terminal outcomes
        _state(
            collision=True,
            min_separation=-discomfort_dist / 4.0,
            humans=close,
            py=start_y / 2.0,
            global_time=time_limit * 0.15,
        ),
        _state(
            reached_goal=True,
            py=near_goal_y,
            progress=time_step,
            humans=crowd,
            global_time=time_limit * 0.4,
        ),
        _state(
            timeout=True,
            vx=0.0,
            vy=0.0,
            progress=0.0,
            humans=crowd,
            global_time=max(0.0, time_limit - time_step),
        ),
        # Stopped (freezing robot)
        _state(vx=0.0, vy=0.0, progress=0.0, humans=crowd, global_time=time_limit * 0.2),
        # Discomfort zone
        _state(
            humans=close,
            min_separation=discomfort_dist / 2.0,
            py=start_y * 0.8,
            global_time=time_limit * 0.1 + time_step,
        ),
        # Moving away from the goal
        _state(
            vx=0.0,
            vy=-1.0,
            progress=-time_step,
            py=away_y,
            humans=crowd,
            global_time=time_step,
        ),
        # Sideways dodge at speed
        _state(
            vx=0.9 * robot_v_pref,
            vy=0.2 * robot_v_pref,
            progress=time_step / 5.0,
            px=circle_radius * 0.1,
            py=start_y * 0.25,
            humans=crowd,
            global_time=time_limit * 0.16,
        ),
        # Start of an episode, empty surroundings
        _state(py=-circle_radius * 0.95, progress=time_step, global_time=time_step),
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
