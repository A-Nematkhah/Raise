"""ORCA / linear reference robot controllers for CrowdNav++ rollouts."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Tuple

import numpy as np
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState

from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv
from domains.crowdnav.settings import (
    UNICYCLE_ROTATION_DELTA_MAX,
    UNICYCLE_SPEED_DELTA_MAX,
    UNICYCLE_SPEED_DELTA_MIN,
)

# (safety_space m, speed scale, time_horizon s). None time_horizon -> ORCA default (5 s).
_ORCA_SPECS: Dict[str, Tuple[float, float, float | None]] = {
    "orca": (0.0, 1.0, None),
    "orca_ss005": (0.05, 1.0, None),
    "orca_ss010": (0.10, 1.0, None),
    "orca_cautious": (0.15, 1.0, None),
    "orca_ss025": (0.25, 1.0, None),
    "orca_slow08": (0.0, 0.8, None),
    "orca_slow06": (0.0, 0.6, None),
    "orca_reckless": (0.0, 1.0, 1.0),
}
BEHAVIOURS = tuple(_ORCA_SPECS) + ("linear", "linear_slow", "stop", "random")
_SLOW_FRACTION = 0.3


def joint_state(env: RewardInjectedCrowdNavEnv) -> JointState:
    visibility = env.sim.human_visibility
    if len(visibility) != len(env.sim.humans):
        raise RuntimeError("CrowdNav++ visibility flags do not match its human population")
    return JointState(
        env.sim.robot.get_full_state(),
        [
            human.get_observable_state()
            for human, visible in zip(env.sim.humans, visibility)
            if visible
        ],
    )


def _velocity_action(env: RewardInjectedCrowdNavEnv, vx: float, vy: float) -> Any:
    if env.settings.kinematics == "holonomic":
        return ActionXY(float(vx), float(vy))

    robot = env.sim.robot
    speed = min(math.hypot(vx, vy), float(robot.v_pref))
    target_heading = math.atan2(vy, vx) if speed > 0.0 else float(robot.theta)
    heading_delta = (target_heading - float(robot.theta) + math.pi) % (2.0 * math.pi) - math.pi
    speed_delta = speed - float(env.sim.desiredVelocity[0])
    return ActionRot(
        float(np.clip(speed_delta, UNICYCLE_SPEED_DELTA_MIN, UNICYCLE_SPEED_DELTA_MAX)),
        float(np.clip(heading_delta, -UNICYCLE_ROTATION_DELTA_MAX, UNICYCLE_ROTATION_DELTA_MAX)),
    )


def _stop_action(env: RewardInjectedCrowdNavEnv) -> Any:
    if env.settings.kinematics == "holonomic":
        return ActionXY(0.0, 0.0)
    speed = float(env.sim.desiredVelocity[0])
    return ActionRot(
        float(np.clip(-speed, UNICYCLE_SPEED_DELTA_MIN, UNICYCLE_SPEED_DELTA_MAX)),
        0.0,
    )


def make_controller(
    behaviour: str, env: RewardInjectedCrowdNavEnv, rng: np.random.Generator
) -> Callable[[], Any]:
    """Per-step upstream action for ``behaviour`` (called after ``env.reset``)."""
    if behaviour in _ORCA_SPECS:
        safety, speed_scale, time_horizon = _ORCA_SPECS[behaviour]
        config = copy.deepcopy(env.env_config)
        config.orca.safety_space = float(safety)
        if time_horizon is not None:
            config.orca.time_horizon = float(time_horizon)
        policy = ORCA(config)
        policy.time_step = env.time_step
        policy.safety_space = float(safety)

        def _orca() -> Any:
            desired = policy.predict(joint_state(env))
            return _velocity_action(
                env, desired.vx * speed_scale, desired.vy * speed_scale
            )

        return _orca
    if behaviour in ("linear", "linear_slow"):
        scale = 1.0 if behaviour == "linear" else _SLOW_FRACTION

        def _linear() -> Any:
            robot = env.sim.robot
            dx = float(robot.gx - robot.px)
            dy = float(robot.gy - robot.py)
            distance = math.hypot(dx, dy)
            speed = float(robot.v_pref) * scale
            if distance == 0.0:
                return _stop_action(env)
            return _velocity_action(env, dx * speed / distance, dy * speed / distance)

        return _linear
    if behaviour == "stop":
        return lambda: _stop_action(env)
    if behaviour == "random":
        if env.settings.kinematics == "holonomic":
            table = env.action_table
            return lambda: ActionXY(*table[int(rng.integers(len(table)))])
        return lambda: ActionRot(
            float(rng.uniform(UNICYCLE_SPEED_DELTA_MIN, UNICYCLE_SPEED_DELTA_MAX)),
            float(rng.uniform(-UNICYCLE_ROTATION_DELTA_MAX, UNICYCLE_ROTATION_DELTA_MAX)),
        )
    raise ValueError(f"unknown behaviour {behaviour!r}")


__all__ = ["BEHAVIOURS", "joint_state", "make_controller"]
