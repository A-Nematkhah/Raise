"""
RewardState snapshot and RewardFunction ABC.

Mirrors mobile_robot_env.rewards.base naming (RewardFunction.reset/compute)
so both codebases can share candidate / sandbox tooling later. CrowdNav uses
RewardState (crowd humans + Rsense) instead of RewardContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RobotRewardState:
    """Robot fields a candidate reward may read."""

    px: float
    py: float
    vx: float
    vy: float
    radius: float
    gx: float
    gy: float
    v_pref: float


@dataclass(frozen=True)
class HumanObservable:
    """
    Human observation fields matching crowd_sim ObservableState
    (px, py, vx, vy, radius). Defined here so reward_search does not import
    crowd_sim (and its rvo2 dependency) at module load time.
    """

    px: float
    py: float
    vx: float
    vy: float
    radius: float


@dataclass(frozen=True)
class RewardState:
    """
    Exact fields a candidate reward function may read for one env step.

    Built by the environment from the same local variables calc_reward already
    computes (dmin, collision, reaching_goal, timeout). Candidates must not
    influence done / episode_info.
    """

    robot: RobotRewardState
    humans: Tuple[HumanObservable, ...]
    dmin: float
    discomfort_dist: float
    collision: bool
    reaching_goal: bool
    timeout: bool
    action: Any
    time_step: float
    global_time: float
    time_limit: float


class RewardFunction(ABC):
    """Abstract interface every reward implementation must satisfy."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode, before any compute() call."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, state: RewardState) -> float:
        """Compute the scalar reward for one step, given state."""
        raise NotImplementedError


class LegacyReward(RewardFunction):
    """
    Exact reproduction of CrowdSim.calc_reward's original scalar formula.

    Stateful only for the potential-based shaping term (self.potential in the
    original env). Call reset() after the env initializes potential so the
    first step matches byte-for-byte.
    """

    def __init__(
        self,
        *,
        success_reward: float,
        collision_penalty: float,
        discomfort_dist: float,
        discomfort_penalty_factor: float,
        time_step: float,
        kinematics: str = "holonomic",
        pot_factor: float = 2.0,
        unicycle_spin_coef: float = -5.0,
    ) -> None:
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.discomfort_dist = float(discomfort_dist)
        self.discomfort_penalty_factor = float(discomfort_penalty_factor)
        self.time_step = float(time_step)
        self.kinematics = kinematics
        self.pot_factor = float(pot_factor)
        self.unicycle_spin_coef = float(unicycle_spin_coef)
        self._potential: Optional[float] = None

    @classmethod
    def from_crowd_sim(cls, env: Any) -> "LegacyReward":
        """Build coeffs from a configured CrowdSim (base-class formula)."""
        return cls(
            success_reward=env.success_reward,
            collision_penalty=env.collision_penalty,
            discomfort_dist=env.discomfort_dist,
            discomfort_penalty_factor=env.discomfort_penalty_factor,
            time_step=env.time_step,
            kinematics=getattr(env.robot, "kinematics", "holonomic")
            if getattr(env, "robot", None) is not None
            else env.config.action_space.kinematics,
            pot_factor=2.0,
            unicycle_spin_coef=-5.0,
        )

    @classmethod
    def from_crowd_sim_var_num(cls, env: Any) -> "LegacyReward":
        """Build coeffs matching CrowdSimVarNum.calc_reward's original formula."""
        kinematics = getattr(env, "action_type", None)
        if kinematics is None and getattr(env, "robot", None) is not None:
            kinematics = env.robot.kinematics
        if kinematics is None:
            kinematics = env.config.action_space.kinematics
        pot_factor = 2.0 if kinematics == "holonomic" else 3.0
        return cls(
            success_reward=env.success_reward,
            collision_penalty=env.collision_penalty,
            discomfort_dist=env.discomfort_dist,
            discomfort_penalty_factor=env.discomfort_penalty_factor,
            time_step=env.time_step,
            kinematics=kinematics,
            pot_factor=pot_factor,
            unicycle_spin_coef=-4.5,
        )

    def sync_potential(self, potential: float) -> None:
        """Copy env.potential after reset (must match original init)."""
        self._potential = float(potential)

    def reset(self) -> None:
        # Potential is synced from the env after it computes the initial value.
        self._potential = None

    def compute(self, state: RewardState) -> float:
        if state.timeout:
            reward = 0.0
        elif state.collision:
            reward = self.collision_penalty
        elif state.reaching_goal:
            reward = self.success_reward
        elif state.dmin < self.discomfort_dist:
            reward = (
                (state.dmin - self.discomfort_dist)
                * self.discomfort_penalty_factor
                * self.time_step
            )
        else:
            potential_cur = (
                (state.robot.px - state.robot.gx) ** 2
                + (state.robot.py - state.robot.gy) ** 2
            ) ** 0.5
            if self._potential is None:
                # Should have been synced at reset; fall back to original init.
                self._potential = -abs(potential_cur)
            reward = self.pot_factor * (-abs(potential_cur) - self._potential)
            self._potential = -abs(potential_cur)

        if self.kinematics == "unicycle":
            action = state.action
            r_spin = self.unicycle_spin_coef * (action.r ** 2)
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.0
            reward = reward + r_spin + r_back

        return float(reward)


def humans_within_rsense(
    robot: Any,
    humans: Sequence[Any],
    sensor_range: float,
) -> Tuple[HumanObservable, ...]:
    """HumanObservable for every human within robot sensor range (Rsense)."""
    in_range = []
    for human in humans:
        dist = (
            (robot.px - human.px) ** 2 + (robot.py - human.py) ** 2
        ) ** 0.5 - robot.radius - human.radius
        if dist <= sensor_range:
            in_range.append(
                HumanObservable(human.px, human.py, human.vx, human.vy, human.radius)
            )
    return tuple(in_range)


def build_reward_state(
    env: Any,
    action: Any,
    *,
    dmin: float,
    collision: bool,
    reaching_goal: bool,
    timeout: bool,
) -> RewardState:
    """Assemble RewardState from env + already-computed detection locals."""
    robot = env.robot
    return RewardState(
        robot=RobotRewardState(
            px=float(robot.px),
            py=float(robot.py),
            vx=float(robot.vx),
            vy=float(robot.vy),
            radius=float(robot.radius),
            gx=float(robot.gx),
            gy=float(robot.gy),
            v_pref=float(robot.v_pref),
        ),
        humans=humans_within_rsense(robot, env.humans, robot.sensor_range),
        dmin=float(dmin),
        discomfort_dist=float(env.discomfort_dist),
        collision=bool(collision),
        reaching_goal=bool(reaching_goal),
        timeout=bool(timeout),
        action=action,
        time_step=float(env.time_step),
        global_time=float(env.global_time),
        time_limit=float(env.time_limit),
    )
