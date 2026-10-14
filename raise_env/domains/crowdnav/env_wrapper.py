"""
Gymnasium env around the upstream CrowdNav simulator with sandboxed rewards.

``RewardInjectedCrowdNavEnv`` owns one unmodified ``crowd_sim.envs.CrowdSim``
(vita-epfl/CrowdNav @ 20d6780, configured from ``env.config``) and only
translates at the boundary:

* actions: discrete CADRL action index or continuous ``Box([-1, 1]^2)``
  -> upstream ``ActionXY`` / ``ActionRot``;
* observations: upstream joint state -> fixed-size robot-centric vector
  (``CADRL.rotate`` per human, self features once);
* episode end: upstream ``done`` + ``info`` -> gymnasium
  ``terminated`` (``Collision`` / ``ReachGoal``) or ``truncated`` (``Timeout``);
* reward: ``reward_fn.compute(CrowdNavRewardState)``; the upstream reward is
  kept in ``info["raise_native_reward"]`` (and used when ``reward_fn`` is None).

Scenario seeding. Upstream ``reset(phase, test_case)`` seeds the global numpy
RNG with ``offset[phase] + case`` (val 0+, test 1000+, train 2000+) and draws
the human start / goal positions. ``reset(seed=s)`` maps ``s`` to a case of
the env's phase (identity modulo the phase size for ``val`` / ``test``, a
hashed start for ``train``); without a seed the upstream case counter simply
advances. The global numpy RNG state is restored after every reset, so the
env does not perturb other consumers of ``np.random``. Stepping is
deterministic (ORCA humans, no noise).
"""

from __future__ import annotations

import configparser
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import raise_paths  # noqa: F401 — vendored crowd_sim on sys.path
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.utils import point_to_segment_dist

from domains.crowdnav.render import render_frame
from domains.crowdnav.settings import (
    CrowdNavSettings,
    continuous_to_velocity,
    discrete_action_table,
    get_settings,
    validate_settings,
)
from domains.crowdnav.state import CrowdNavRewardState, HumanState, RobotState
from domains.crowdnav.watchdog import beat as watchdog_beat

PHASES = ("train", "val", "test")
# dg, v_pref, theta, radius, vx, vy (robot) | px1, py1, vx1, vy1, radius1, da, radius_sum (per human)
SELF_FEATURES = 6
HUMAN_FEATURES = 7
_EVAL_CASE_STRIDE = 97
_REQUIRED_SECTIONS = ("env", "reward", "sim", "humans", "robot")


def load_env_config(path: str) -> configparser.RawConfigParser:
    """Parse an upstream-format ``env.config``."""
    cfg = configparser.RawConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"CrowdNav env config not found: {path}")
    missing = [s for s in _REQUIRED_SECTIONS if not cfg.has_section(s)]
    if missing:
        raise ValueError(f"CrowdNav env config {path} lacks sections {missing}")
    return cfg


class ExternalRobotPolicy(Policy):
    """
    Robot policy placeholder: actions come from the RAISE PPO policy via ``step``.

    Upstream ``reset`` reads ``robot.policy.multiagent_training`` (True: all
    ``human_num`` humans in train/val, as for SARL) and sets
    ``policy.time_step``; the robot's kinematics come from the policy.
    """

    def __init__(self, kinematics: str) -> None:
        super().__init__()
        self.name = "raise_ppo"
        self.trainable = True
        self.kinematics = str(kinematics)
        self.multiagent_training = True

    def configure(self, config: Any) -> None:
        return None

    def predict(self, state: Any) -> Any:
        raise RuntimeError("the robot is driven through RewardInjectedCrowdNavEnv.step")


def make_crowd_sim(env_config: configparser.RawConfigParser, kinematics: str) -> CrowdSim:
    """Configured upstream ``CrowdSim`` with an externally driven robot."""
    sim = CrowdSim()
    sim.configure(env_config)
    for rule in (sim.train_val_sim, sim.test_sim):
        if rule == "mixed":
            # 'mixed' redraws the human count every episode; the policy
            # observation has a fixed number of human slots.
            raise ValueError("CrowdNav 'mixed' scenarios (variable human count) are not supported")
    if int(sim.human_num) < 1:
        raise ValueError("CrowdNav env needs human_num >= 1")
    robot = Robot(env_config, "robot")
    robot.set_policy(ExternalRobotPolicy(kinematics))
    sim.set_robot(robot)
    return sim


def scenario_case(seed: int, phase: str, size: int) -> int:
    """Upstream case index for ``reset(seed=...)`` in ``phase``."""
    if phase == "train":
        # Hashed start: consecutive PPO env seeds (s, s+1, ...) must not
        # replay overlapping case sequences.
        draw = np.random.SeedSequence(abs(int(seed))).generate_state(1, dtype=np.uint64)[0]
        return int(draw % np.uint64(size))
    return int(seed) % int(size)


def eval_cases(seed_base: int, n_episodes: int, size: int) -> List[int]:
    """``n_episodes`` distinct cases (while ``n_episodes <= size``) for one evaluation."""
    start = int(
        np.random.SeedSequence(abs(int(seed_base))).generate_state(1, dtype=np.uint64)[0]
        % np.uint64(size)
    )
    stride = _EVAL_CASE_STRIDE if math.gcd(_EVAL_CASE_STRIDE, int(size)) == 1 else 1
    return [(start + i * stride) % int(size) for i in range(int(n_episodes))]


def step_min_separation(sim: CrowdSim, action: Any) -> float:
    """
    Upstream ``dmin`` of the coming step (``CrowdSim.step`` collision check).

    Closest boundary distance between the robot and each human along this
    step's relative motion; unlike upstream it does not stop at the first
    colliding human, so a collision step reports the most negative value.
    """
    robot = sim.robot
    dmin = float("inf")
    for human in sim.humans:
        px = human.px - robot.px
        py = human.py - robot.py
        if robot.kinematics == "holonomic":
            vx = human.vx - action.vx
            vy = human.vy - action.vy
        else:
            vx = human.vx - action.v * np.cos(action.r + robot.theta)
            vy = human.vy - action.v * np.sin(action.r + robot.theta)
        ex = px + vx * sim.time_step
        ey = py + vy * sim.time_step
        dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot.radius
        dmin = min(dmin, float(dist))
    return dmin


def robot_centric_observation(sim: CrowdSim) -> np.ndarray:
    """
    ``CADRL.rotate`` applied to every (robot, human) pair; robot features once.

    The x axis points from the robot to its goal. Layout:
    ``[dg, v_pref, theta, radius, vx, vy]`` then per human
    ``[px1, py1, vx1, vy1, radius1, da, radius_sum]``. ``theta`` is the robot
    heading relative to the goal direction for unicycle robots, else 0
    (as upstream), wrapped to [-pi, pi).
    """
    r = sim.robot
    dx, dy = r.gx - r.px, r.gy - r.py
    rot = math.atan2(dy, dx)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    theta = 0.0
    if r.kinematics == "unicycle":
        theta = (r.theta - rot + math.pi) % (2.0 * math.pi) - math.pi
    out = [
        math.hypot(dx, dy),
        r.v_pref,
        theta,
        r.radius,
        r.vx * cos_r + r.vy * sin_r,
        r.vy * cos_r - r.vx * sin_r,
    ]
    for h in sim.humans:
        rx, ry = h.px - r.px, h.py - r.py
        out += [
            rx * cos_r + ry * sin_r,
            ry * cos_r - rx * sin_r,
            h.vx * cos_r + h.vy * sin_r,
            h.vy * cos_r - h.vx * sin_r,
            h.radius,
            math.hypot(rx, ry),
            r.radius + h.radius,
        ]
    return np.asarray(out, dtype=np.float32)


def outcome_of(info: Any) -> Optional[str]:
    """``success`` / ``collision`` / ``timeout`` for terminal upstream infos, else None."""
    if isinstance(info, ReachGoal):
        return "success"
    if isinstance(info, Collision):
        return "collision"
    if isinstance(info, Timeout):
        return "timeout"
    return None


class RewardInjectedCrowdNavEnv(gym.Env):
    """Upstream CrowdSim whose reward is ``reward_fn.compute(CrowdNavRewardState)``."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        reward_fn: Any = None,
        *,
        phase: str = "train",
        seed: Optional[int] = None,
        settings: Optional[CrowdNavSettings] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.settings = validate_settings(settings or get_settings())
        self.env_config = load_env_config(self.settings.env_config)
        self.sim = make_crowd_sim(self.env_config, self.settings.kinematics)
        self.reward_fn = reward_fn
        self.phase = phase
        self.render_mode = render_mode
        self._seed = seed
        self._seed_consumed = False

        robot = self.sim.robot
        self.time_step = float(self.sim.time_step)
        self.time_limit = float(self.sim.time_limit)
        self.discomfort_dist = float(self.sim.discomfort_dist)
        self.human_num = int(self.sim.human_num)
        self.v_pref = float(robot.v_pref)
        if self.settings.action_mode == "discrete":
            self.action_table = discrete_action_table(self.settings, self.v_pref)
            self.action_space = gym.spaces.Discrete(len(self.action_table))
        else:
            self.action_table = []
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        dim = SELF_FEATURES + HUMAN_FEATURES * self.human_num
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

        self.current_phase = phase
        self.case: Optional[int] = None
        self.last_state: Optional[CrowdNavRewardState] = None
        self._goal_distance = 0.0
        self._trail: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        opts = dict(options or {})
        phase = str(opts.get("phase", self.phase))
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        size = int(self.sim.case_size[phase])
        case: Optional[int] = None
        if opts.get("case") is not None:
            case = int(opts["case"]) % size
        elif seed is not None:
            case = scenario_case(int(seed), phase, size)
            self._seed_consumed = True
        elif self._seed is not None and not self._seed_consumed:
            # Construction seed applies once; later resets advance the upstream counter.
            case = scenario_case(int(self._seed), phase, size)
            self._seed_consumed = True
        used = int(case) if case is not None else int(self.sim.case_counter[phase])

        rng_state = np.random.get_state()
        try:
            self.sim.reset(phase=phase, test_case=case)
        finally:
            np.random.set_state(rng_state)

        robot = self.sim.robot
        self.current_phase = phase
        self.case = used
        self.last_state = None
        self._goal_distance = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        self._trail = [(float(robot.px), float(robot.py))]
        info = {"raise_phase": phase, "raise_case": used}
        return robot_centric_observation(self.sim), info

    def step(self, action: Any):
        return self.step_upstream(self.to_upstream_action(action))

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        r = self.sim.robot
        return render_frame(
            robot_xy=(float(r.px), float(r.py)),
            robot_radius=float(r.radius),
            goal_xy=(float(r.gx), float(r.gy)),
            humans=[(float(h.px), float(h.py), float(h.radius)) for h in self.sim.humans],
            global_time=float(self.sim.global_time or 0.0),
            trail=self._trail,
        )

    # ------------------------------------------------------------------ helpers

    def to_upstream_action(self, action: Any) -> Any:
        """Policy action -> upstream ``ActionXY`` / ``ActionRot``."""
        if self.settings.action_mode == "discrete":
            idx = int(np.asarray(action).reshape(-1)[0])
            a, b = self.action_table[idx]
        else:
            a, b = continuous_to_velocity(action, self.settings, self.v_pref)
        if self.settings.kinematics == "holonomic":
            return ActionXY(float(a), float(b))
        return ActionRot(float(a), float(b))

    def step_upstream(self, up_action: Any):
        """One upstream ``CrowdSim.step`` with an upstream action (collectors call this directly)."""
        watchdog_beat()
        sim = self.sim
        dmin = step_min_separation(sim, up_action)
        prev_dist = self._goal_distance
        _ob, native_reward, _done, info_obj = sim.step(up_action)

        timeout = isinstance(info_obj, Timeout)
        collision = isinstance(info_obj, Collision)
        reached = isinstance(info_obj, ReachGoal)
        danger = isinstance(info_obj, Danger)
        robot = sim.robot
        goal_dist = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        self._goal_distance = goal_dist
        self._trail.append((float(robot.px), float(robot.py)))

        state = CrowdNavRewardState(
            robot=RobotState(
                px=float(robot.px),
                py=float(robot.py),
                vx=float(robot.vx),
                vy=float(robot.vy),
                gx=float(robot.gx),
                gy=float(robot.gy),
                theta=float(robot.theta),
                radius=float(robot.radius),
                v_pref=float(robot.v_pref),
            ),
            humans=tuple(
                HumanState(
                    px=float(h.px),
                    py=float(h.py),
                    vx=float(h.vx),
                    vy=float(h.vy),
                    radius=float(h.radius),
                )
                for h in sim.humans
            ),
            action=(float(up_action[0]), float(up_action[1])),
            collision=collision,
            reached_goal=reached,
            timeout=timeout,
            min_separation=float(dmin),
            discomfort_dist=self.discomfort_dist,
            goal_distance=float(goal_dist),
            progress=float(prev_dist - goal_dist),
            time_step=self.time_step,
            global_time=float(sim.global_time),
            time_limit=self.time_limit,
        )
        self.last_state = state

        comps: Dict[str, float] = {}
        if self.reward_fn is None:
            reward = float(native_reward)
        else:
            reward = float(self.reward_fn.compute(state))
            getter = getattr(self.reward_fn, "last_reward_components", None)
            if callable(getter):
                raw = getter()
                if isinstance(raw, dict):
                    comps = {str(k): float(v) for k, v in raw.items()}

        info = {
            "raise_outcome": outcome_of(info_obj),
            "raise_collision": collision,
            "raise_reached_goal": reached,
            "raise_timeout": timeout,
            "raise_danger": danger,
            "raise_danger_min_dist": float(info_obj.min_dist) if danger else None,
            "raise_min_separation": float(dmin),
            "raise_native_reward": float(native_reward),
            "raise_progress": float(state.progress),
            "raise_speed": math.hypot(state.robot.vx, state.robot.vy),
            "raise_goal_distance": float(goal_dist),
            "raise_global_time": float(sim.global_time),
            "raise_phase": self.current_phase,
            "raise_case": self.case,
            "raise_reward_components": comps,
        }
        terminated = bool(collision or reached)
        truncated = bool(timeout)
        return robot_centric_observation(sim), reward, terminated, truncated, info


__all__ = [
    "HUMAN_FEATURES",
    "PHASES",
    "SELF_FEATURES",
    "ExternalRobotPolicy",
    "RewardInjectedCrowdNavEnv",
    "eval_cases",
    "load_env_config",
    "make_crowd_sim",
    "outcome_of",
    "robot_centric_observation",
    "scenario_case",
    "step_min_separation",
]
