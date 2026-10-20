"""
Gymnasium adapter for CrowdNav++'s variable-human ``CrowdSimVarNum``.

The simulator is configured from the upstream Python ``Config`` object/file.
CrowdNav++ seeds global NumPy in ``reset``; the wrapper serializes simulator
RNG access and keeps a per-env RNG state so resets and later random events do
not perturb or interleave with other consumers.

The upstream variable-human ``step`` always requests future-danger scoring,
but only computes ``human_future_traj`` while ``phase == 'test'``. For val
cases the wrapper therefore resets with the real val phase (preserving its
seed offset and case counter), then uses the test-time future-danger branch
while stepping, identically to test evaluation. No upstream files are changed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import logging
import math
import os
import sys
import threading
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

import gymnasium as gym
import numpy as np

import raise_paths  # noqa: F401 — active crowd_sim vendor on sys.path
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout
from crowd_nav.policy.srnn import SRNN

from domains.crowdnav.render import render_frame
from domains.crowdnav.settings import (
    CrowdNavSettings,
    UNICYCLE_ROTATION_DELTA_MAX,
    UNICYCLE_SPEED_DELTA_MAX,
    UNICYCLE_SPEED_DELTA_MIN,
    continuous_to_velocity,
    discrete_action_table,
    get_settings,
    validate_settings,
)
from domains.crowdnav.state import CrowdNavRewardState, HumanState, RobotState
from domains.crowdnav.watchdog import beat as watchdog_beat

logger = logging.getLogger(__name__)

PHASES = ("train", "val", "test")
HUMAN_OBSERVATION_KEYS = (
    "robot_node",
    "temporal_edges",
    "visible_masks",
    "spatial_edges",
    "detected_human_num",
)
_REQUIRED_CONFIG_SECTIONS = (
    "env",
    "reward",
    "sim",
    "humans",
    "robot",
    "action_space",
    "orca",
    "data",
    "sim2real",
)
_UPSTREAM_RNG_LOCK = threading.RLock()
_CONFIG_IMPORT_LOCK = threading.RLock()
_EVAL_CASE_STRIDE = 97


@contextmanager
def _preserve_numpy_rng() -> Iterator[None]:
    with _UPSTREAM_RNG_LOCK:
        state = np.random.get_state()
        try:
            yield
        finally:
            np.random.set_state(state)


def _configuration_candidate(module: types.ModuleType) -> Any:
    candidate = getattr(module, "Config", None)
    if candidate is None:
        candidate = getattr(module, "config", None)
    if candidate is None:
        factory = getattr(module, "get_config", None)
        if callable(factory):
            candidate = factory()
    if candidate is None:
        raise ValueError(
            f"Python env config {module.__file__!r} must expose Config, config, "
            "or get_config()"
        )
    return candidate() if isinstance(candidate, type) else candidate


def _clone_configuration(candidate: Any) -> Any:
    if isinstance(candidate, types.ModuleType):
        candidate = _configuration_candidate(candidate)
    elif isinstance(candidate, type):
        candidate = candidate()

    cloned = copy.copy(candidate)
    attributes: Dict[str, Any] = {}
    for cls in reversed(type(candidate).__mro__):
        attributes.update(
            {
                name: value
                for name, value in vars(cls).items()
                if not name.startswith("__") and not callable(value)
            }
        )
    attributes.update(vars(candidate))
    for name, value in attributes.items():
        if not name.startswith("__") and not callable(value):
            setattr(cloned, name, copy.deepcopy(value))
    return cloned


def _load_config_module(path: str) -> types.ModuleType:
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"CrowdNav++ Python config not found: {resolved}")
    if os.path.splitext(resolved)[1].lower() != ".py":
        raise ValueError(f"CrowdNav++ env config must be a Python file: {resolved}")

    name = "_raise_crowdnavpp_config_" + hashlib.sha1(
        os.path.normcase(resolved).encode("utf-8")
    ).hexdigest()
    with _CONFIG_IMPORT_LOCK:
        cached = sys.modules.get(name)
        if isinstance(cached, types.ModuleType):
            return cached
        spec = importlib.util.spec_from_file_location(name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load CrowdNav++ Python config: {resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        old_argv = sys.argv
        sys.argv = [old_argv[0] if old_argv else resolved]
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.argv = old_argv
        return module


def load_env_config(source: Any = None) -> Any:
    """Load and isolate an upstream Python config object or ``.py`` file."""
    if source is None:
        source = get_settings().env_config_source
    if isinstance(source, (str, os.PathLike)):
        source = _configuration_candidate(_load_config_module(os.fspath(source)))
    config = _clone_configuration(source)
    missing = [name for name in _REQUIRED_CONFIG_SECTIONS if not hasattr(config, name)]
    if missing:
        raise ValueError(f"CrowdNav++ Python config lacks sections {missing}")
    return config


class ExternalRobotPolicy(SRNN):
    """Keep CrowdNav++'s action clipping while taking actions from the caller."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.name = "raise_external"

    def predict(self, state: Any) -> Any:
        raise RuntimeError("The robot is controlled through RewardInjectedCrowdNavEnv.step")


def make_crowd_sim(
    env_config: Any,
    settings: CrowdNavSettings | str,
) -> CrowdSimVarNum:
    """Build the unmodified CrowdNav++ variable-human simulator."""
    if isinstance(settings, str):
        settings = CrowdNavSettings(kinematics=settings)
    active = validate_settings(settings)
    config = load_env_config(env_config)

    if active.human_num_override is not None:
        config.sim.human_num = active.human_num_override
        config.sim.human_num_range = 0
    human_num = int(config.sim.human_num)
    human_num_range = int(config.sim.human_num_range)
    if human_num < 1 or human_num_range < 0 or human_num_range >= human_num:
        raise ValueError(
            "CrowdNav++ requires human_num >= 1 and "
            "0 <= human_num_range < human_num"
        )
    prediction_method = str(getattr(config.sim, "predict_method", "none")).strip().lower()
    if prediction_method == "inferred":
        # RAISE has no configured GST predictor; preserve its prior no-prediction behavior.
        prediction_method = "none"
    if prediction_method not in ("none", "const_vel"):
        raise ValueError(
            "RAISE supports only 'none' or 'const_vel' human prediction; "
            f"got {prediction_method!r}"
        )
    config.sim.predict_method = prediction_method
    if prediction_method == "const_vel":
        predict_steps = int(config.sim.predict_steps)
        prediction_interval = int(config.data.pred_timestep // config.env.time_step)
        if predict_steps < 1 or prediction_interval < 1:
            raise ValueError(
                "constant-velocity prediction requires predict_steps >= 1 and "
                "data.pred_timestep >= env.time_step"
            )

    config.action_space.kinematics = active.kinematics
    config.robot.policy = "srnn"
    config.env.use_wrapper = False

    with _preserve_numpy_rng():
        sim = CrowdSimVarNum()
        sim.configure(config)
        sim.robot.policy = ExternalRobotPolicy(config)
    sim.thisSeed = 0
    sim.nenv = 1
    sim.phase = None
    sim.test_case = None
    return sim


def scenario_case(seed: int, phase: str, size: int) -> int:
    """Map wrapper seeds to deterministic upstream case indices."""
    size = int(size)
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if size < 1:
        raise ValueError(f"phase case-set size must be >= 1, got {size}")
    if phase == "train":
        draw = np.random.SeedSequence(abs(int(seed))).generate_state(1, dtype=np.uint64)[0]
        return int(draw % np.uint64(size))
    return int(seed) % size


def eval_cases(seed_base: int, n_episodes: int, size: int) -> List[int]:
    """Return a reproducible case cycle, distinct until the case set is exhausted."""
    size = int(size)
    n_episodes = int(n_episodes)
    if size < 1:
        raise ValueError(f"phase case-set size must be >= 1, got {size}")
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
    if n_episodes > size:
        logger.warning(
            "eval_cases: n_episodes=%d > case_set_size=%d; cases will cycle",
            n_episodes,
            size,
        )
    start = int(
        np.random.SeedSequence(abs(int(seed_base))).generate_state(1, dtype=np.uint64)[0]
        % np.uint64(size)
    )
    stride = _EVAL_CASE_STRIDE if math.gcd(_EVAL_CASE_STRIDE, size) == 1 else 1
    return [(start + i * stride) % size for i in range(n_episodes)]


def step_min_separation(sim: CrowdSimVarNum, action: Any = None) -> float:
    """Minimum boundary gap at the start of the step, using upstream's collision geometry.

    CrowdNav++ checks current robot-human positions before moving either agent;
    it does not project relative motion along a segment. This diagnostic reports
    the smallest gap over all humans, including negative gaps on collision.
    """
    del action
    robot = sim.robot
    if robot is None:
        raise RuntimeError("CrowdNav++ simulator has no robot")
    distances = [
        math.hypot(human.px - robot.px, human.py - robot.py)
        - human.radius
        - robot.radius
        for human in sim.humans
    ]
    return float(min(distances)) if distances else float("inf")


def _observation_space(sim: CrowdSimVarNum) -> gym.spaces.Dict:
    max_humans = int(sim.max_human_num)
    if max_humans < 1:
        raise ValueError("CrowdNav++ observation requires at least one human slot")
    prediction_steps = (
        int(sim.config.sim.predict_steps)
        if str(sim.config.sim.predict_method).strip().lower() == "const_vel"
        else 0
    )
    unbounded = np.inf
    return gym.spaces.Dict(
        {
            "robot_node": gym.spaces.Box(
                -unbounded, unbounded, shape=(7,), dtype=np.float32
            ),
            "temporal_edges": gym.spaces.Box(
                -unbounded, unbounded, shape=(2,), dtype=np.float32
            ),
            "visible_masks": gym.spaces.Box(
                np.zeros(max_humans, dtype=np.bool_),
                np.ones(max_humans, dtype=np.bool_),
                dtype=np.bool_,
            ),
            "spatial_edges": gym.spaces.Box(
                -unbounded,
                unbounded,
                shape=(max_humans, 2 * (prediction_steps + 1)),
                dtype=np.float32,
            ),
            "detected_human_num": gym.spaces.Box(
                0, max_humans, shape=(), dtype=np.int64
            ),
        }
    )


def _format_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in HUMAN_OBSERVATION_KEYS if key not in observation]
    if missing:
        raise KeyError(f"CrowdNav++ observation is missing keys {missing}")
    converted = {
        "robot_node": np.asarray(observation["robot_node"], dtype=np.float32),
        "temporal_edges": np.asarray(observation["temporal_edges"], dtype=np.float32),
        "visible_masks": np.asarray(observation["visible_masks"], dtype=np.bool_),
        "spatial_edges": np.asarray(observation["spatial_edges"], dtype=np.float32),
        "detected_human_num": int(observation["detected_human_num"]),
    }
    for key, value in converted.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"CrowdNav++ observation {key!r} contains non-finite values")
    return converted


def outcome_of(info: Any) -> Optional[str]:
    """Return the wrapper outcome for a CrowdNav++ terminal info object."""
    if isinstance(info, ReachGoal):
        return "success"
    if isinstance(info, Collision):
        return "collision"
    if isinstance(info, Timeout):
        return "timeout"
    if isinstance(info, (Nothing, Danger)):
        return None
    raise TypeError(f"Unsupported CrowdNav++ info type: {type(info).__name__}")


class RewardInjectedCrowdNavEnv(gym.Env):
    """CrowdNav++ variable-human env with optional RAISE reward injection.

    CrowdNav++'s unicycle reset samples 1..human_num+human_num_range, while
    its invariant checks human_num-human_num_range..human_num+human_num_range.
    Expand too-small ranges to human_num-1 so every sampled count is valid.
    """

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
        self.env_config = load_env_config(self.settings.env_config_source)
        self.sim = make_crowd_sim(self.env_config, self.settings)
        self.reward_fn = reward_fn
        self.phase = phase
        self.render_mode = render_mode
        self._seed = seed
        self._seed_consumed = False
        self._sim_rng_state = None

        robot = self.sim.robot
        self.time_step = float(self.sim.time_step)
        self.time_limit = float(self.sim.time_limit)
        self.discomfort_dist = float(self.sim.discomfort_dist)
        self.human_num = int(self.sim.config.sim.human_num)
        self.min_human_num = int(self.sim.min_human_num)
        self.max_human_num = int(self.sim.max_human_num)
        self.v_pref = float(robot.v_pref)
        if self.settings.action_mode == "discrete":
            self.action_table = discrete_action_table(self.settings, self.v_pref)
            self.action_space = gym.spaces.Discrete(len(self.action_table))
        else:
            self.action_table = []
            if self.settings.kinematics == "holonomic":
                low = np.full(2, -self.v_pref, dtype=np.float32)
                high = np.full(2, self.v_pref, dtype=np.float32)
            else:
                low = np.asarray(
                    [UNICYCLE_SPEED_DELTA_MIN, -UNICYCLE_ROTATION_DELTA_MAX],
                    dtype=np.float32,
                )
                high = np.asarray(
                    [UNICYCLE_SPEED_DELTA_MAX, UNICYCLE_ROTATION_DELTA_MAX],
                    dtype=np.float32,
                )
            self.action_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )
        self.observation_space = _observation_space(self.sim)

        self.current_phase = phase
        self.case: Optional[int] = None
        self.last_state: Optional[CrowdNavRewardState] = None
        self._goal_distance = 0.0
        self._trail: List[tuple[float, float]] = []

    def _format_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prediction_method = str(self.sim.config.sim.predict_method).strip().lower()
        if prediction_method == "const_vel":
            observation = dict(observation)
            prediction_steps = int(self.sim.config.sim.predict_steps)
            prediction_interval = int(self.sim.pred_interval)
            robot = self.sim.robot
            visibility = np.asarray(self.sim.human_visibility, dtype=np.bool_)
            max_humans = int(self.sim.max_human_num)
            predicted_edges = np.full(
                (max_humans, 2 * (prediction_steps + 1)),
                15.0,
                dtype=np.float32,
            )
            current_positions = np.full((max_humans, 2), 15.0, dtype=np.float64)
            for index, human in enumerate(self.sim.humans):
                if not visibility[index]:
                    continue
                position = np.asarray([human.px, human.py], dtype=np.float64)
                velocity = np.asarray([human.vx, human.vy], dtype=np.float64)
                current_positions[index] = position - np.asarray([robot.px, robot.py])
                for step in range(prediction_steps + 1):
                    predicted_edges[index, 2 * step : 2 * step + 2] = (
                        position
                        + velocity * step * prediction_interval * self.sim.time_step
                        - np.asarray([robot.px, robot.py])
                    )

            if bool(getattr(self.sim.config.args, "sort_humans", False)):
                order = np.argsort(
                    np.linalg.norm(current_positions, axis=1),
                    kind="stable",
                )
                predicted_edges = predicted_edges[order]
            observation["spatial_edges"] = predicted_edges
        formatted = _format_observation(observation)
        if not self.observation_space.contains(formatted):
            raise ValueError("CrowdNav++ observation is outside the declared Gymnasium space")
        return formatted

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        opts = dict(options or {})
        phase = str(opts.get("phase", self.phase))
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        size = int(self.sim.case_size[phase])
        if size < 1:
            raise ValueError(f"CrowdNav++ {phase} case set must contain at least one case")

        case: Optional[int] = None
        if opts.get("case") is not None:
            case = int(opts["case"]) % size
        elif seed is not None:
            case = scenario_case(int(seed), phase, size)
            self._seed_consumed = True
        elif self._seed is not None and not self._seed_consumed:
            case = scenario_case(int(self._seed), phase, size)
            self._seed_consumed = True
        used = int(case) if case is not None else int(self.sim.case_counter[phase])

        with _UPSTREAM_RNG_LOCK:
            global_rng_state = np.random.get_state()
            try:
                self.sim.phase = None
                self.sim.test_case = None
                if self.settings.kinematics == "unicycle":
                    generate_robot_humans = self.sim.generate_robot_humans

                    def generate_bounded_unicycle_population(
                        reset_phase: str, human_num: Optional[int] = None
                    ) -> None:
                        generate_robot_humans(reset_phase, human_num)
                        target_num = int(
                            np.random.randint(
                                self.sim.min_human_num,
                                self.sim.max_human_num + 1,
                            )
                        )
                        current_num = len(self.sim.humans)
                        if target_num < current_num:
                            del self.sim.humans[target_num:]
                        elif target_num > current_num:
                            self.sim.generate_random_human_position(
                                target_num - current_num
                            )
                        self.sim.human_num = target_num
                        self.sim.last_human_states = np.zeros(
                            (target_num, 5), dtype=np.float64
                        )
                        for human_id, human in enumerate(self.sim.humans):
                            human.id = human_id

                    self.sim.generate_robot_humans = (
                        generate_bounded_unicycle_population
                    )
                    try:
                        self.sim.reset(phase=phase, test_case=case)
                    finally:
                        del self.sim.generate_robot_humans
                else:
                    self.sim.reset(phase=phase, test_case=case)
                self._sim_rng_state = np.random.get_state()
            finally:
                np.random.set_state(global_rng_state)

        self.sim.phase = "test" if phase == "val" else phase
        self.current_phase = phase
        self.case = used
        self.last_state = None
        robot = self.sim.robot
        self._goal_distance = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        self._trail = [(float(robot.px), float(robot.py))]
        return self._format_observation(self.sim.ob), {
            "raise_phase": phase,
            "raise_case": used,
        }

    def step(self, action: Any):
        return self.step_upstream(self.to_upstream_action(action))

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        robot = self.sim.robot
        return render_frame(
            robot_xy=(float(robot.px), float(robot.py)),
            robot_radius=float(robot.radius),
            goal_xy=(float(robot.gx), float(robot.gy)),
            humans=[
                (float(human.px), float(human.py), float(human.radius))
                for human in self.sim.humans
            ],
            global_time=float(self.sim.global_time or 0.0),
            trail=self._trail,
        )

    def to_upstream_action(self, action: Any) -> Any:
        """Convert a policy action to an upstream ``ActionXY`` or ``ActionRot``."""
        if self.settings.action_mode == "discrete":
            index = int(np.asarray(action).reshape(-1)[0])
            if not 0 <= index < len(self.action_table):
                raise ValueError(f"discrete action index {index} is out of range")
            first, second = self.action_table[index]
        else:
            first, second = continuous_to_velocity(
                action, self.settings, self.v_pref
            )
        if self.settings.kinematics == "holonomic":
            return ActionXY(float(first), float(second))
        return ActionRot(float(first), float(second))

    def _run_with_env_rng(self, operation: Any) -> Any:
        with _UPSTREAM_RNG_LOCK:
            global_rng_state = np.random.get_state()
            if self._sim_rng_state is not None:
                np.random.set_state(self._sim_rng_state)
            try:
                return operation()
            finally:
                self._sim_rng_state = np.random.get_state()
                np.random.set_state(global_rng_state)

    def _step_with_visible_unicycle_compat(self, raw_action: Any) -> Any:
        robot = self.sim.robot
        if (
            self.sim.phase != "test"
            or robot.kinematics != "unicycle"
            or not robot.visible
        ):
            return self.sim.step(raw_action)

        original_lookahead = robot.one_step_lookahead

        def lookahead(position: Any, action: Any) -> Any:
            if not isinstance(action, ActionXY):
                return original_lookahead(position, action)
            # CrowdNav++'s visible-robot prediction uses linear ActionXY lookahead,
            # while its unicycle validity check accepts only ActionRot.
            return [
                position[0] + action.vx * self.sim.time_step,
                position[1] + action.vy * self.sim.time_step,
                action.vx,
                action.vy,
            ]

        robot.one_step_lookahead = lookahead
        try:
            return self.sim.step(raw_action)
        finally:
            robot.one_step_lookahead = original_lookahead

    def _raw_action_and_executed(self, action: Any) -> tuple[np.ndarray, Any]:
        if isinstance(action, (ActionXY, ActionRot)):
            raw = np.asarray(action, dtype=np.float64)
        else:
            raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != 2 or not np.all(np.isfinite(raw)):
            raise ValueError("CrowdNav++ robot actions must contain two finite values")
        raw = raw.copy()
        executed = self.sim.robot.policy.clip_action(raw.copy(), self.v_pref)
        return raw, executed

    def step_upstream(self, upstream_action: Any):
        """One native CrowdNav++ step, used by policies and Stage I collectors."""
        watchdog_beat()
        raw_action, executed_action = self._raw_action_and_executed(upstream_action)
        dmin = step_min_separation(self.sim, executed_action)
        previous_goal_distance = self._goal_distance
        robot_before = self.sim.robot
        potential_cur = math.hypot(
            robot_before.gx - robot_before.px,
            robot_before.gy - robot_before.py,
        )
        potential_delta = -abs(potential_cur) - float(self.sim.potential)
        reward_action = (float(executed_action[0]), float(executed_action[1]))
        if robot_before.kinematics == "unicycle":
            reward_action = (
                float(
                    np.clip(
                        self.sim.desiredVelocity[0] + executed_action[0],
                        -self.v_pref,
                        self.v_pref,
                    )
                ),
                float(executed_action[1]),
            )
        observation, native_reward, upstream_done, wrapped_info = self._run_with_env_rng(
            lambda: self._step_with_visible_unicycle_compat(raw_action)
        )
        if not isinstance(wrapped_info, dict) or "info" not in wrapped_info:
            raise TypeError("CrowdNav++ step info must contain the native 'info' object")
        native_info = wrapped_info["info"]
        outcome = outcome_of(native_info)
        collision = isinstance(native_info, Collision)
        reached_goal = isinstance(native_info, ReachGoal)
        timeout = isinstance(native_info, Timeout)
        danger = isinstance(native_info, Danger)
        if bool(upstream_done) != bool(collision or reached_goal or timeout):
            raise RuntimeError("CrowdNav++ done flag disagrees with its terminal info type")

        robot = self.sim.robot
        visibility = self.sim.human_visibility
        if len(visibility) != len(self.sim.humans):
            raise RuntimeError(
                "CrowdNav++ visibility flags do not match the current human population"
            )
        goal_distance = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        self._goal_distance = goal_distance
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
                kinematics=str(robot.kinematics),
            ),
            humans=tuple(
                HumanState(
                    px=float(human.px),
                    py=float(human.py),
                    vx=float(human.vx),
                    vy=float(human.vy),
                    radius=float(human.radius),
                    visible=bool(visibility[index]),
                )
                for index, human in enumerate(self.sim.humans)
            ),
            action=reward_action,
            collision=collision,
            reached_goal=reached_goal,
            timeout=timeout,
            min_separation=float(dmin),
            discomfort_dist=self.discomfort_dist,
            goal_distance=float(goal_distance),
            progress=float(previous_goal_distance - goal_distance),
            time_step=self.time_step,
            global_time=float(self.sim.global_time),
            time_limit=self.time_limit,
            danger=danger,
            success_reward=float(self.sim.success_reward),
            collision_penalty=float(self.sim.collision_penalty),
            discomfort_penalty_factor=float(self.sim.discomfort_penalty_factor),
            potential_delta=float(potential_delta),
            potential_factor=3.0 if robot.kinematics == "unicycle" else 2.0,
        )
        self.last_state = state

        components: Dict[str, float] = {}
        if self.reward_fn is None:
            reward = float(native_reward)
        else:
            reward = float(self.reward_fn.compute(state))
            getter = getattr(self.reward_fn, "last_reward_components", None)
            if callable(getter):
                raw_components = getter()
                if isinstance(raw_components, dict):
                    components = {
                        str(key): float(value)
                        for key, value in raw_components.items()
                    }

        danger_min_dist = float(native_info.min_dist) if danger else None
        info = {
            "raise_outcome": outcome,
            "raise_collision": collision,
            "raise_reached_goal": reached_goal,
            "raise_timeout": timeout,
            "raise_danger": danger,
            "raise_danger_min_dist": danger_min_dist,
            "raise_min_separation": float(dmin),
            "raise_native_reward": float(native_reward),
            "raise_progress": float(state.progress),
            "raise_speed": math.hypot(state.robot.vx, state.robot.vy),
            "raise_goal_distance": float(goal_distance),
            "raise_global_time": float(self.sim.global_time),
            "raise_phase": self.current_phase,
            "raise_case": self.case,
            "raise_reward_components": components,
        }
        return (
            self._format_observation(observation),
            reward,
            bool(collision or reached_goal),
            bool(timeout),
            info,
        )


__all__ = [
    "HUMAN_OBSERVATION_KEYS",
    "PHASES",
    "ExternalRobotPolicy",
    "RewardInjectedCrowdNavEnv",
    "eval_cases",
    "load_env_config",
    "make_crowd_sim",
    "outcome_of",
    "scenario_case",
    "step_min_separation",
]
