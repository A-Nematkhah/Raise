"""Gymnasium wrapper: highway-fast-v0 + sandboxed HighwayRewardState rewards."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-fast-v0)
import numpy as np

from domains.highway.action_config import (
    META_DEFAULT_SPEEDS_MPS as TARGET_SPEEDS_MPS,
    resolve_action_dict,
)
from domains.highway.state import (
    DEFAULT_LANES,
    LANE_WIDTH_M,
    EgoVehicle,
    HighwayRewardState,
    NearbyVehicle,
    lane_index_from_y,
)
from domains.highway.watchdog import beat as watchdog_beat

ENV_ID = "highway-fast-v0"
DEFAULT_DURATION = 40  # seconds
DEFAULT_POLICY_FREQ = 5
DEFAULT_VEHICLES = 20
MAX_OTHERS = 5
# Denser traffic + distinct seed band for selection / anti-hacking eval.
# highway-env spawns each vehicle ahead of the front-most one with a gap scaled
# by 1/vehicles_density, so density (not vehicles_count) controls local traffic.
HOLDOUT_VEHICLES = 20
HOLDOUT_VEHICLES_DENSITY = 1.5
HOLDOUT_SEED_OFFSET = 50_003

# Policy observation scales (ego-centric, roughly unit range).
OBS_X_SCALE = 100.0  # m; half the 200 m perception range
OBS_Y_SCALE = LANE_WIDTH_M * (DEFAULT_LANES - 1)  # m; outer lane centres
OBS_V_SCALE = 40.0  # m/s; Vehicle.MAX_SPEED
OBS_DV_SCALE = 20.0  # m/s; relative speeds


def _base_env_config(action: Dict[str, Any]) -> Dict[str, Any]:
    # absolute=True: ego x is world frame so progress = Δx is well-defined.
    # RewardInjectedHighwayEnv hands PPO policy_observation(obs), not this raw obs.
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": MAX_OTHERS + 1,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "absolute": True,
            "normalize": False,
            "see_behind": True,
        },
        "action": dict(action),
        "lanes_count": DEFAULT_LANES,
        "vehicles_count": DEFAULT_VEHICLES,
        "duration": DEFAULT_DURATION,
        "policy_frequency": DEFAULT_POLICY_FREQ,
        "simulation_frequency": 15,
        "collision_reward": 0.0,
        "high_speed_reward": 0.0,
        "right_lane_reward": 0.0,
        "lane_change_reward": 0.0,
        "reward_speed_range": [20, 30],
        "normalize_reward": False,
        "offroad_terminal": True,
    }


def default_env_config() -> Dict[str, Any]:
    """Always ``meta_default`` eleven gears — Score1 collectors / legacy tests."""
    return _base_env_config(
        {
            "type": "DiscreteMetaAction",
            "target_speeds": list(TARGET_SPEEDS_MPS),
        }
    )


def training_env_config() -> Dict[str, Any]:
    """Env config for PPO train/eval using the active ``action_mode`` settings."""
    return _base_env_config(resolve_action_dict())


def holdout_env_config() -> Dict[str, Any]:
    """
    Harder eval distribution (not used for PPO rollouts).

    Denser traffic so a constant-speed lane-keep cruise is more likely to
    collide — selection metrics should come from this profile.
    Uses the active action_mode so train/eval action spaces match.
    """
    cfg = training_env_config()
    cfg["vehicles_count"] = int(HOLDOUT_VEHICLES)
    cfg["vehicles_density"] = float(HOLDOUT_VEHICLES_DENSITY)
    cfg["lanes_count"] = int(DEFAULT_LANES)
    return cfg


def make_base_env(
    *,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Create a configured highway-fast-v0 env (native reward unused)."""
    cfg = default_env_config()
    if config:
        nested = dict(cfg)
        for key, value in config.items():
            # Always replace the whole action dict so ContinuousAction does not
            # inherit leftover DiscreteMetaAction target_speeds from defaults.
            if key == "action" and isinstance(value, dict):
                nested[key] = dict(value)
            elif isinstance(value, dict) and isinstance(nested.get(key), dict):
                merged = dict(nested[key])
                merged.update(value)
                nested[key] = merged
            else:
                nested[key] = value
        cfg = nested
    env = gym.make(ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(cfg)
    env.reset(seed=int(seed) if seed is not None else None)
    return env


def _row_presence(row: np.ndarray) -> bool:
    return float(row[0]) > 0.5


def detect_off_road(env: Any, info: Dict[str, Any]) -> bool:
    """Off-road flag for one step: info keys, on-road reward, then ``vehicle.on_road``."""
    off_road = bool(info.get("off_road", False))
    rewards = info.get("rewards")
    if isinstance(rewards, dict) and rewards.get("on_road_reward", 1.0) == 0.0:
        off_road = True
    # highway-env often omits info['off_road']; trust the vehicle flag.
    try:
        veh = getattr(getattr(env, "unwrapped", env), "vehicle", None)
        if veh is not None and hasattr(veh, "on_road") and not bool(veh.on_road):
            off_road = True
    except Exception:  # noqa: BLE001
        pass
    return off_road


def kinematics_to_state(
    obs: Any,
    *,
    action: Any,
    collision: bool,
    off_road: bool,
    timeout: bool,
    time_step: float,
    global_time: float,
    time_limit: float,
    prev_ego_x: Optional[float],
) -> Tuple[HighwayRewardState, float]:
    """
    Map absolute Kinematics observation to HighwayRewardState.

    ``progress`` is forward Δx since the previous frame (clamped at 0).
    Nearby vehicles are stored ego-relative (x/y minus ego).
    """
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected kinematics matrix Nx>=6, got {arr.shape}")

    ego_row = arr[0]
    ego_x = float(ego_row[1]) if _row_presence(ego_row) else float(prev_ego_x or 0.0)
    ego_y = float(ego_row[2]) if _row_presence(ego_row) else 0.0
    vx = float(ego_row[3]) if _row_presence(ego_row) else 0.0
    vy = float(ego_row[4]) if _row_presence(ego_row) else 0.0
    heading = float(ego_row[5]) if _row_presence(ego_row) else 0.0
    speed = float((vx ** 2 + vy ** 2) ** 0.5)
    progress = 0.0 if prev_ego_x is None else max(0.0, ego_x - float(prev_ego_x))

    others = []
    for row in arr[1 : MAX_OTHERS + 1]:
        if not _row_presence(row):
            continue
        others.append(
            NearbyVehicle(
                x=float(row[1]) - ego_x,
                y=float(row[2]) - ego_y,
                # Ego-relative velocities (spec: relative kinematics).
                vx=float(row[3]) - vx,
                vy=float(row[4]) - vy,
                heading=float(row[5]),
            )
        )

    lane_index = float(lane_index_from_y(ego_y))
    on_road = not bool(off_road)

    state = HighwayRewardState(
        ego=EgoVehicle(
            x=ego_x,
            y=ego_y,
            vx=vx,
            vy=vy,
            heading=heading,
            speed=speed,
            lane_index=lane_index,
            on_road=on_road,
        ),
        others=tuple(others),
        collision=bool(collision),
        off_road=bool(off_road),
        timeout=bool(timeout),
        action=action,
        time_step=float(time_step),
        global_time=float(global_time),
        time_limit=float(time_limit),
        progress=float(progress),
        speed=speed,
    )
    return state, ego_x


def policy_observation(obs: Any) -> np.ndarray:
    """
    Ego-centric, scaled view of the absolute Kinematics obs for the PPO policy.

    Row 0 (ego): ``[1, 0, y/Y, vx/V, vy/V, heading]`` — lateral position stays
    absolute so the policy knows its lane; world x is dropped (it only grows).
    Rows 1..: ``[1, dx/X, dy/Y, dvx/DV, dvy/DV, heading]`` relative to ego.
    Absent vehicles stay all-zero. Shape and dtype match the env obs space.
    """
    arr = np.asarray(obs, dtype=np.float32)
    out = np.zeros_like(arr)
    if arr.ndim != 2 or arr.shape[0] == 0 or not _row_presence(arr[0]):
        return out
    ego = arr[0]
    out[0, 0] = 1.0
    out[0, 2] = ego[2] / OBS_Y_SCALE
    out[0, 3] = ego[3] / OBS_V_SCALE
    out[0, 4] = ego[4] / OBS_V_SCALE
    out[0, 5] = ego[5]
    present = arr[1:, 0] > 0.5
    rows = arr[1:][present]
    if rows.size:
        rel = np.zeros_like(rows)
        rel[:, 0] = 1.0
        rel[:, 1] = (rows[:, 1] - ego[1]) / OBS_X_SCALE
        rel[:, 2] = (rows[:, 2] - ego[2]) / OBS_Y_SCALE
        rel[:, 3] = (rows[:, 3] - ego[3]) / OBS_DV_SCALE
        rel[:, 4] = (rows[:, 4] - ego[4]) / OBS_DV_SCALE
        rel[:, 5] = rows[:, 5]
        out[1:][present] = rel
    return out


def nearest_gap(state: HighwayRewardState) -> Optional[float]:
    """Euclidean distance (m) to the closest observed vehicle, or None."""
    best: Optional[float] = None
    for v in state.others:
        dist = float((v.x * v.x + v.y * v.y) ** 0.5)
        if best is None or dist < best:
            best = dist
    return best


def action_for_state(action: Any, action_space: Any) -> Any:
    """Plain Python ``state.action``: ``int`` for Discrete, tuple of floats for Box.

    SB3 hands the env ``np.int64`` (VecEnv) or 0-d arrays (``predict``); the
    sandbox only accepts real ``int``/``float`` returns, and smoke states use
    the same plain types.
    """
    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    return tuple(float(a) for a in np.asarray(action, dtype=np.float64).reshape(-1))


class RewardInjectedHighwayEnv(gym.Wrapper):
    """
    Gymnasium Wrapper that replaces the native reward with
    ``reward_fn.compute(HighwayRewardState)``.
    """

    def __init__(
        self,
        reward_fn: Any,
        *,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        env = make_base_env(seed=seed, config=config, render_mode=render_mode)
        super().__init__(env)
        self.reward_fn = reward_fn
        self._seed = seed
        self._seed_consumed = False
        self._ego_x: Optional[float] = None
        self._global_time = 0.0
        merged = default_env_config()
        if config:
            merged.update(config)
        self._time_step = float(
            1.0 / float(merged.get("policy_frequency", DEFAULT_POLICY_FREQ))
        )
        self._time_limit = float(merged.get("duration", DEFAULT_DURATION))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        self._ego_x = None
        self._global_time = 0.0
        kwargs: Dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
            self._seed_consumed = True
        elif self._seed is not None and not self._seed_consumed:
            # Apply construction seed ONCE so training episodes advance RNG.
            # Re-passing the same seed every reset made all episodes identical
            # → SR/CR only {0,1} under deterministic eval.
            kwargs["seed"] = int(self._seed)
            self._seed_consumed = True
        if options is not None:
            kwargs["options"] = options
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0 and _row_presence(arr[0]):
            self._ego_x = float(arr[0][1])
            info["raise_lane_index"] = int(lane_index_from_y(float(arr[0][2])))
        else:
            self._ego_x = 0.0
        return policy_observation(obs), info

    def step(self, action):
        watchdog_beat()
        obs, _native_r, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        collision = bool(info.get("crashed", False))
        off_road = detect_off_road(self.env, info)
        timeout = bool(truncated) and not collision and not off_road

        self._global_time = min(self._time_limit, self._global_time + self._time_step)
        state, self._ego_x = kinematics_to_state(
            obs,
            action=action_for_state(action, self.env.action_space),
            collision=collision,
            off_road=off_road,
            timeout=timeout,
            time_step=self._time_step,
            global_time=self._global_time,
            time_limit=self._time_limit,
            prev_ego_x=self._ego_x,
        )
        reward = float(self.reward_fn.compute(state))
        # Last-step components only (no accumulation here — RolloutDiagnosticsCallback
        # owns rollout-end aggregation for EUREKA reflection).
        comps: Dict[str, float] = {}
        getter = getattr(self.reward_fn, "last_reward_components", None)
        if callable(getter):
            raw = getter()
            if isinstance(raw, dict):
                comps = {str(k): float(v) for k, v in raw.items()}
        info["raise_collision"] = collision
        info["raise_off_road"] = off_road
        info["raise_timeout"] = timeout
        info["raise_progress"] = float(state.progress)
        info["raise_speed"] = float(state.speed)
        info["raise_ego_x"] = float(self._ego_x if self._ego_x is not None else 0.0)
        info["raise_lane_index"] = int(state.ego.lane_index)
        info["raise_nearest_gap"] = nearest_gap(state)
        info["raise_reward_components"] = comps
        return policy_observation(obs), reward, bool(terminated), bool(truncated), info
