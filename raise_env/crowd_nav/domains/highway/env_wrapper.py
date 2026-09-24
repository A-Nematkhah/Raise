"""Gymnasium wrapper: highway-fast-v0 + sandboxed HighwayRewardState rewards."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from crowd_nav.domains.highway.state import (
    EgoVehicle,
    HighwayRewardState,
    NearbyVehicle,
)

ENV_ID = "highway-fast-v0"
DEFAULT_DURATION = 40  # seconds
DEFAULT_POLICY_FREQ = 5
DEFAULT_TIME_STEP = 1.0 / DEFAULT_POLICY_FREQ
DEFAULT_VEHICLES = 20
DEFAULT_LANES = 4
MAX_OTHERS = 5


def _require_highway_deps() -> None:
    try:
        import gymnasium  # noqa: F401
        import highway_env  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Highway domain requires gymnasium + highway-env. "
            "Install: pip install -r requirements_highway.txt"
        ) from exc


def default_env_config() -> Dict[str, Any]:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": MAX_OTHERS + 1,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "absolute": False,
            "normalize": False,
            "see_behind": True,
        },
        "action": {"type": "DiscreteMetaAction"},
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


def make_base_env(*, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
    """Create a configured highway-fast-v0 env (native reward unused)."""
    _require_highway_deps()
    import gymnasium as gym

    cfg = default_env_config()
    if config:
        cfg.update(config)
    env = gym.make(ENV_ID, render_mode=None, config=cfg)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _row_presence(row: np.ndarray) -> bool:
    return float(row[0]) > 0.5


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
    Map Kinematics observation matrix to HighwayRewardState.

    Returns ``(state, ego_x_absolute_proxy)`` where the second value is the
    ego x used for next-step progress (relative obs → cumulative).
    """
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected kinematics matrix Nx>=6, got {arr.shape}")

    ego_row = arr[0]
    # Relative kinematics: ego is at origin; track cumulative x via caller.
    rel_x = float(ego_row[1]) if _row_presence(ego_row) else 0.0
    ego_x = float(prev_ego_x or 0.0) + rel_x
    # After first frame with relative obs, ego row x is usually ~0; progress
    # comes from integrating vehicle speed * dt instead when rel_x≈0.
    vx = float(ego_row[3]) if _row_presence(ego_row) else 0.0
    vy = float(ego_row[4]) if _row_presence(ego_row) else 0.0
    heading = float(ego_row[5]) if _row_presence(ego_row) else 0.0
    speed = float((vx ** 2 + vy ** 2) ** 0.5)
    progress = speed * float(time_step)
    if prev_ego_x is not None and abs(rel_x) > 1e-6:
        progress = max(progress, rel_x)

    others = []
    for row in arr[1 : MAX_OTHERS + 1]:
        if not _row_presence(row):
            continue
        others.append(
            NearbyVehicle(
                x=float(row[1]),
                y=float(row[2]),
                vx=float(row[3]),
                vy=float(row[4]),
                heading=float(row[5]),
            )
        )

    # Lane index not in default features — use y banding as a soft proxy.
    lane_index = float(max(0, min(DEFAULT_LANES - 1, int(round(float(ego_row[2]) / 4.0 + 1.5)))))
    on_road = not bool(off_road)

    state = HighwayRewardState(
        ego=EgoVehicle(
            x=ego_x,
            y=float(ego_row[2]) if _row_presence(ego_row) else 0.0,
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


class RewardInjectedHighwayEnv:
    """
    Thin adapter around highway-fast-v0 that replaces the native reward with
    ``reward_fn.compute(HighwayRewardState)``.
    """

    def __init__(
        self,
        reward_fn: Any,
        *,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reward_fn = reward_fn
        self._seed = seed
        self._config = config
        self.env = make_base_env(seed=seed, config=config)
        self._ego_x = 0.0
        self._global_time = 0.0
        self._time_step = float(
            1.0 / float((config or default_env_config()).get("policy_frequency", DEFAULT_POLICY_FREQ))
        )
        self._time_limit = float(
            (config or default_env_config()).get("duration", DEFAULT_DURATION)
        )
        self._last_info: Dict[str, Any] = {}

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        self._ego_x = 0.0
        self._global_time = 0.0
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        elif self._seed is not None:
            kwargs["seed"] = int(self._seed)
        if options is not None:
            kwargs["options"] = options
        obs, info = self.env.reset(**kwargs)
        self._last_info = dict(info or {})
        return obs, info

    def step(self, action):
        obs, _native_r, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        collision = bool(info.get("crashed", False) or terminated and info.get("crashed", terminated))
        # highway-env sets crashed on collision; off-road may truncate.
        off_road = bool(info.get("rewards", {}).get("on_road_reward", 1.0) == 0.0) if isinstance(
            info.get("rewards"), dict
        ) else bool(info.get("off_road", False))
        if truncated and not collision:
            # Duration end vs off-road: prefer explicit flags.
            off_road = off_road or bool(info.get("off_road", False))
        timeout = bool(truncated) and not collision and not off_road
        if terminated and collision:
            timeout = False

        self._global_time = min(self._time_limit, self._global_time + self._time_step)
        state, self._ego_x = kinematics_to_state(
            obs,
            action=action,
            collision=collision,
            off_road=off_road,
            timeout=timeout,
            time_step=self._time_step,
            global_time=self._global_time,
            time_limit=self._time_limit,
            prev_ego_x=self._ego_x,
        )
        reward = float(self.reward_fn.compute(state))
        info["raise_collision"] = collision
        info["raise_off_road"] = off_road
        info["raise_timeout"] = timeout
        info["raise_progress"] = float(state.progress)
        info["raise_speed"] = float(state.speed)
        info["raise_ego_x"] = float(self._ego_x)
        self._last_info = info
        return obs, reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

    def render(self):
        return self.env.render()
