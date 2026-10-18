"""CrowdNav++ settings shared by prompts, data tooling, and the env wrapper."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

KINEMATICS = ("holonomic", "unicycle")
ACTION_MODES = ("discrete", "continuous")
UNICYCLE_SPEED_DELTA_MIN = -0.1
UNICYCLE_SPEED_DELTA_MAX = 0.087
UNICYCLE_ROTATION_DELTA_MAX = 0.06

_VENDOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "crowdnavpp")
)
DEFAULT_ENV_CONFIG_PATH = os.path.join(
    _VENDOR_DIR, "crowd_nav", "configs", "config.py"
)


@dataclass(frozen=True)
class CrowdNavSettings:
    kinematics: str = "unicycle"
    action_mode: str = "continuous"
    speed_samples: int = 5
    rotation_samples: int = 16
    env_config_path: Optional[str] = None
    env_config_obj: Any = field(default=None, repr=False, compare=False)
    human_num_override: Optional[int] = None

    @property
    def env_config_source(self) -> Any:
        """Python config object or file, defaulting to upstream ``Config``."""
        if self.env_config_obj is not None:
            return self.env_config_obj
        return self.env_config_path or DEFAULT_ENV_CONFIG_PATH


def validate_settings(settings: CrowdNavSettings) -> CrowdNavSettings:
    kin = str(settings.kinematics).strip().lower()
    mode = str(settings.action_mode).strip().lower()
    if kin not in KINEMATICS:
        raise ValueError(f"robot kinematics must be one of {KINEMATICS}, got {settings.kinematics!r}")
    if mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {settings.action_mode!r}")
    if int(settings.speed_samples) < 1 or int(settings.rotation_samples) < 1:
        raise ValueError("speed_samples and rotation_samples must be >= 1")
    if settings.env_config_path is not None:
        if settings.env_config_obj is not None:
            raise ValueError("set only one of env_config_path and env_config_obj")
        if not os.path.isfile(str(settings.env_config_path)):
            raise FileNotFoundError(
                f"CrowdNav++ Python config not found: {settings.env_config_path}"
            )
    if settings.human_num_override is not None and int(settings.human_num_override) < 1:
        raise ValueError("human_num_override must be >= 1")
    return replace(
        settings,
        kinematics=kin,
        action_mode=mode,
        speed_samples=int(settings.speed_samples),
        rotation_samples=int(settings.rotation_samples),
        human_num_override=(
            None
            if settings.human_num_override is None
            else int(settings.human_num_override)
        ),
    )


_SETTINGS = CrowdNavSettings()


def get_settings() -> CrowdNavSettings:
    return _SETTINGS


def set_settings(settings: CrowdNavSettings) -> CrowdNavSettings:
    global _SETTINGS
    _SETTINGS = validate_settings(settings)
    return _SETTINGS


def settings_from_config(config: Any) -> CrowdNavSettings:
    """Settings from a run / stage config (missing fields keep the defaults)."""
    default = CrowdNavSettings()
    path = getattr(config, "env_config_path", None)
    config_obj = getattr(config, "env_config_obj", None)
    return validate_settings(
        CrowdNavSettings(
            kinematics=str(getattr(config, "robot_kinematics", None) or default.kinematics),
            action_mode=str(getattr(config, "action_mode", None) or default.action_mode),
            speed_samples=int(getattr(config, "action_speed_samples", None) or default.speed_samples),
            rotation_samples=int(
                getattr(config, "action_rotation_samples", None) or default.rotation_samples
            ),
            env_config_path=str(path) if path else None,
            env_config_obj=config_obj,
            human_num_override=getattr(config, "human_num_override", None),
        )
    )


def configure_from_config(config: Any) -> CrowdNavSettings:
    """Apply ``config``'s CrowdNav fields process-wide (DomainPack ``configure`` hook)."""
    return set_settings(settings_from_config(config))


def discrete_action_table(
    settings: CrowdNavSettings, v_pref: float
) -> List[Tuple[float, float]]:
    """Optional wrapper-level discrete actions translated to the upstream controls."""
    n_speed = int(settings.speed_samples)
    speeds = [
        (np.exp((i + 1) / n_speed) - 1) / (np.e - 1) * float(v_pref)
        for i in range(n_speed)
    ]
    table: List[Tuple[float, float]] = [(0.0, 0.0)]
    if settings.kinematics == "holonomic":
        rotations = np.linspace(0, 2 * np.pi, settings.rotation_samples, endpoint=False)
        for rotation in rotations:
            for speed in speeds:
                table.append(
                    (float(speed * np.cos(rotation)), float(speed * np.sin(rotation)))
                )
    else:
        speed_deltas = np.linspace(
            UNICYCLE_SPEED_DELTA_MIN,
            UNICYCLE_SPEED_DELTA_MAX,
            n_speed,
        )
        rotations = np.linspace(
            -UNICYCLE_ROTATION_DELTA_MAX,
            UNICYCLE_ROTATION_DELTA_MAX,
            settings.rotation_samples,
        )
        for rotation in rotations:
            for speed_delta in speed_deltas:
                table.append((float(speed_delta), float(rotation)))
    return table


def n_discrete_actions(settings: CrowdNavSettings) -> int:
    return 1 + int(settings.speed_samples) * int(settings.rotation_samples)


def continuous_to_velocity(
    action: Any, settings: CrowdNavSettings, v_pref: float
) -> Tuple[float, float]:
    """Clip continuous wrapper actions to the control limits used by CrowdNav++."""
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError("continuous CrowdNav++ actions must contain two finite values")
    if settings.kinematics == "holonomic":
        vx, vy = float(values[0]), float(values[1])
        speed = math.hypot(vx, vy)
        if speed > float(v_pref) > 0.0:
            scale = float(v_pref) / speed
            vx, vy = vx * scale, vy * scale
        return vx, vy
    return (
        float(np.clip(values[0], UNICYCLE_SPEED_DELTA_MIN, UNICYCLE_SPEED_DELTA_MAX)),
        float(np.clip(values[1], -UNICYCLE_ROTATION_DELTA_MAX, UNICYCLE_ROTATION_DELTA_MAX)),
    )


def spaces_compatible(a: Any, b: Any) -> bool:
    """Same space type and shape / size (warm-start guard)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, spaces.Dict):
        return a.spaces.keys() == b.spaces.keys() and all(
            spaces_compatible(a.spaces[key], b.spaces[key]) for key in a.spaces
        )
    if hasattr(a, "n") and hasattr(b, "n"):
        return int(a.n) == int(b.n)
    return tuple(getattr(a, "shape", ()) or ()) == tuple(getattr(b, "shape", ()) or ())


def action_space_prompt_block(settings: Optional[CrowdNavSettings] = None) -> str:
    """Describe the policy and executed action formats used by the wrapper."""
    s = settings or get_settings()
    if s.kinematics == "holonomic":
        meaning = (
            "state.action is the executed robot velocity command (vx, vy) in m/s "
            "in world coordinates. Its norm is at most state.robot.v_pref; "
            "(0.0, 0.0) means stop."
        )
    else:
        meaning = (
            "state.action is the executed unicycle command (v, r): v is the "
            "resulting speed in m/s and r is the heading change in radians. "
            "The policy action changes speed by [-0.1, 0.087] m/s and heading "
            "by [-0.06, 0.06] radians per step."
        )
    policy_space = (
        f"{n_discrete_actions(s)} wrapper-level discrete actions (stop + "
        f"{s.rotation_samples} directions x {s.speed_samples} speeds)"
        if s.action_mode == "discrete"
        else "a continuous 2-D Box; the simulator clips it to its physical limits"
    )
    return (
        f"Robot action space ({s.kinematics} kinematics): the policy chooses "
        f"{policy_space}. {meaning}"
    )


__all__ = [
    "ACTION_MODES",
    "CrowdNavSettings",
    "DEFAULT_ENV_CONFIG_PATH",
    "KINEMATICS",
    "action_space_prompt_block",
    "configure_from_config",
    "continuous_to_velocity",
    "discrete_action_table",
    "get_settings",
    "n_discrete_actions",
    "set_settings",
    "settings_from_config",
    "spaces_compatible",
    "validate_settings",
]
