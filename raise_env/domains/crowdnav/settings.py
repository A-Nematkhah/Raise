"""
Process-wide CrowdNav settings: robot kinematics, policy action space, env config.

The prompts, smoke states and Score1 dataset must describe the same
``state.action`` as the env the policy trains in, so the run configures one
:class:`CrowdNavSettings` (``configure_from_config``) before any of them is
built. Stage II/III trainers re-apply it from their config.

Discrete actions are the CADRL/SARL action set of upstream
``crowd_nav/policy/cadrl.py::build_action_space`` with the defaults of
``crowd_nav/configs/policy.config`` ``[action_space]``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple

import numpy as np

KINEMATICS = ("holonomic", "unicycle")
ACTION_MODES = ("discrete", "continuous")

# crowd_nav/configs/policy.config [action_space].
DEFAULT_SPEED_SAMPLES = 5
DEFAULT_ROTATION_SAMPLES = 16
# Unicycle rotations span [-pi/4, pi/4] (CADRL.build_action_space).
UNICYCLE_MAX_ROTATION = math.pi / 4.0

_VENDOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "crowdnav")
)
DEFAULT_ENV_CONFIG_PATH = os.path.join(_VENDOR_DIR, "configs", "env.config")
DEFAULT_POLICY_CONFIG_PATH = os.path.join(_VENDOR_DIR, "configs", "policy.config")


@dataclass(frozen=True)
class CrowdNavSettings:
    kinematics: str = "holonomic"
    action_mode: str = "discrete"
    speed_samples: int = DEFAULT_SPEED_SAMPLES
    rotation_samples: int = DEFAULT_ROTATION_SAMPLES
    # None -> upstream crowd_nav/configs/env.config.
    env_config_path: Optional[str] = None

    @property
    def env_config(self) -> str:
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
    path = settings.env_config_path
    if path is not None and not os.path.isfile(str(path)):
        raise FileNotFoundError(f"CrowdNav env config not found: {path}")
    return replace(
        settings,
        kinematics=kin,
        action_mode=mode,
        speed_samples=int(settings.speed_samples),
        rotation_samples=int(settings.rotation_samples),
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
    return validate_settings(
        CrowdNavSettings(
            kinematics=str(getattr(config, "robot_kinematics", None) or default.kinematics),
            action_mode=str(getattr(config, "action_mode", None) or default.action_mode),
            speed_samples=int(getattr(config, "action_speed_samples", None) or default.speed_samples),
            rotation_samples=int(
                getattr(config, "action_rotation_samples", None) or default.rotation_samples
            ),
            env_config_path=str(path) if path else None,
        )
    )


def configure_from_config(config: Any) -> CrowdNavSettings:
    """Apply ``config``'s CrowdNav fields process-wide (DomainPack ``configure`` hook)."""
    return set_settings(settings_from_config(config))


# ------------------------------------------------------------------ action space


def discrete_action_table(
    settings: CrowdNavSettings, v_pref: float
) -> List[Tuple[float, float]]:
    """
    CADRL/SARL action set: stop + rotation x exponentially spaced speed.

    Holonomic entries are ``(vx, vy)`` in the world frame, unicycle entries
    ``(v, r)`` (speed, heading change), exactly as ``build_action_space``.
    """
    n_speed = int(settings.speed_samples)
    speeds = [
        (np.exp((i + 1) / n_speed) - 1) / (np.e - 1) * float(v_pref) for i in range(n_speed)
    ]
    holonomic = settings.kinematics == "holonomic"
    if holonomic:
        rotations = np.linspace(0, 2 * np.pi, settings.rotation_samples, endpoint=False)
    else:
        rotations = np.linspace(
            -UNICYCLE_MAX_ROTATION, UNICYCLE_MAX_ROTATION, settings.rotation_samples
        )
    table: List[Tuple[float, float]] = [(0.0, 0.0)]
    for rotation in rotations:
        for speed in speeds:
            if holonomic:
                table.append((float(speed * np.cos(rotation)), float(speed * np.sin(rotation))))
            else:
                table.append((float(speed), float(rotation)))
    return table


def n_discrete_actions(settings: CrowdNavSettings) -> int:
    return 1 + int(settings.speed_samples) * int(settings.rotation_samples)


def continuous_to_velocity(
    action: Any, settings: CrowdNavSettings, v_pref: float
) -> Tuple[float, float]:
    """
    Map a ``Box([-1, 1]^2)`` policy output to an executable robot command.

    Holonomic: ``(vx, vy) = a * v_pref`` with the speed capped at ``v_pref``.
    Unicycle: ``v = (a0 + 1) / 2 * v_pref`` and ``r = a1 * pi/4`` (the
    discrete unicycle set spans the same ranges).
    """
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    if settings.kinematics == "holonomic":
        vx, vy = float(a[0]) * float(v_pref), float(a[1]) * float(v_pref)
        speed = math.hypot(vx, vy)
        if speed > float(v_pref) > 0.0:
            scale = float(v_pref) / speed
            vx, vy = vx * scale, vy * scale
        return vx, vy
    return (float(a[0]) + 1.0) * 0.5 * float(v_pref), float(a[1]) * UNICYCLE_MAX_ROTATION


def spaces_compatible(a: Any, b: Any) -> bool:
    """Same space type and shape / size (warm-start guard)."""
    if type(a) is not type(b):
        return False
    if hasattr(a, "n") and hasattr(b, "n"):
        return int(a.n) == int(b.n)
    return tuple(getattr(a, "shape", ()) or ()) == tuple(getattr(b, "shape", ()) or ())


def action_space_prompt_block(settings: Optional[CrowdNavSettings] = None) -> str:
    """What ``state.action`` holds for the active kinematics (LLM prompt block)."""
    s = settings or get_settings()
    if s.kinematics == "holonomic":
        meaning = (
            "state.action is the executed robot velocity command (vx, vy) in m/s, "
            "world frame: a tuple of two floats; read state.action[0] / state.action[1]. "
            "Its norm is at most state.robot.v_pref; (0.0, 0.0) means stop."
        )
    else:
        meaning = (
            "state.action is the executed unicycle command (v, r): a tuple of two floats, "
            "v = speed in m/s (0..state.robot.v_pref) and r = heading change in radians "
            f"applied this step (|r| <= {UNICYCLE_MAX_ROTATION:.3f}); read state.action[0] / state.action[1]."
        )
    space = (
        f"{n_discrete_actions(s)} discrete CADRL actions (stop + "
        f"{s.rotation_samples} directions x {s.speed_samples} speeds)"
        if s.action_mode == "discrete"
        else "a continuous 2-D command"
    )
    return (
        f"Robot action space ({s.kinematics} kinematics): the policy chooses {space}. "
        + meaning
    )


__all__ = [
    "ACTION_MODES",
    "CrowdNavSettings",
    "DEFAULT_ENV_CONFIG_PATH",
    "DEFAULT_POLICY_CONFIG_PATH",
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
