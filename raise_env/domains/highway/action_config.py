"""Highway action-space modes (DiscreteMetaAction vs ContinuousAction).

Process-wide settings are set from pipeline/CLI config. Default is
``meta_default`` (eleven gears 20..30) for denser DiscreteMetaAction steps.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from raise_core.sandbox.errors import RewardSandboxError

logger = logging.getLogger(__name__)

ACTION_MODES = ("meta_default", "meta_fine", "continuous")

# meta_default gears: 11 evenly spaced targets in [20, 30] m/s.
META_DEFAULT_SPEEDS_MPS: Tuple[float, ...] = tuple(
    float(x) for x in np.linspace(20.0, 30.0, 11)
)

_DEFAULT_ACCEL_RANGE: Tuple[float, float] = (-5.0, 5.0)
_DEFAULT_STEER_RANGE: Tuple[float, float] = (-math.pi / 4.0, math.pi / 4.0)

_TARGET_NAME_RE = re.compile(
    r"(target_speed|speed_target|traffic_speed|v_target|cruise_speed)",
    re.IGNORECASE,
)


@dataclass
class ActionModeSettings:
    mode: str = "meta_default"
    continuous_lateral: bool = True
    meta_fine_low: float = 15.0
    meta_fine_high: float = 35.0
    meta_fine_n: int = 21
    acceleration_range: Tuple[float, float] = _DEFAULT_ACCEL_RANGE
    steering_range: Tuple[float, float] = _DEFAULT_STEER_RANGE
    # Optional ContinuousAction speed_range; None → probe vehicle limits at runtime.
    continuous_speed_range: Optional[Tuple[float, float]] = None
    _cached_physical_range: Optional[Tuple[float, float]] = field(
        default=None, repr=False, compare=False
    )


_SETTINGS = ActionModeSettings()


def get_action_settings() -> ActionModeSettings:
    return _SETTINGS


def set_action_settings(settings: ActionModeSettings) -> None:
    global _SETTINGS
    mode = str(settings.mode or "meta_default").strip().lower()
    if mode not in ACTION_MODES:
        raise ValueError(f"Unknown action_mode {mode!r}; expected one of {ACTION_MODES}")
    _SETTINGS = replace(
        settings,
        mode=mode,
        _cached_physical_range=None,
    )
    logger.info(
        "Highway action_mode=%s continuous_lateral=%s",
        _SETTINGS.mode,
        _SETTINGS.continuous_lateral,
    )


def configure_action_mode_from_config(config: Any) -> ActionModeSettings:
    """Read highway_action_* fields from a pipeline/stage config object."""
    mode = str(getattr(config, "highway_action_mode", None) or "meta_default")
    lateral = bool(getattr(config, "highway_action_continuous_lateral", True))
    fine_lo = float(getattr(config, "highway_meta_fine_low", 15.0) or 15.0)
    fine_hi = float(getattr(config, "highway_meta_fine_high", 35.0) or 35.0)
    fine_n = int(getattr(config, "highway_meta_fine_n", 21) or 21)
    accel = getattr(config, "highway_acceleration_range", None)
    steer = getattr(config, "highway_steering_range", None)
    spd = getattr(config, "highway_continuous_speed_range", None)

    def _pair(raw: Any, default: Tuple[float, float]) -> Tuple[float, float]:
        if raw is None:
            return default
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return (float(raw[0]), float(raw[1]))
        return default

    settings = ActionModeSettings(
        mode=mode,
        continuous_lateral=lateral,
        meta_fine_low=fine_lo,
        meta_fine_high=fine_hi,
        meta_fine_n=max(2, fine_n),
        acceleration_range=_pair(accel, _DEFAULT_ACCEL_RANGE),
        steering_range=_pair(steer, _DEFAULT_STEER_RANGE),
        continuous_speed_range=(
            _pair(spd, (15.0, 35.0)) if spd is not None else None
        ),
    )
    set_action_settings(settings)
    return settings


def meta_target_speeds(settings: Optional[ActionModeSettings] = None) -> Tuple[float, ...]:
    s = settings or get_action_settings()
    if s.mode == "meta_fine":
        arr = np.linspace(float(s.meta_fine_low), float(s.meta_fine_high), int(s.meta_fine_n))
        return tuple(float(x) for x in arr)
    # meta_default (and any non-continuous fallback for gear lists)
    return tuple(META_DEFAULT_SPEEDS_MPS)


def resolve_action_dict(settings: Optional[ActionModeSettings] = None) -> Dict[str, Any]:
    """Full ``config['action']`` dict for the active mode (no leftover keys)."""
    s = settings or get_action_settings()
    mode = s.mode
    if mode == "continuous":
        action: Dict[str, Any] = {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": bool(s.continuous_lateral),
            "acceleration_range": list(s.acceleration_range),
            "steering_range": list(s.steering_range),
        }
        if s.continuous_speed_range is not None:
            action["speed_range"] = list(s.continuous_speed_range)
        return action
    speeds = meta_target_speeds(s)
    return {
        "type": "DiscreteMetaAction",
        "target_speeds": list(speeds),
    }


def is_continuous_mode(settings: Optional[ActionModeSettings] = None) -> bool:
    return (settings or get_action_settings()).mode == "continuous"


def physical_speed_range(
    settings: Optional[ActionModeSettings] = None,
    *,
    force_probe: bool = False,
) -> Tuple[float, float]:
    """
    Reachable ego speed band for prompts / sandbox checks.

    - meta_*: ``[min(target_speeds), max(target_speeds)]``
    - continuous: configured ``speed_range`` if set, else probe the live
      ContinuousAction / vehicle MIN_SPEED–MAX_SPEED (no hard-coded band).
    """
    s = settings or get_action_settings()
    if s.mode != "continuous":
        gears = meta_target_speeds(s)
        return (float(min(gears)), float(max(gears)))
    if s.continuous_speed_range is not None:
        lo, hi = s.continuous_speed_range
        return (float(lo), float(hi))
    if s._cached_physical_range is not None and not force_probe:
        return s._cached_physical_range
    lo, hi = _probe_continuous_speed_range(s)
    s._cached_physical_range = (lo, hi)
    return (lo, hi)


def _probe_continuous_speed_range(settings: ActionModeSettings) -> Tuple[float, float]:
    from domains.highway.env_wrapper import make_base_env

    env = make_base_env(seed=0, config={"action": resolve_action_dict(settings)})
    try:
        at = getattr(env.unwrapped, "action_type", None)
        sr = getattr(at, "speed_range", None) if at is not None else None
        if sr is not None and len(sr) >= 2:
            return (float(sr[0]), float(sr[1]))
        veh = getattr(env.unwrapped, "vehicle", None)
        cls = type(veh) if veh is not None else None
        lo = float(getattr(cls, "MIN_SPEED", -40.0))
        hi = float(getattr(cls, "MAX_SPEED", 40.0))
        return (lo, hi)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def action_space_prompt_block(settings: Optional[ActionModeSettings] = None) -> str:
    """Plain-language action-space description for LLM reward prompts."""
    s = settings or get_action_settings()
    if s.mode == "continuous":
        lo, hi = physical_speed_range(s)
        axes = (
            "Box(2,) = [acceleration, steering] (normalized)"
            if s.continuous_lateral
            else "Box(1,) = [acceleration] (normalized); lateral control disabled"
        )
        return (
            "Action space (this run):\n"
            f"- ContinuousAction ({axes}).\n"
            f"- Ego speed is continuous; physical speed limits from the simulator "
            f"are approximately [{lo:.1f}, {hi:.1f}] m/s "
            "(read from the env/vehicle — not a hand-picked cruise target).\n"
            "- Do NOT impose any specific target_speed / cruise setpoint in the "
            "reward. Shape with state.speed / state.progress / safety signals only.\n"
        )
    gears = meta_target_speeds(s)
    return (
        "Action space (this run):\n"
        f"- DiscreteMetaAction with target_speeds={list(gears)} m/s "
        f"(mode={s.mode}).\n"
        "- Actions are discrete meta-actions (lane left/idle/right/faster/slower); "
        "ego speed snaps toward those gears.\n"
        f"- If you introduce a speed setpoint constant, keep it inside "
        f"[{min(gears):.1f}, {max(gears):.1f}] m/s (reachable gears).\n"
    )


def _assign_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _literal_number(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    return None


def extract_speed_setpoints(code: str) -> List[Tuple[str, float]]:
    """Collect numeric assignments to *target_speed*-like names."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RewardSandboxError(f"syntax error: {exc}") from exc
    found: List[Tuple[str, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            val = _literal_number(node.value)
            if val is None:
                continue
            for t in node.targets:
                name = _assign_name(t)
                if name and _TARGET_NAME_RE.search(name):
                    found.append((name, val))
        elif isinstance(node, ast.AnnAssign):
            val = _literal_number(node.value) if node.value is not None else None
            if val is None:
                continue
            name = _assign_name(node.target)
            if name and _TARGET_NAME_RE.search(name):
                found.append((name, val))
    return found


def check_target_speed_literals(code: str, settings: Optional[ActionModeSettings] = None) -> None:
    """
    Reject reward code with unreachable speed setpoints.

    - meta_*: reject any target below ``min(target_speeds)`` or above ``max``.
    - continuous: reject targets outside the probed physical speed range.
    """
    s = settings or get_action_settings()
    lo, hi = physical_speed_range(s)
    # Meta: user asked to reject below min(target_speeds); also reject above max.
    for name, val in extract_speed_setpoints(code):
        if s.mode != "continuous" and val < lo:
            raise RewardSandboxError(
                f"{name}={val} is below reachable meta gear floor "
                f"{lo} m/s (target_speeds min); unreachable under DiscreteMetaAction"
            )
        if s.mode != "continuous" and val > hi:
            raise RewardSandboxError(
                f"{name}={val} is above reachable meta gear ceiling "
                f"{hi} m/s (target_speeds max)"
            )
        if s.mode == "continuous" and (val < lo or val > hi):
            raise RewardSandboxError(
                f"{name}={val} is outside physical speed range "
                f"[{lo}, {hi}] m/s for ContinuousAction"
            )


def warn_inline_speed_literals(
    code: str, settings: Optional[ActionModeSettings] = None
) -> List[str]:
    """
    Soft warnings for numeric literals used as speed setpoints in comparisons
    or subtractions against ``*.speed`` / ``*speed*`` names
    (e.g. ``abs(state.speed - 16)``), when outside the reachable action range.

    Does not reject — only reports. Hard rejection stays in
    ``check_target_speed_literals``.
    """
    s = settings or get_action_settings()
    lo, hi = physical_speed_range(s)
    warnings: List[str] = []
    try:
        hard = {round(v, 6) for _, v in extract_speed_setpoints(code)}
        tree = ast.parse(code)
    except (SyntaxError, RewardSandboxError):
        return warnings

    def _mentions_speed(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return bool(_TARGET_NAME_RE.search(node.id) or "speed" in node.id.lower())
        if isinstance(node, ast.Attribute):
            return bool(
                _TARGET_NAME_RE.search(node.attr) or "speed" in node.attr.lower()
            )
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left_speed = _mentions_speed(node.left)
            right_speed = _mentions_speed(node.right)
            if not (left_speed or right_speed):
                continue
            other = node.right if left_speed else node.left
            val = _literal_number(other)
            if val is None:
                continue
            if round(val, 6) in hard:
                continue
            if lo <= val <= hi:
                continue
            if not (5.0 <= abs(val) <= 50.0):
                continue
            warnings.append(
                f"numeric literal {val} looks like a speed setpoint outside the "
                f"reachable action range [{lo:.1f}, {hi:.1f}] m/s"
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        speedish = _mentions_speed(node.left) or any(
            _mentions_speed(c) for c in node.comparators
        )
        if not speedish:
            continue
        candidates = [_literal_number(node.left)] + [
            _literal_number(c) for c in node.comparators
        ]
        for val in candidates:
            if val is None or round(val, 6) in hard:
                continue
            if lo <= val <= hi or not (5.0 <= abs(val) <= 50.0):
                continue
            warnings.append(
                f"numeric literal {val} looks like a speed setpoint outside the "
                f"reachable action range [{lo:.1f}, {hi:.1f}] m/s"
            )

    seen: set = set()
    uniq: List[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def spaces_compatible(a: Any, b: Any) -> bool:
    """True if two Gymnasium spaces can share a PPO policy head."""
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if hasattr(a, "n") and hasattr(b, "n"):
        return int(a.n) == int(b.n)
    if hasattr(a, "shape") and hasattr(b, "shape"):
        return tuple(a.shape) == tuple(b.shape)
    return str(a) == str(b)
