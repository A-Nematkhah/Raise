"""Highway sandbox field allowlist (AST Attribute policy)."""

from __future__ import annotations

from typing import Tuple

from raise_core.sandbox.config import SandboxConfig

# Fields on HighwayRewardState / EgoVehicle / NearbyVehicle (state.py).
HIGHWAY_STATE_ATTRIBUTES: Tuple[str, ...] = (
    # HighwayRewardState
    "ego",
    "others",
    "collision",
    "off_road",
    "timeout",
    "action",
    "time_step",
    "global_time",
    "time_limit",
    "progress",
    "speed",
    # EgoVehicle + NearbyVehicle kinematics
    "x",
    "y",
    "vx",
    "vy",
    "heading",
    "lane_index",
    "on_road",
)

# Common hallucinated names LLMs invent for highway (must stay OFF the allowlist).
HIGHWAY_FORBIDDEN_HALLUCINATIONS: Tuple[str, ...] = (
    "lane_position",
    "distance",
    "robot",
    "humans",
    "gx",
    "gy",
    "px",
    "py",
    "vx_ego",
    "history",
    "lane_id",
    "ego_vehicle",
    "nearby",
    "vehicles",
    "position",
    "velocity",
    "reaching_goal",
)


def highway_sandbox_config(**overrides) -> SandboxConfig:
    """SandboxConfig with Attribute allowlist for highway-fast-v0 rewards."""
    return SandboxConfig(
        allowed_attributes=HIGHWAY_STATE_ATTRIBUTES,
        **overrides,
    )
