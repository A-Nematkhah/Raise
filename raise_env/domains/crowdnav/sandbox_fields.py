"""CrowdNav sandbox field allowlist (AST Attribute policy)."""

from __future__ import annotations

from typing import Tuple

from raise_core.sandbox.config import SandboxConfig

# Fields on CrowdNavRewardState / RobotState / HumanState (state.py).
CROWDNAV_STATE_ATTRIBUTES: Tuple[str, ...] = (
    # CrowdNavRewardState
    "robot",
    "humans",
    "action",
    "collision",
    "reached_goal",
    "timeout",
    "min_separation",
    "discomfort_dist",
    "goal_distance",
    "progress",
    "time_step",
    "global_time",
    "time_limit",
    "danger",
    "success_reward",
    "collision_penalty",
    "discomfort_penalty_factor",
    "potential_delta",
    "potential_factor",
    # RobotState + HumanState
    "px",
    "py",
    "vx",
    "vy",
    "gx",
    "gy",
    "theta",
    "radius",
    "v_pref",
    "kinematics",
    "visible",
)

# Names LLMs commonly invent for this state (must stay OFF the allowlist).
CROWDNAV_FORBIDDEN_HALLUCINATIONS: Tuple[str, ...] = (
    "self_state",
    "human_states",
    "position",
    "velocity",
    "goal_position",
    "dmin",
    "reaching_goal",
    "distance",
    "speed",
    "history",
    "ego",
    "others",
    "agents",
    "obstacles",
    "info",
)

# Scalar (non-indexable) fields; ``action`` is a tuple and deliberately absent.
CROWDNAV_SCALAR_ATTRIBUTES: Tuple[str, ...] = tuple(
    a for a in CROWDNAV_STATE_ATTRIBUTES if a not in ("robot", "humans", "action")
)


def crowdnav_sandbox_config(**overrides) -> SandboxConfig:
    """SandboxConfig with the Attribute allowlist for CrowdNav rewards."""
    overrides.setdefault("restrict_top_level", True)
    return SandboxConfig(
        allowed_attributes=CROWDNAV_STATE_ATTRIBUTES,
        **overrides,
    )
