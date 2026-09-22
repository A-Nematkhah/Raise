"""
Research-grade reward sandbox / validator (ported from mobile_robot_env).

Not a security-grade sandbox. AST checks + restricted exec + smoke tests
filter obviously bad LLM reward code before expensive RL.
"""

from crowd_nav.reward_search.sandbox.config import SandboxConfig
from crowd_nav.reward_search.sandbox.errors import RewardSandboxError
from crowd_nav.reward_search.sandbox.runtime import SandboxedReward, default_smoke_states
from crowd_nav.reward_search.sandbox.validator import RewardValidator

__all__ = [
    "RewardSandboxError",
    "RewardValidator",
    "SandboxConfig",
    "SandboxedReward",
    "default_smoke_states",
]
