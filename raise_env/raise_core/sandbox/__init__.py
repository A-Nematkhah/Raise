"""
Research-grade reward sandbox / validator (ported from mobile_robot_env).

Not a security-grade sandbox. AST checks + restricted exec + smoke tests
filter obviously bad LLM reward code before expensive RL.
"""

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import SandboxedReward, default_smoke_states
from raise_core.sandbox.validator import RewardValidator

__all__ = [
    "RewardSandboxError",
    "RewardValidator",
    "SandboxConfig",
    "SandboxedReward",
    "default_smoke_states",
]
