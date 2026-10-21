"""
Research-grade reward sandbox / validator for CrowdNav++ LLM reward code.

Not a security-grade sandbox. AST checks + restricted exec + smoke tests
filter obviously bad LLM reward code before expensive PPO training.
"""

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import RewardFunction, SandboxedReward
from raise_core.sandbox.validator import RewardValidator

__all__ = [
    "RewardFunction",
    "RewardSandboxError",
    "RewardValidator",
    "SandboxConfig",
    "SandboxedReward",
]
