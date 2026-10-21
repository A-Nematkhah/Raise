"""
Sandbox configuration.

Ported from mobile_robot_env.rewards.sandbox.config. Candidates receive the
domain reward state as an argument named ``state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from raise_core.config_defaults import (
    safety_max_abs_reward,
    safety_max_call_seconds,
    safety_max_code_length,
    safety_max_factorial_arg,
    safety_max_mean_call_seconds,
    safety_max_memory_keys,
    safety_max_memory_value_len,
    safety_max_pow_bits,
    safety_max_pow_exponent,
    safety_max_range_length,
    safety_max_sequence_repeat,
    safety_max_shift_bits,
    safety_mean_call_window,
    safety_smoke_max_call_seconds,
    safety_smoke_rlimit_as_bytes,
    safety_startup_timeout_seconds,
    safety_timeout_seconds,
)

REQUIRED_FUNCTION_NAME = "compute_reward"
REQUIRED_ARG_NAME = "state"
REQUIRED_MEMORY_ARG_NAME = "memory"

# Keep math listed for namespace injection only. AST rejects import statements
# when ``allow_imports`` is False (default) so nested ``import math`` cannot
# reach smoke tests as ``ImportError: __import__ not found``.
ALLOWED_MODULES: Tuple[str, ...] = ("math",)

FORBIDDEN_NAME_IDS: Tuple[str, ...] = (
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "type",
    "object",
    "super",
    "property",
    "classmethod",
    "staticmethod",
    "breakpoint",
    "memoryview",
    "exit",
    "quit",
    "help",
    "print",
    # Explicit (also covered by Name.startswith("__") in ast_policy):
    "__import__",
    "__builtins__",
    "__build_class__",
    "__loader__",
    "__spec__",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "pickle",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "builtins",
    "io",
    "shutil",
    "requests",
    "urllib",
)


# Safe method names on memory dict / sequences (not state schema fields).
SAFE_METHOD_ATTRIBUTES: Tuple[str, ...] = (
    "get",
    "pop",
    "setdefault",
    "clear",
    "update",
    "items",
    "keys",
    "values",
    "append",
    "extend",
    "add",
)

# ``math.<name>`` uses allowed on the pre-injected ``math`` module (pure functions
# and constants; anything else on ``math`` is rejected).
SAFE_MATH_CONSTANTS: Tuple[str, ...] = ("pi", "e", "tau", "inf")
SAFE_MATH_ATTRIBUTES: Tuple[str, ...] = SAFE_MATH_CONSTANTS + (
    "sqrt",
    "exp",
    "log",
    "log1p",
    "log10",
    "tanh",
    "sin",
    "cos",
    "tan",
    "atan",
    "atan2",
    "hypot",
    "fabs",
    "floor",
    "ceil",
    "copysign",
    "isfinite",
    "isnan",
)


@dataclass(frozen=True)
class SandboxConfig:
    # Default raised from 1s → 5s: smoke validation now runs in a spawn'd
    # child process (killable on timeout); Windows cold-start can exceed 1s.
    timeout_seconds: float = field(default_factory=safety_timeout_seconds)
    # Interpreter spawn + re-import of the parent's __main__ happen before the
    # child runs any candidate code and are bounded separately.
    startup_timeout_seconds: float = field(default_factory=safety_startup_timeout_seconds)
    max_code_length: int = field(default_factory=safety_max_code_length)
    required_function_name: str = REQUIRED_FUNCTION_NAME
    required_arg_name: str = REQUIRED_ARG_NAME
    required_memory_arg_name: str = REQUIRED_MEMORY_ARG_NAME
    allowed_modules: Tuple[str, ...] = ALLOWED_MODULES
    forbidden_names: Tuple[str, ...] = FORBIDDEN_NAME_IDS
    allow_while: bool = False
    allow_imports: bool = False
    # When set (domain packs), only these Attribute names + SAFE_METHOD_ATTRIBUTES
    # are allowed. None = only the underscore ban.
    allowed_attributes: Optional[Tuple[str, ...]] = None
    # When True (domain packs), module-level assignments may only bind constant
    # expressions and bare expressions may only be docstrings: no calls,
    # lambdas or comprehensions run at import time.
    restrict_top_level: bool = False
    # Wall-clock budget for compute_reward at train/score time. Stage II labels
    # in threads, so single calls include GIL waits: the per-call cap is loose
    # and the rolling mean catches steadily slow (data-dependent) rewards.
    max_call_seconds: float = field(default_factory=safety_max_call_seconds)
    max_mean_call_seconds: float = field(default_factory=safety_max_mean_call_seconds)
    mean_call_window: int = field(default_factory=safety_mean_call_window)
    # Smoke states run alone in a child process, so a tighter cap is safe.
    smoke_max_call_seconds: float = field(default_factory=safety_smoke_max_call_seconds)
    # Per-step |total reward| bound: PPO runs without reward normalization, so
    # huge scales wreck the value loss regardless of the reward's idea.
    max_abs_reward: float = field(default_factory=safety_max_abs_reward)
    # Guarded ``range(...)`` length limit (DoS via ``sum(range(10**12))``).
    max_range_length: int = field(default_factory=safety_max_range_length)
    # Episode ``memory`` dict bounds.
    max_memory_keys: int = field(default_factory=safety_max_memory_keys)
    max_memory_value_len: int = field(default_factory=safety_max_memory_value_len)
    # Smoke-child address-space cap on POSIX (bytes); 0 disables. Skipped on Windows.
    smoke_rlimit_as_bytes: int = field(default_factory=safety_smoke_rlimit_as_bytes)
    # Integer ``**`` / ``pow`` guards (exponent magnitude and result bit estimate).
    max_pow_exponent: int = field(default_factory=safety_max_pow_exponent)
    max_pow_bits: int = field(default_factory=safety_max_pow_bits)
    # Left-shift and sequence-repetition guards.
    max_shift_bits: int = field(default_factory=safety_max_shift_bits)
    max_sequence_repeat: int = field(default_factory=safety_max_sequence_repeat)
    max_factorial_arg: int = field(default_factory=safety_max_factorial_arg)
