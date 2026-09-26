"""
Sandbox configuration.

Ported from mobile_robot_env.rewards.sandbox.config. CrowdNav candidates
receive a RewardState argument named ``state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class SandboxConfig:
    # Default raised from 1s → 5s: smoke validation now runs in a spawn'd
    # child process (killable on timeout); Windows cold-start can exceed 1s.
    timeout_seconds: float = 5.0
    max_code_length: int = 20_000
    required_function_name: str = REQUIRED_FUNCTION_NAME
    required_arg_name: str = REQUIRED_ARG_NAME
    required_memory_arg_name: str = REQUIRED_MEMORY_ARG_NAME
    allowed_modules: Tuple[str, ...] = ALLOWED_MODULES
    forbidden_names: Tuple[str, ...] = FORBIDDEN_NAME_IDS
    allow_while: bool = False
    allow_imports: bool = False
    # When set (e.g. highway), only these Attribute names + SAFE_METHOD_ATTRIBUTES
    # are allowed. None = CrowdNav / legacy (only underscore ban).
    allowed_attributes: Optional[Tuple[str, ...]] = None
