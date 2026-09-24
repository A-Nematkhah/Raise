"""
Restricted execution of validated reward source.

Ported from mobile_robot_env.rewards.sandbox.runtime; adapted to RewardState.

Episode memory contract (option a — function-only stateful rewards):
  ``compute_reward(state, memory)`` where ``memory`` is a plain ``dict``
  owned by :class:`SandboxedReward`. ``reset()`` clears it; ``compute()``
  passes the same dict for every step within an episode. Classes are
  forbidden by the AST policy — do not teach class-based rewards in prompts.
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Any, Callable, Dict, Optional, Sequence

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError
from domains.crowdnav.state import (
    HumanObservable,
    RewardFunction,
    RewardState,
    RobotRewardState,
)

ComputeFn = Callable[[RewardState, Dict[str, Any]], float]

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "bool": bool,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sum": sum,
    "round": round,
    "pow": pow,
    "sorted": sorted,
    "reversed": reversed,
    "all": all,
    "any": any,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    # Safe read-only type check (no getattr/eval reflection surface).
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


def require_finite_float(value: object) -> float:
    """Reject bool, non-numeric, NaN, and Inf return values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardSandboxError(
            f"compute_reward must return a finite int or float, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RewardSandboxError(f"compute_reward returned a non-finite value: {number}")
    return number


def _strip_import_nodes(tree: ast.AST) -> ast.AST:
    """Remove every Import/ImportFrom anywhere in the tree (math is injected)."""

    class _Strip(ast.NodeTransformer):
        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            return None

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            return None

    cleaned = _Strip().visit(tree)
    return ast.fix_missing_locations(cleaned)


def compile_compute_reward(code: str, config: SandboxConfig) -> ComputeFn:
    """
    exec() the already-AST-checked source and return compute_reward.

    ``code`` must have passed parse/check_structure/check_interface first.
    Import statements are stripped recursively before exec because allowed
    modules (e.g. ``math``) are injected into the sandbox namespace —
    ``__import__`` is not available in restricted builtins (nested
    ``import math`` inside the function used to crash as a confusing
    ``ImportError: __import__ not found`` during smoke tests).
    """
    namespace = {"__builtins__": dict(_SAFE_BUILTINS)}
    if "math" in config.allowed_modules:
        namespace["math"] = math
    try:
        tree = ast.parse(code, mode="exec")
        tree = _strip_import_nodes(tree)
        exec(compile(tree, "<reward_candidate>", "exec"), namespace, namespace)
    except RewardSandboxError:
        raise
    except Exception as exc:
        raise RewardSandboxError(f"exec failed: {type(exc).__name__}: {exc}") from exc

    fn = namespace.get(config.required_function_name)
    if not callable(fn):
        raise RewardSandboxError(
            f"{config.required_function_name} was not defined as a callable"
        )
    return fn


def run_with_timeout(fn: Callable[[], object], timeout_seconds: float) -> object:
    """
    Run ``fn()`` on a daemon thread and raise RewardSandboxError on timeout.

    Research timeout only — the worker is not forcibly killed.
    """
    if timeout_seconds <= 0.0:
        raise RewardSandboxError("timeout_seconds must be positive")

    box: dict = {"value": None, "error": None}

    def _target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - must surface any smoke failure
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise RewardSandboxError(f"execution exceeded timeout of {timeout_seconds}s")
    if box["error"] is not None:
        raise box["error"]
    return box["value"]


def smoke_test_compute(
    compute_fn: ComputeFn,
    states: Sequence[RewardState],
    config: SandboxConfig,
) -> None:
    """Call compute_fn on each smoke state with a fresh memory dict."""
    if not states:
        raise RewardSandboxError("smoke test requires at least one RewardState")

    def _run() -> None:
        memory: Dict[str, Any] = {}
        for state in states:
            try:
                value = compute_fn(state, memory)
            except RewardSandboxError:
                raise
            except Exception as exc:
                raise RewardSandboxError(
                    f"runtime error during smoke test: {type(exc).__name__}: {exc}"
                ) from exc
            require_finite_float(value)

    run_with_timeout(_run, config.timeout_seconds)


def default_smoke_states() -> tuple[RewardState, ...]:
    """Small, deterministic RewardState snapshots for sandbox smoke tests."""

    def make_state(
        *,
        dmin: float,
        collision: bool = False,
        reaching_goal: bool = False,
        timeout: bool = False,
        px: float = 0.0,
        py: float = 0.0,
        gx: float = 4.0,
        gy: float = 0.0,
        human_d: float = 2.0,
    ) -> RewardState:
        action = (0.5, 0.0)
        humans = (
            HumanObservable(px + human_d, py, 0.0, 0.0, 0.3),
        )
        return RewardState(
            robot=RobotRewardState(
                px=px,
                py=py,
                vx=0.5,
                vy=0.0,
                radius=0.3,
                gx=gx,
                gy=gy,
                v_pref=1.0,
            ),
            humans=humans,
            dmin=float(dmin),
            discomfort_dist=0.25,
            collision=collision,
            reaching_goal=reaching_goal,
            timeout=timeout,
            action=action,
            time_step=0.25,
            global_time=1.0,
            time_limit=50.0,
        )

    return (
        make_state(dmin=1.0),
        make_state(dmin=0.1),
        make_state(dmin=0.5, reaching_goal=True, px=3.9, gx=4.0),
        make_state(dmin=-0.05, collision=True, human_d=0.4),
        make_state(dmin=2.0, timeout=True),
        make_state(dmin=0.5, px=1.0, gx=5.0),
        # Exercise common LLM failure modes before a candidate reaches rollout.
        make_state(
            dmin=0.0,
            px=1.0e6,
            py=-1.0e6,
            gx=-1.0e6,
            gy=1.0e6,
            human_d=1.0e-6,
        ),
        make_state(
            dmin=1.0e-8,
            px=-1.0e6,
            py=1.0e6,
            gx=1.0e6,
            gy=-1.0e6,
            human_d=1.0e-6,
        ),
    )


class SandboxedReward(RewardFunction):
    """
    RewardFunction wrapper around a sandbox-compiled compute_reward.

    Owns a per-episode ``memory`` dict: cleared on ``reset()``, passed as the
    second argument on every ``compute(state)`` call. Source code is retained
    so the object can be cloudpickled into ShmemVecEnv workers (Windows spawn).
    """

    def __init__(
        self,
        compute_fn: ComputeFn,
        config: SandboxConfig,
        *,
        source_code: Optional[str] = None,
    ) -> None:
        self._compute_fn = compute_fn
        self._config = config
        self._source_code = source_code
        self._memory: Dict[str, Any] = {}

    def reset(self) -> None:
        self._memory.clear()

    def compute(self, state: RewardState) -> float:
        try:
            value = self._compute_fn(state, self._memory)
        except RewardSandboxError:
            raise
        except Exception as exc:
            raise RewardSandboxError(
                f"runtime error in sandboxed compute(): {type(exc).__name__}: {exc}"
            ) from exc
        return require_finite_float(value)

    def __getstate__(self) -> Dict[str, Any]:
        if not self._source_code:
            raise RewardSandboxError(
                "SandboxedReward cannot be pickled without source_code "
                "(required for multi-process vec envs)"
            )
        return {
            "config": self._config,
            "source_code": self._source_code,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._config = state["config"]
        self._source_code = state["source_code"]
        self._memory = {}
        self._compute_fn = compile_compute_reward(self._source_code, self._config)
