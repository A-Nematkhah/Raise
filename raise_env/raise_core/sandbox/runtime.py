"""
Restricted execution of validated reward source.

Ported from mobile_robot_env.rewards.sandbox.runtime.

Episode memory contract (option a — function-only stateful rewards):
  ``compute_reward(state, memory)`` where ``memory`` is a plain ``dict``
  owned by :class:`SandboxedReward`. ``reset()`` clears it; ``compute()``
  passes the same dict for every step within an episode. Classes are
  forbidden by the AST policy — do not teach class-based rewards in prompts.
"""

from __future__ import annotations

import ast
import math
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError

ComputeFn = Callable[[Any, Dict[str, Any]], Any]


class RewardFunction(ABC):
    """Abstract interface every reward implementation must satisfy."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode, before any compute() call."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, state: Any) -> float:
        """Compute the scalar reward for one step, given state."""
        raise NotImplementedError

# Cap on reward-component dict size (highway EUREKA-style returns).
MAX_REWARD_COMPONENT_KEYS = 12

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


def require_reward_components(value: object) -> Dict[str, float]:
    """
    Validate a flat ``dict[str, float]`` of per-step reward components.

    Rejects nested containers, non-str keys, non-finite values, and dicts
    larger than ``MAX_REWARD_COMPONENT_KEYS``. Does not coerce — fails hard.
    """
    if not isinstance(value, dict):
        raise RewardSandboxError(
            "reward components must be a dict[str, float], "
            f"got {type(value).__name__}"
        )
    if len(value) > MAX_REWARD_COMPONENT_KEYS:
        raise RewardSandboxError(
            f"reward components dict has {len(value)} keys; "
            f"max allowed is {MAX_REWARD_COMPONENT_KEYS}"
        )
    out: Dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise RewardSandboxError(
                "reward component keys must be str, "
                f"got {type(key).__name__}"
            )
        if isinstance(raw, (dict, list, tuple, set)):
            raise RewardSandboxError(
                f"reward component {key!r} must be a finite float, "
                f"got nested {type(raw).__name__}"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RewardSandboxError(
                f"reward component {key!r} must be a finite float, "
                f"got {type(raw).__name__}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise RewardSandboxError(
                f"reward component {key!r} is non-finite: {number}"
            )
        out[key] = number
    return out


def unpack_reward_return(
    value: object,
    *,
    allow_components: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Normalize a ``compute_reward`` return value to ``(total, components)``.

    When ``allow_components`` is False, only a bare
    finite float is accepted. When True (highway), also accepts
    ``(float, dict[str, float])`` and shims a bare float to ``(float, {})``.
    """
    if isinstance(value, bool):
        raise RewardSandboxError(
            f"compute_reward must return a finite int or float, got {type(value).__name__}"
        )
    if isinstance(value, (int, float)):
        return require_finite_float(value), {}
    if not allow_components:
        raise RewardSandboxError(
            f"compute_reward must return a finite int or float, got {type(value).__name__}"
        )
    if not isinstance(value, tuple) or len(value) != 2:
        raise RewardSandboxError(
            "compute_reward must return a finite float or "
            "(total_reward: float, reward_components: dict[str, float])"
        )
    total = require_finite_float(value[0])
    comps = require_reward_components(value[1])
    return total, comps


def require_bounded_reward(total: float, config: SandboxConfig) -> float:
    if abs(total) > config.max_abs_reward:
        raise RewardSandboxError(
            f"reward magnitude too large: {total:.3g} "
            f"(keep per-step rewards within +/-{config.max_abs_reward:g})"
        )
    return total


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


def _timeout_worker(queue: Any, fn: Callable[[], object]) -> None:
    """Child-process entry for ``run_with_timeout`` (must stay top-level for spawn)."""
    try:
        queue.put(("ok", fn()))
    except BaseException as exc:  # noqa: BLE001 — surface any failure to parent
        queue.put(("err", exc))


def run_with_timeout(fn: Callable[[], object], timeout_seconds: float) -> object:
    """
    Run ``fn()`` in a child process; kill it on timeout.

    Granularity: called once per **candidate smoke validation** (a handful of
    states), not once per env step — so process spawn cost is acceptable.
    ``fn`` must be picklable under the spawn start method (top-level function
    or equivalently reconstructible payload). Raises ``RewardSandboxError``
    on timeout; the worker process is terminated and joined.
    """
    if timeout_seconds <= 0.0:
        raise RewardSandboxError("timeout_seconds must be positive")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_timeout_worker, args=(queue, fn))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            try:
                proc.kill()
            except AttributeError:
                pass
            proc.join(timeout=2.0)
        raise RewardSandboxError(f"execution exceeded timeout of {timeout_seconds}s")
    if proc.exitcode not in (0, None) and queue.empty():
        raise RewardSandboxError(
            f"sandbox worker exited with code {proc.exitcode}"
        )
    try:
        status, payload = queue.get(timeout=1.0)
    except Exception as exc:  # noqa: BLE001
        raise RewardSandboxError(
            f"sandbox worker returned no result: {type(exc).__name__}: {exc}"
        ) from exc
    if status == "err":
        raise payload
    return payload


def _smoke_worker(
    queue: Any,
    source_code: str,
    config: SandboxConfig,
    states: Sequence[Any],
    allow_components: bool = False,
) -> None:
    """Child-process entry: recompile source and smoke-test states."""
    try:
        compute_fn = compile_compute_reward(source_code, config)
        memory: Dict[str, Any] = {}
        for state in states:
            started = time.perf_counter()
            try:
                value = compute_fn(state, memory)
            except RewardSandboxError:
                raise
            except Exception as exc:
                raise RewardSandboxError(
                    f"runtime error during smoke test: {type(exc).__name__}: {exc}"
                ) from exc
            elapsed = time.perf_counter() - started
            if elapsed > config.smoke_max_call_seconds:
                raise RewardSandboxError(
                    f"reward too slow: one compute_reward call took {elapsed:.3f}s "
                    f"(limit {config.smoke_max_call_seconds}s per step); "
                    "avoid loops whose length grows with the state"
                )
            total, _ = unpack_reward_return(value, allow_components=bool(allow_components))
            require_bounded_reward(total, config)
        queue.put(("ok", None))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("err", exc))


def smoke_test_compute(
    compute_fn: Optional[ComputeFn],
    states: Sequence[Any],
    config: SandboxConfig,
    *,
    source_code: Optional[str] = None,
    allow_components: bool = False,
) -> None:
    """
    Call compute_fn on each smoke state with a fresh memory dict.

    When ``source_code`` is provided (preferred), runs in a killable child
    process. Without source, falls back to ``run_with_timeout`` on a local
    closure (must be picklable — prefer passing source_code from the validator).

    ``allow_components`` (highway only): accept ``(float, dict[str, float])``
    returns in addition to bare floats (compatibility shim).
    """
    if not states:
        raise RewardSandboxError("smoke test requires at least one state")

    if source_code:
        if config.timeout_seconds <= 0.0:
            raise RewardSandboxError("timeout_seconds must be positive")
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_smoke_worker,
            args=(
                queue,
                str(source_code),
                config,
                tuple(states),
                bool(allow_components),
            ),
        )
        proc.start()
        proc.join(timeout=float(config.timeout_seconds))
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
            if proc.is_alive():
                try:
                    proc.kill()
                except AttributeError:
                    pass
                proc.join(timeout=2.0)
            raise RewardSandboxError(
                f"execution exceeded timeout of {config.timeout_seconds}s"
            )
        try:
            status, payload = queue.get(timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            raise RewardSandboxError(
                f"smoke worker returned no result: {type(exc).__name__}: {exc}"
            ) from exc
        if status == "err":
            raise payload
        return

    if compute_fn is None:
        raise RewardSandboxError("smoke test needs either compute_fn or source_code")

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
            unpack_reward_return(value, allow_components=bool(allow_components))

    run_with_timeout(_run, config.timeout_seconds)


class SandboxedReward(RewardFunction):
    """
    RewardFunction wrapper around a sandbox-compiled compute_reward.

    Owns a per-episode ``memory`` dict: cleared on ``reset()``, passed as the
    second argument on every ``compute(state)`` call. Source code is retained
    so the object can be cloudpickled into ShmemVecEnv workers (Windows spawn).

    ``allow_components`` must only be True for highway validators (domain pack
    wiring); otherwise tuple returns are rejected.
    """

    def __init__(
        self,
        compute_fn: ComputeFn,
        config: SandboxConfig,
        *,
        source_code: Optional[str] = None,
        allow_components: bool = False,
    ) -> None:
        self._compute_fn = compute_fn
        self._config = config
        self._source_code = source_code
        self._allow_components = bool(allow_components)
        self._memory: Dict[str, Any] = {}
        self._last_components: Dict[str, float] = {}
        self._init_call_timing()

    def _init_call_timing(self) -> None:
        self._recent_call_seconds: deque = deque(maxlen=int(self._config.mean_call_window))
        self._recent_call_total = 0.0

    def _check_call_time(self, elapsed: float) -> None:
        cfg = self._config
        if elapsed > cfg.max_call_seconds:
            raise RewardSandboxError(
                f"reward too slow: one compute() call took {elapsed:.3f}s "
                f"(limit {cfg.max_call_seconds}s)"
            )
        window = self._recent_call_seconds
        if len(window) == window.maxlen:
            self._recent_call_total -= window[0]
        window.append(elapsed)
        self._recent_call_total += elapsed
        if len(window) == window.maxlen:
            mean = self._recent_call_total / len(window)
            if mean > cfg.max_mean_call_seconds:
                raise RewardSandboxError(
                    f"reward too slow: mean compute() time {mean * 1000:.1f}ms over "
                    f"the last {len(window)} calls (limit "
                    f"{cfg.max_mean_call_seconds * 1000:.1f}ms)"
                )

    def reset(self) -> None:
        self._memory.clear()
        self._last_components = {}

    def last_reward_components(self) -> Dict[str, float]:
        """Most recent step's component dict (empty if none)."""
        return dict(self._last_components)

    def compute(self, state: Any) -> float:
        started = time.perf_counter()
        try:
            value = self._compute_fn(state, self._memory)
        except RewardSandboxError:
            raise
        except Exception as exc:
            raise RewardSandboxError(
                f"runtime error in sandboxed compute(): {type(exc).__name__}: {exc}"
            ) from exc
        self._check_call_time(time.perf_counter() - started)
        total, comps = unpack_reward_return(
            value, allow_components=self._allow_components
        )
        require_bounded_reward(total, self._config)
        # Stash last step only — never accumulate here (diagnostics callback owns trends).
        self._last_components = comps
        return total

    def __getstate__(self) -> Dict[str, Any]:
        if not self._source_code:
            raise RewardSandboxError(
                "SandboxedReward cannot be pickled without source_code "
                "(required for multi-process vec envs)"
            )
        return {
            "config": self._config,
            "source_code": self._source_code,
            "allow_components": self._allow_components,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._config = state["config"]
        self._source_code = state["source_code"]
        self._allow_components = bool(state.get("allow_components", False))
        self._memory = {}
        self._last_components = {}
        self._init_call_timing()
        self._compute_fn = compile_compute_reward(self._source_code, self._config)
