"""
Restricted execution of validated CrowdNav++ reward source.

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
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty as QueueEmpty
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError

ComputeFn = Callable[[Any, Dict[str, Any]], Any]


class BoundedMemory(dict):
    """Episode memory dict with hard caps on keys and container value sizes."""

    def __init__(
        self,
        *args: Any,
        max_keys: int = 64,
        max_value_len: int = 10_000,
        **kwargs: Any,
    ) -> None:
        self._max_keys = int(max_keys)
        self._max_value_len = int(max_value_len)
        super().__init__()
        if args or kwargs:
            self.update(dict(*args, **kwargs))

    def _check_value(self, value: Any) -> None:
        if isinstance(value, (list, dict, tuple, set)) and len(value) > self._max_value_len:
            raise RewardSandboxError(
                f"memory value length {len(value)} exceeds max_memory_value_len "
                f"{self._max_value_len}"
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        self._check_value(value)
        if key not in self and len(self) >= self._max_keys:
            raise RewardSandboxError(
                f"memory has {self._max_keys} keys; max_memory_keys reached"
            )
        super().__setitem__(key, value)

    def update(self, other: Any = (), **kwargs: Any) -> None:  # type: ignore[override]
        items = dict(other, **kwargs) if kwargs else dict(other)
        for key, value in items.items():
            self[key] = value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: Any, *args: Any) -> Any:
        return super().pop(key, *args)


def _safe_pow_factory(config: SandboxConfig) -> Callable[..., Any]:
    max_exp = int(config.max_pow_exponent)
    max_bits = int(config.max_pow_bits)

    def _safe_pow(base: Any, exp: Any, *rest: Any) -> Any:
        if rest:
            # Ternary pow(base, exp, mod) — still bound the exponent.
            if isinstance(exp, int) and abs(exp) > max_exp:
                raise RewardSandboxError(
                    f"pow exponent |{exp}| exceeds max_pow_exponent {max_exp}"
                )
            return pow(base, exp, rest[0])
        if isinstance(base, bool) or isinstance(exp, bool):
            return pow(base, exp)
        if isinstance(base, int) and isinstance(exp, int):
            if abs(exp) > max_exp:
                raise RewardSandboxError(
                    f"integer ** / pow exponent |{exp}| exceeds max_pow_exponent {max_exp}"
                )
            if exp < 0:
                return pow(base, exp)
            # Rough bit-length of |base|**exp ≈ exp * bit_length(base).
            if base != 0 and exp * max(1, abs(base).bit_length()) > max_bits:
                raise RewardSandboxError(
                    f"integer ** / pow result would exceed max_pow_bits {max_bits}"
                )
            return pow(base, exp)
        return pow(base, exp)

    return _safe_pow


def _safe_range_factory(config: SandboxConfig) -> Callable[..., range]:
    max_len = int(config.max_range_length)

    def _safe_range(*args: Any) -> range:
        try:
            r = range(*args)
        except TypeError as exc:
            raise RewardSandboxError(f"invalid range(): {exc}") from exc
        if abs(len(r)) > max_len:
            raise RewardSandboxError(
                f"range length {len(r)} exceeds max_range_length {max_len}"
            )
        return r

    return _safe_range


def _safe_mul_factory(config: SandboxConfig) -> Callable[..., Any]:
    max_len = int(config.max_sequence_repeat)

    def _safe_mul(a: Any, b: Any) -> Any:
        # Sequence * int (either order) must not explode memory.
        if isinstance(a, (str, list, tuple)) and isinstance(b, int) and not isinstance(b, bool):
            if b < 0 or (len(a) > 0 and b > max_len // max(len(a), 1)):
                raise RewardSandboxError(
                    f"sequence repetition would exceed max_sequence_repeat {max_len}"
                )
            out = a * b
            if len(out) > max_len:
                raise RewardSandboxError(
                    f"sequence repetition length {len(out)} exceeds max_sequence_repeat {max_len}"
                )
            return out
        if isinstance(b, (str, list, tuple)) and isinstance(a, int) and not isinstance(a, bool):
            return _safe_mul(b, a)
        return a * b

    return _safe_mul


def _safe_lshift_factory(config: SandboxConfig) -> Callable[..., Any]:
    max_shift = int(config.max_shift_bits)

    def _safe_lshift(a: Any, b: Any) -> Any:
        # Non-constant shifts are rejected at rewrite time; this is a belt-and-
        # braces runtime check for constant integer shifts only.
        if not isinstance(b, int) or isinstance(b, bool):
            raise RewardSandboxError(
                "left shift requires a constant integer shift count "
                f"(got {type(b).__name__})"
            )
        if b < 0 or b > max_shift:
            raise RewardSandboxError(
                f"left shift count {b} exceeds max_shift_bits {max_shift}"
            )
        return a << b

    return _safe_lshift


def _guarded_math(config: SandboxConfig) -> Any:
    """math module with factorial/comb/perm arg caps."""
    max_arg = int(config.max_factorial_arg)
    base = math

    class _MathProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(base, name)

        def factorial(self, n: Any) -> Any:
            if isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > max_arg:
                raise RewardSandboxError(
                    f"math.factorial arg must be int in 0..{max_arg}, got {n!r}"
                )
            return base.factorial(n)

        def comb(self, n: Any, k: Any) -> Any:
            for label, v in (("n", n), ("k", k)):
                if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > max_arg:
                    raise RewardSandboxError(
                        f"math.comb {label} must be int in 0..{max_arg}, got {v!r}"
                    )
            return base.comb(n, k)

        def perm(self, n: Any, k: Any = None) -> Any:
            if isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > max_arg:
                raise RewardSandboxError(
                    f"math.perm n must be int in 0..{max_arg}, got {n!r}"
                )
            if k is not None and (
                isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > max_arg
            ):
                raise RewardSandboxError(
                    f"math.perm k must be int in 0..{max_arg}, got {k!r}"
                )
            return base.perm(n, k) if k is not None else base.perm(n)

    return _MathProxy()


class _RewriteBinOps(ast.NodeTransformer):
    """Rewrite ``**``, ``<<``, and sequence ``*`` to guarded helpers."""

    def __init__(self, config: SandboxConfig) -> None:
        self._max_shift = int(config.max_shift_bits)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node, ast.BinOp):
            return node
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_safe_pow", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        if isinstance(node.op, ast.LShift):
            # Reject non-constant shifts and constant shifts > max_shift_bits
            # at rewrite time (before exec).
            right = node.right
            const = None
            if isinstance(right, ast.Constant) and isinstance(right.value, int):
                const = int(right.value)
            elif isinstance(right, ast.UnaryOp) and isinstance(right.op, ast.USub):
                if isinstance(right.operand, ast.Constant) and isinstance(
                    right.operand.value, int
                ):
                    const = -int(right.operand.value)
            if const is None:
                raise RewardSandboxError(
                    "left shift requires a constant integer shift count"
                )
            if const < 0 or const > self._max_shift:
                raise RewardSandboxError(
                    f"left shift count {const} exceeds max_shift_bits {self._max_shift}"
                )
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_safe_lshift", ctx=ast.Load()),
                    args=[node.left, ast.Constant(value=const)],
                    keywords=[],
                ),
                node,
            )
        if isinstance(node.op, ast.Mult):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_safe_mul", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node


def _apply_smoke_rlimit(config: SandboxConfig) -> None:
    """Best-effort memory cap for the smoke child (POSIX only).

    Applied only immediately before candidate code runs (after imports and
    compilation). Limit is ``current_rss_or_as + configured_budget`` so child
    startup with torch/numpy already loaded never fails. Prefers
    ``RLIMIT_AS``; falls back to ``RLIMIT_DATA``. Never raises.
    """
    if sys.platform == "win32":
        return
    budget = int(getattr(config, "smoke_rlimit_as_bytes", 0) or 0)
    if budget <= 0:
        return
    try:
        import resource

        soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
        # Estimate current usage from max RSS when available (kB on Linux).
        usage = 0
        try:
            usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                pass  # macOS reports bytes
            else:
                usage *= 1024  # Linux: kilobytes
        except (AttributeError, ValueError, OSError):
            usage = 0
        limit = max(budget, usage + budget)
        # Respect an existing tighter hard cap.
        if hard_as != resource.RLIM_INFINITY:
            limit = min(limit, hard_as)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_as))
            return
        except (ValueError, OSError):
            pass
        if hasattr(resource, "RLIMIT_DATA"):
            soft_d, hard_d = resource.getrlimit(resource.RLIMIT_DATA)
            if hard_d != resource.RLIM_INFINITY:
                limit = min(limit, hard_d)
            resource.setrlimit(resource.RLIMIT_DATA, (limit, hard_d))
    except (ValueError, OSError, AttributeError):
        pass


def _make_memory(config: SandboxConfig) -> BoundedMemory:
    return BoundedMemory(
        max_keys=int(config.max_memory_keys),
        max_value_len=int(config.max_memory_value_len),
    )


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

# Cap on reward-component dict size (EUREKA-style component returns).
MAX_REWARD_COMPONENT_KEYS = 12

_SAFE_BUILTINS_BASE = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "bool": bool,
    "len": len,
    "enumerate": enumerate,
    "zip": zip,
    "sum": sum,
    "round": round,
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
    finite float is accepted. When True (domain validators), also accepts
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

    ``**`` / ``<<`` / sequence ``*`` are rewritten to guarded helpers;
    ``range`` / ``pow`` builtins use the same rules as ``**``.
    """
    safe_pow = _safe_pow_factory(config)
    safe_range = _safe_range_factory(config)
    safe_mul = _safe_mul_factory(config)
    safe_lshift = _safe_lshift_factory(config)
    builtins = dict(_SAFE_BUILTINS_BASE)
    builtins["range"] = safe_range
    builtins["pow"] = safe_pow
    namespace: Dict[str, Any] = {
        "__builtins__": builtins,
        "_safe_pow": safe_pow,
        "_safe_mul": safe_mul,
        "_safe_lshift": safe_lshift,
    }
    if "math" in config.allowed_modules:
        namespace["math"] = _guarded_math(config)
    try:
        tree = ast.parse(code, mode="exec")
        tree = _strip_import_nodes(tree)
        tree = _RewriteBinOps(config).visit(tree)
        ast.fix_missing_locations(tree)
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


def _smoke_worker(
    queue: Any,
    source_code: str,
    config: SandboxConfig,
    states: Sequence[Any],
    allow_components: bool = False,
    ready: Any = None,
) -> None:
    """Child-process entry: recompile source and smoke-test states."""
    if ready is not None:
        ready.set()
    try:
        # Compile first (imports / AST rewrite) with no rlimit so startup cannot
        # fail; apply the memory cap only right before candidate execution.
        compute_fn = compile_compute_reward(source_code, config)
        _apply_smoke_rlimit(config)
        memory: Dict[str, Any] = _make_memory(config)
        prev_time: Optional[float] = None
        for state in states:
            # Smoke states come from unrelated episodes. Time that does not move
            # forward marks an episode boundary, where the env would reset memory.
            now = getattr(state, "global_time", None)
            if now is not None:
                now = float(now)
                if prev_time is not None and now <= prev_time:
                    memory.clear()
                prev_time = now
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


def _stop_process(proc: Any) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2.0)


def _await_smoke_result(
    proc: Any,
    queue: Any,
    timeout_seconds: float,
    startup_timeout_seconds: Optional[float] = None,
    ready: Any = None,
) -> Tuple[str, Any]:
    """Read the worker's result before joining (a full pipe would block its exit).

    With ``ready`` (an Event the worker sets before touching candidate code),
    the ``timeout_seconds`` budget starts when it is set, so slow process
    startup is not charged to the candidate.
    """
    started = ready is None or startup_timeout_seconds is None
    budget = float(timeout_seconds) if started else float(startup_timeout_seconds)
    deadline = time.monotonic() + budget
    while True:
        if not started and ready.is_set():
            started = True
            deadline = time.monotonic() + float(timeout_seconds)
        try:
            return queue.get(timeout=0.05)
        except QueueEmpty:
            pass
        if not proc.is_alive():
            try:
                return queue.get(timeout=0.5)
            except QueueEmpty:
                raise RewardSandboxError(
                    f"smoke worker exited with code {proc.exitcode} without a result"
                ) from None
        if time.monotonic() >= deadline:
            if not started:
                raise RewardSandboxError(
                    f"smoke worker did not start within {startup_timeout_seconds}s"
                )
            raise RewardSandboxError(f"execution exceeded timeout of {timeout_seconds}s")


def smoke_test_compute(
    source_code: str,
    states: Sequence[Any],
    config: SandboxConfig,
    *,
    allow_components: bool = False,
) -> None:
    """
    Compile ``source_code`` in a killable spawned child and run every smoke state.

    ``allow_components`` (domain validators only): accept ``(float, dict[str, float])``
    returns in addition to bare floats (compatibility shim).
    """
    if not states:
        raise RewardSandboxError("smoke test requires at least one state")
    if not source_code:
        raise RewardSandboxError("smoke test requires source_code")
    if config.timeout_seconds <= 0.0:
        raise RewardSandboxError("timeout_seconds must be positive")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    ready = ctx.Event()
    proc = ctx.Process(
        target=_smoke_worker,
        args=(queue, str(source_code), config, tuple(states), bool(allow_components), ready),
    )
    proc.start()
    try:
        status, payload = _await_smoke_result(
            proc,
            queue,
            float(config.timeout_seconds),
            float(config.startup_timeout_seconds),
            ready,
        )
        proc.join(timeout=2.0)
    finally:
        _stop_process(proc)
        queue.close()
        proc.close()
    if status == "err":
        raise payload


class SandboxedReward(RewardFunction):
    """
    RewardFunction wrapper around a sandbox-compiled compute_reward.

    Owns a per-episode ``memory`` dict: cleared on ``reset()``, passed as the
    second argument on every ``compute(state)`` call. Source code is retained
    so the object can be cloudpickled into ShmemVecEnv workers (Windows spawn).

    ``allow_components`` must only be True for domain validators (domain pack
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
        self._memory: Dict[str, Any] = _make_memory(config)
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
        self._memory = _make_memory(self._config)
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
        except (MemoryError, RecursionError) as exc:
            # Reward-caused resource faults: penalize the candidate, not the run.
            raise RewardSandboxError(
                f"runtime error in sandboxed compute(): {type(exc).__name__}: {exc}"
            ) from exc
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
        self._memory = _make_memory(self._config)
        self._last_components = {}
        self._init_call_timing()
        self._compute_fn = compile_compute_reward(self._source_code, self._config)
