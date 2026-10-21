"""Helpers to speed Stage II without changing K2/K3 budgets."""

from __future__ import annotations

import gc
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

INFRA_ERROR_POLICIES = ("retry_then_abort", "abort", "penalize")
# Failure kinds returned by classify_label_failure / _safe_label.
FAILURE_KINDS = ("reward", "infra", "unknown")


class LabelStats(NamedTuple):
    n_ok: int
    n_fail: int
    n_reward_fail: int
    n_infra_fail: int


class LabelAbortError(RuntimeError):
    """Raised after a resumable ``aborted_infra`` / unknown abort checkpoint."""

    def __init__(self, message: str, *, result: Dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def _existing_path(path: Any) -> Optional[str]:
    if not path:
        return None
    raw = str(path)
    for cand in (raw, raw.replace("\\", os.sep).replace("/", os.sep), Path(raw).as_posix()):
        if os.path.isfile(cand):
            return cand
    return None


def attach_warm_start_checkpoints(
    to_label: Sequence[Any],
    population: Sequence[Any],
) -> None:
    """
    Stamp ``warm_start_checkpoint`` on children from a parent's saved PPO zip.

    Uses ``parent_ids`` on the candidate and ``checkpoint_path`` on parents
    already present in the (full) population / prior labels. Stored paths use
    POSIX separators so artifacts stay portable across OSes.
    """
    by_id: Dict[str, str] = {}
    for c in population:
        path = _existing_path((getattr(c, "metadata", None) or {}).get("checkpoint_path"))
        if path:
            by_id[str(c.candidate_id)] = Path(path).as_posix()
    for cand in to_label:
        md = dict(getattr(cand, "metadata", None) or {})
        if md.get("warm_start_checkpoint"):
            continue
        parents = getattr(cand, "parent_ids", ()) or ()
        for pid in parents:
            path = by_id.get(str(pid))
            if path:
                md["warm_start_checkpoint"] = path
                cand.metadata = md
                break


def classify_label_failure(exc: BaseException) -> str:
    """Return ``reward``, ``infra``, or ``unknown`` for a label exception.

    Prefer type checks. Message substrings are only used for known
    ``RuntimeError`` CUDA/OOM infra cases (torch may raise plain RuntimeError)
    and never for unknown classification.
    """
    try:
        from raise_core.sandbox.errors import RewardSandboxError
    except ImportError:  # pragma: no cover
        RewardSandboxError = ()  # type: ignore[assignment,misc]
    try:
        from domains.crowdnav.watchdog import RewardStallError
    except ImportError:  # pragma: no cover
        RewardStallError = ()  # type: ignore[assignment,misc]

    if isinstance(exc, RewardSandboxError) or isinstance(exc, RewardStallError):
        return "reward"
    # Memory/recursion inside SandboxedReward.compute are wrapped as
    # RewardSandboxError; an unwrapped MemoryError here is process/infra OOM.
    name = type(exc).__name__
    module = getattr(type(exc), "__module__", "") or ""
    if name == "OutOfMemoryError" and "torch" in module:
        return "infra"
    if isinstance(exc, (OSError, MemoryError, FileNotFoundError)):
        return "infra"
    if type(exc) is RuntimeError:
        msg = str(exc).lower()
        if "cuda" in msg or "out of memory" in msg:
            return "infra"
    return "unknown"


def _clear_infra_caches() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            try:
                cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _abort_result(
    cand: Any, exc: BaseException, *, kind: str, after_retry: bool
) -> Dict[str, Any]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "status": "aborted_infra",
        "failure_kind": kind,
        "reason": f"{type(exc).__name__}: {exc}",
        "traceback": tb,
        "after_retry": bool(after_retry),
        "candidate_id": str(getattr(cand, "candidate_id", "")),
    }


def label_candidates_parallel(
    to_label: Sequence[Any],
    *,
    already_done: Set[str],
    workers: int,
    label_one: Callable[[Any, int], Dict[str, Any]],
    on_done: Callable[[Any, Dict[str, Any], int], None],
    round_index_fn: Callable[[int], int],
    infra_error_policy: str = "retry_then_abort",
) -> LabelStats:
    """
    Run Stage II labels with a thread pool.

    Infrastructure / unknown failures honour ``infra_error_policy``. On abort,
    ``on_done`` is invoked with ``status="aborted_infra"`` (so the caller can
    persist a resumable checkpoint) and then :class:`LabelAbortError` is raised.
    """
    policy = str(infra_error_policy or "retry_then_abort")
    if policy not in INFRA_ERROR_POLICIES:
        raise ValueError(
            f"infra_error_policy must be one of {INFRA_ERROR_POLICIES}, got {policy!r}"
        )

    pending: List[Tuple[int, Any]] = []
    for i, cand in enumerate(to_label):
        if str(cand.candidate_id) in already_done:
            continue
        pending.append((i, cand))
    if not pending:
        return LabelStats(0, 0, 0, 0)

    counts = {"ok": 0, "failed": 0, "reward": 0, "infra": 0, "unknown": 0}

    def _emit(i: int, cand: Any, result: Dict[str, Any]) -> None:
        status = str(result.get("status") or "")
        if status == "ok":
            counts["ok"] += 1
        elif status == "failed":
            counts["failed"] += 1
            kind = str(result.get("failure_kind") or "reward")
            if kind in counts:
                counts[kind] += 1
        elif status == "aborted_infra":
            kind = str(result.get("failure_kind") or "infra")
            if kind in counts:
                counts[kind] += 1
            counts["failed"] += 1
        # Persist checkpoint (including aborted_infra) before any raise.
        on_done(cand, result, i)
        if status == "aborted_infra":
            raise LabelAbortError(
                f"{kind} error labeling {cand.candidate_id} "
                f"({'after retry' if result.get('after_retry') else 'abort'}): "
                f"{result.get('reason')}. Full traceback:\n{result.get('traceback')}",
                result=result,
            )

    n_workers = max(1, int(workers))
    if n_workers == 1 or len(pending) == 1:
        for i, cand in pending:
            _emit(
                i,
                cand,
                _safe_label(
                    label_one, cand, round_index_fn(i), infra_error_policy=policy
                ),
            )
        return LabelStats(
            counts["ok"],
            counts["failed"],
            counts["reward"],
            counts["infra"] + counts["unknown"],
        )

    logger.info(
        "Parallel Stage II labeling: %d candidates, %d workers",
        len(pending),
        n_workers,
    )
    by_index = dict(pending)
    order = [i for i, _ in pending]
    finished: Dict[int, Dict[str, Any]] = {}
    next_pos = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_map = {
            pool.submit(
                _safe_label,
                label_one,
                cand,
                round_index_fn(i),
                infra_error_policy=policy,
            ): i
            for i, cand in pending
        }
        for fut in as_completed(fut_map):
            finished[fut_map[fut]] = fut.result()
            while next_pos < len(order) and order[next_pos] in finished:
                i = order[next_pos]
                _emit(i, by_index[i], finished.pop(i))
                next_pos += 1
    return LabelStats(
        counts["ok"],
        counts["failed"],
        counts["reward"],
        counts["infra"] + counts["unknown"],
    )


def _safe_label(
    label_one: Callable[[Any, int], Dict[str, Any]],
    cand: Any,
    round_index: int,
    *,
    infra_error_policy: str = "retry_then_abort",
) -> Dict[str, Any]:
    if cand.reward_fn is None:
        return {
            "status": "failed",
            "reason": "no_reward_fn",
            "failure_kind": "reward",
        }
    try:
        return label_one(cand, round_index)
    except Exception as exc:  # noqa: BLE001
        kind = classify_label_failure(exc)
        logger.exception(
            "Stage II label crashed for %s (%s): %s",
            cand.candidate_id,
            kind,
            exc,
        )
        if kind in ("infra", "unknown"):
            policy = str(infra_error_policy or "retry_then_abort")
            if policy == "penalize":
                return {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "failure_kind": kind,
                }
            if policy == "abort":
                return _abort_result(cand, exc, kind=kind, after_retry=False)
            # retry_then_abort
            _clear_infra_caches()
            try:
                return label_one(cand, round_index)
            except Exception as retry_exc:  # noqa: BLE001
                retry_kind = classify_label_failure(retry_exc)
                logger.exception(
                    "Stage II label retry failed for %s (%s): %s",
                    cand.candidate_id,
                    retry_kind,
                    retry_exc,
                )
                if retry_kind == "reward":
                    return {
                        "status": "failed",
                        "reason": f"{type(retry_exc).__name__}: {retry_exc}",
                        "failure_kind": "reward",
                    }
                return _abort_result(cand, retry_exc, kind=retry_kind, after_retry=True)
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "failure_kind": kind,
        }
