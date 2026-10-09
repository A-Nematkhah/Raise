"""Helpers to speed highway Stage II without changing K2/K3 budgets."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def attach_warm_start_checkpoints(
    to_label: Sequence[Any],
    population: Sequence[Any],
) -> None:
    """
    Stamp ``warm_start_checkpoint`` on children from a parent's saved PPO zip.

    Uses ``parent_ids`` on the candidate and ``checkpoint_path`` on parents
    already present in the (full) population / prior labels.
    """
    by_id: Dict[str, str] = {}
    for c in population:
        path = (getattr(c, "metadata", None) or {}).get("checkpoint_path")
        if path and os.path.isfile(str(path)):
            by_id[str(c.candidate_id)] = str(path)
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


def label_candidates_parallel(
    to_label: Sequence[Any],
    *,
    already_done: Set[str],
    workers: int,
    label_one: Callable[[Any, int], Dict[str, Any]],
    on_done: Callable[[Any, Dict[str, Any], int], None],
    round_index_fn: Callable[[int], int],
) -> Tuple[int, int]:
    """
    Run Stage II labels with a thread pool.

    Note: highway-env stepping is mostly Python/NumPy on small arrays and may
    hold the GIL; PyTorch also shares an intra-op pool. ``workers>1`` can help
    when PPO/BLAS release the GIL, but can also contend with ``n_envs>1`` —
    benchmark wall-clock before assuming speedup.

    ``label_one(cand, round_index) -> result dict`` must be thread-safe for the
    trainer/env stack. ``on_done`` runs in the main thread for append/checkpoint,
    always in ``to_label`` order (results that finish early are held back), so
    ledgers and checkpoints do not depend on thread timing. A crash in one
    label is recorded as ``failed`` instead of aborting the epoch.
    Returns ``(n_ok, n_fail)``.
    """
    pending: List[Tuple[int, Any]] = []
    for i, cand in enumerate(to_label):
        if str(cand.candidate_id) in already_done:
            continue
        pending.append((i, cand))
    if not pending:
        return 0, 0

    counts = {"ok": 0, "failed": 0}

    def _emit(i: int, cand: Any, result: Dict[str, Any]) -> None:
        status = str(result.get("status") or "")
        if status in counts:
            counts[status] += 1
        on_done(cand, result, i)

    n_workers = max(1, int(workers))
    if n_workers == 1 or len(pending) == 1:
        for i, cand in pending:
            _emit(i, cand, _safe_label(label_one, cand, round_index_fn(i)))
        return counts["ok"], counts["failed"]

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
            pool.submit(_safe_label, label_one, cand, round_index_fn(i)): i
            for i, cand in pending
        }
        for fut in as_completed(fut_map):
            finished[fut_map[fut]] = fut.result()
            while next_pos < len(order) and order[next_pos] in finished:
                i = order[next_pos]
                _emit(i, by_index[i], finished.pop(i))
                next_pos += 1
    return counts["ok"], counts["failed"]


def _safe_label(
    label_one: Callable[[Any, int], Dict[str, Any]], cand: Any, round_index: int
) -> Dict[str, Any]:
    if cand.reward_fn is None:
        return {"status": "failed", "reason": "no_reward_fn"}
    try:
        return label_one(cand, round_index)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage II label crashed for %s: %s", cand.candidate_id, exc)
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
