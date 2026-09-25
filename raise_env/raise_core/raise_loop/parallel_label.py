"""Helpers to speed highway Stage II without changing K2/K3 budgets."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    trainer/env stack. ``on_done`` runs in the main thread for append/checkpoint.
    Returns ``(n_ok, n_fail)``.
    """
    pending: List[Tuple[int, Any]] = []
    for i, cand in enumerate(to_label):
        if str(cand.candidate_id) in already_done:
            continue
        pending.append((i, cand))
    if not pending:
        return 0, 0

    n_workers = max(1, int(workers))
    if n_workers == 1 or len(pending) == 1:
        n_ok = n_fail = 0
        for i, cand in pending:
            if cand.reward_fn is None:
                n_fail += 1
                on_done(cand, {"status": "failed", "reason": "no_reward_fn"}, i)
                continue
            result = label_one(cand, round_index_fn(i))
            status = str(result.get("status") or "")
            if status == "ok":
                n_ok += 1
            elif status == "failed":
                n_fail += 1
            on_done(cand, result, i)
        return n_ok, n_fail

    n_ok = n_fail = 0
    logger.info(
        "Parallel Stage II labeling: %d candidates, %d workers",
        len(pending),
        n_workers,
    )
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_map = {
            pool.submit(label_one, cand, round_index_fn(i)): (i, cand)
            for i, cand in pending
            if cand.reward_fn is not None
        }
        for i, cand in pending:
            if cand.reward_fn is None:
                n_fail += 1
                on_done(cand, {"status": "failed", "reason": "no_reward_fn"}, i)
        for fut in as_completed(fut_map):
            i, cand = fut_map[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("parallel label crashed for %s: %s", cand.candidate_id, exc)
                result = {"status": "failed", "reason": str(exc)}
            status = str(result.get("status") or "")
            if status == "ok":
                n_ok += 1
            elif status == "failed":
                n_fail += 1
            on_done(cand, result, i)
    return n_ok, n_fail
