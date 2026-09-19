"""
Closed loop: score → enqueue → acquire → optional surrogate re-fit (stub).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def run_active_learning_step(
    *,
    surrogate_model_dir: str = "artifacts/surrogate",
    queue_root: str = "data/active_learning",
    max_queries: int = 5,
    refit_every: int = 20,
    force_refit: bool = False,
) -> Dict[str, Any]:
    """
    One AL iteration. See PLAN.md §5.

    Returns summary counts for logging / manifest.
    """
    raise NotImplementedError(
        "active_learning.loop.run_active_learning_step — see PLAN.md"
    )
