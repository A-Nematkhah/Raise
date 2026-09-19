"""
Query scoring: which candidates / scenarios to acquire next (stub).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryItem:
    """One unit of work for the active-learning worker."""

    kind: str  # "stage1_scenario" | "stage2_label" | "stage3_label"
    priority: float
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def score_queries(
    candidates: List[Any],
    *,
    surrogate_preds: List[Any],
    score1_results: Optional[List[Any]] = None,
    top_k: int = 5,
) -> List[QueryItem]:
    """
    Rank acquisition targets by disagreement / uncertainty.

    See PLAN.md §3–4.
    """
    raise NotImplementedError(
        "active_learning.query.score_queries — see active_learning/PLAN.md"
    )
