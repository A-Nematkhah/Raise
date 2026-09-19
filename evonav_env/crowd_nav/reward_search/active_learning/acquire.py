"""
Execute query items: collect trajs or run short labeling (stub).
"""

from __future__ import annotations

from typing import Any, Dict

from crowd_nav.reward_search.active_learning.query import QueryItem


def execute_query(item: QueryItem, *, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one acquisition action and return a result dict for ``mark_done``.

    kind=stage1_scenario → extend Stage I dataset / side pool
    kind=stage2_label    → append surrogate_dataset example
    """
    raise NotImplementedError(
        "active_learning.acquire.execute_query — see active_learning/PLAN.md"
    )
