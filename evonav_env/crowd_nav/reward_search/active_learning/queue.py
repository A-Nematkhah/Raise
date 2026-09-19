"""
Active-learning queue persistence (stub).

Default: ``data/active_learning/queue.jsonl`` + ``manifest.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from crowd_nav.reward_search.active_learning.query import QueryItem


def default_queue_root() -> str:
    return "data/active_learning"


def enqueue(root: str, items: List[QueryItem]) -> None:
    raise NotImplementedError("active_learning.queue.enqueue — see PLAN.md")


def dequeue_batch(root: str, *, limit: int = 5) -> List[QueryItem]:
    raise NotImplementedError("active_learning.queue.dequeue_batch — see PLAN.md")


def mark_done(root: str, item: QueryItem, *, result: Dict[str, Any]) -> None:
    raise NotImplementedError("active_learning.queue.mark_done — see PLAN.md")
