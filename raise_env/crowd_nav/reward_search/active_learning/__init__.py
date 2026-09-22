"""
Active learning package (targeted data / label acquisition).

See ``PLAN.md`` (locked v1). Depends on Surrogate v1 uncertainty + SR/CR/TR API.
Do not enable in the main pipeline until deliberately wired.
"""

from __future__ import annotations

from crowd_nav.reward_search.active_learning.loop import run_active_learning_step
from crowd_nav.reward_search.active_learning.query import QueryItem, score_queries

__all__ = [
    "QueryItem",
    "score_queries",
    "run_active_learning_step",
]
