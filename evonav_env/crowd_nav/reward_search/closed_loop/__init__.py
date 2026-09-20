"""Closed-loop multi-fidelity evolution (innovation path). See PLAN.md."""

from __future__ import annotations

from crowd_nav.reward_search.closed_loop.config import ClosedLoopConfig
from crowd_nav.reward_search.closed_loop.runner import ClosedLoopResult, ClosedLoopRunner

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "ClosedLoopRunner",
    "PLAN_PATH",
]

PLAN_PATH = __file__.replace("__init__.py", "PLAN.md")
