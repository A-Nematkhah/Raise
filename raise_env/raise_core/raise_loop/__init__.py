"""Closed-loop multi-fidelity evolution (innovation path). See PLAN.md.

Canonical package name: ``raise_loop``.
On-disk run subdirectory remains ``closed_loop/`` for resume compatibility.
Class names ``ClosedLoop*`` are kept as the public API; ``RaiseLoop*`` aliases
exist for readability (identical objects — no behavior change).
"""

from __future__ import annotations

from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import ClosedLoopResult, ClosedLoopRunner

# Readable aliases (identical objects — no behavior change).
RaiseLoopConfig = ClosedLoopConfig
RaiseLoopResult = ClosedLoopResult
RaiseLoopRunner = ClosedLoopRunner

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "ClosedLoopRunner",
    "RaiseLoopConfig",
    "RaiseLoopResult",
    "RaiseLoopRunner",
    "PLAN_PATH",
]

PLAN_PATH = __file__.replace("__init__.py", "PLAN.md")
