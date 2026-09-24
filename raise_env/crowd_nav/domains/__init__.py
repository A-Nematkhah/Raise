"""Backward-compatible domain registry re-export.

Canonical: ``raise_core.domains``.
"""

from __future__ import annotations

from raise_core.domains import *  # noqa: F401,F403
from raise_core.domains import __all__ as _all

__all__ = list(_all)
