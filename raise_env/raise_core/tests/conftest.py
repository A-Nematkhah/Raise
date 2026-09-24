"""Pytest fixtures: stub rvo2 only if the native module is missing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import rvo2  # noqa: F401
except ImportError:
    if "rvo2" not in sys.modules:
        sys.modules["rvo2"] = MagicMock()
