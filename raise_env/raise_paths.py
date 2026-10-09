"""Put the ``raise_env`` root on ``sys.path`` (scripts / conftest import this)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))


def ensure_raise_paths() -> str:
    """Insert the raise_env root onto ``sys.path``. Idempotent."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT


ensure_raise_paths()
