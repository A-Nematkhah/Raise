"""Bootstrap ``raise_env`` on ``sys.path`` for CLI entry points."""

from __future__ import annotations

import os
import sys


def raise_env_root() -> str:
    """Absolute path to the ``raise_env`` package root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def bootstrap_raise_env(*, chdir: bool = False) -> str:
    """Ensure imports from ``raise_env`` work when running a script by path."""
    root = raise_env_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    if chdir:
        os.chdir(root)
    return root


__all__ = ["bootstrap_raise_env", "raise_env_root"]
