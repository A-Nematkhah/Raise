"""Pytest root conftest: domain runtime path + stub rvo2 when unavailable."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import raise_paths  # noqa: F401 — arms domains/crowdnav/runtime on sys.path

try:
    import rvo2  # noqa: F401
except ImportError:
    if "rvo2" not in sys.modules:
        sys.modules["rvo2"] = MagicMock()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real simulator / long-running tests")
