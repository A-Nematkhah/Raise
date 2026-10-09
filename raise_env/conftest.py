"""Pytest root conftest: put raise_env on sys.path."""

from __future__ import annotations

import raise_paths  # noqa: F401 — puts raise_env root on sys.path


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real simulator / long-running tests")
