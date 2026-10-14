"""Core modules must import cleanly as the first import of a fresh interpreter."""

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize(
    "module",
    [
        "raise_core.candidate",
        "raise_core.explore",
        "raise_core.selection",
        "raise_core.refine",
        "raise_core.validate",
        "raise_core.population_io",
        "raise_core.pipeline",
        "raise_core.raise_loop.runner",
        "domains.crowdnav.metrics",
        "domains.crowdnav.env_wrapper",
        "domains.crowdnav.adapter",
        "domains.crowdnav.pack",
    ],
)
def test_module_imports_first(module):
    env = {**os.environ, "PYTHONPATH": _ROOT}
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
