"""The project installs, configures and runs without highway-env."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


_RAISE_ROOT = Path(__file__).resolve().parents[3]
_SUBPROCESS_SNIPPET = """
import sys

def _block(name, *args, **kwargs):
    if name.split(".")[0] == "highway_env":
        raise ImportError("highway_env is blocked in this test")
    return _real(name, *args, **kwargs)

import builtins
_real = builtins.__import__
builtins.__import__ = _block

from raise_core.domains import get_pack

pack = get_pack()
bad = sorted(
    m
    for m in sys.modules
    if m == "highway_env" or m.startswith("highway_env.")
)
print("PACK", pack.name)
print("HIGHWAY_MODULES", bad)
"""


def test_core_and_pack_import_without_highway_env():
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SNIPPET],
        cwd=str(_RAISE_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "PACK crowdnav" in proc.stdout
    assert "HIGHWAY_MODULES []" in proc.stdout


def test_requirements_and_ci_do_not_install_highway_env():
    for rel in ("requirements.txt", "../.github/workflows/raise-ci.yml"):
        path = (_RAISE_ROOT / rel).resolve()
        text = path.read_text(encoding="utf-8").lower()
        assert "highway-env" not in text and "highway_env" not in text, rel


def test_no_source_file_imports_highway():
    pattern = re.compile(r"^\s*(from|import)\s+(highway_env|domains\.highway)\b", re.M)
    roots = (_RAISE_ROOT / "raise_core", _RAISE_ROOT / "domains", _RAISE_ROOT / "scripts")
    for root in roots:
        for path in root.rglob("*.py"):
            if path.match("**/tests/**"):
                continue
            text = path.read_text(encoding="utf-8")
            assert not pattern.search(text), str(path.relative_to(_RAISE_ROOT))
