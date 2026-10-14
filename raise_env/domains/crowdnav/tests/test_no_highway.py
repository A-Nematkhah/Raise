"""The project installs, configures and runs without highway-env."""

from __future__ import annotations

import os
import re
import subprocess
import sys

from domains.crowdnav.tests.conftest import RAISE_ENV_ROOT

_REPO_ROOT = os.path.dirname(RAISE_ENV_ROOT)

_PROBE = r"""
import sys
import builtins
_real_import = builtins.__import__
def _guard(name, *args, **kwargs):
    if name.split(".")[0] == "highway_env":
        raise ImportError("highway_env is blocked in this test")
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _guard

import raise_paths
from raise_core.domains import DEFAULT_DOMAIN, available_domains, load_domain
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
import domains.crowdnav.adapter, domains.crowdnav.viz, domains.crowdnav.post_run

pack = load_domain(DEFAULT_DOMAIN)
RaisePipeline(RaiseRunConfig(fast=True))
bad = sorted(m for m in sys.modules if "highway" in m)
print("DOMAINS", sorted(available_domains()), pack.name)
print("HIGHWAY_MODULES", bad)
"""


def test_core_and_pack_import_without_highway_env():
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=RAISE_ENV_ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "DOMAINS ['crowdnav'] crowdnav" in proc.stdout
    assert "HIGHWAY_MODULES []" in proc.stdout


def test_requirements_and_ci_do_not_install_highway_env():
    for rel in ("raise_env/requirements.txt", ".github/workflows/raise-ci.yml"):
        with open(os.path.join(_REPO_ROOT, rel), encoding="utf-8") as fh:
            text = fh.read().lower()
        assert "highway" not in text, rel
    with open(os.path.join(RAISE_ENV_ROOT, "requirements.txt"), encoding="utf-8") as fh:
        reqs = fh.read()
    assert re.search(r"^gym==0\.15\.7$", reqs, re.M)
    assert "Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d" in reqs


def test_no_source_file_imports_highway():
    pattern = re.compile(r"^\s*(from|import)\s+(highway_env|domains\.highway)\b", re.M)
    offenders = []
    for base in ("raise_core", "domains", "scripts"):
        for root, dirs, files in os.walk(os.path.join(RAISE_ENV_ROOT, base)):
            dirs[:] = [d for d in dirs if d not in ("data", "__pycache__")]
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(root, name)
                    with open(path, encoding="utf-8") as fh:
                        if pattern.search(fh.read()):
                            offenders.append(os.path.relpath(path, RAISE_ENV_ROOT))
    assert offenders == []
