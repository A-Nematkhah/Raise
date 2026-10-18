"""Skip CrowdNav tests cleanly when the simulator / PPO dependencies are not installed."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_CROWDNAV_DEPS = ("gymnasium", "gym", "rvo2", "stable_baselines3")
_MISSING = [name for name in _CROWDNAV_DEPS if importlib.util.find_spec(name) is None]

collect_ignore_glob = ["test_*.py"] if _MISSING else []

RAISE_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_collector():
    """``scripts/collect_crowdnav_stage1_dataset.py`` as a module (scripts/ is not a package)."""
    path = os.path.join(RAISE_ENV_ROOT, "scripts", "collect_crowdnav_stage1_dataset.py")
    spec = importlib.util.spec_from_file_location("collect_crowdnav_stage1_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def crowdnav_collector():
    return load_collector()


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    """Small real Stage I dataset (6 train scenarios x 5 robot behaviours)."""
    out = tmp_path_factory.mktemp("cn_stage1")
    config_dir = tmp_path_factory.mktemp("cn_config")
    config_path = config_dir / "config.py"
    config_source = Path(load_collector().DEFAULT_ENV_CONFIG_PATH).read_text(encoding="utf-8")
    if "    sim.human_num = 20\n" not in config_source:
        raise RuntimeError("Unexpected CrowdNav++ default human_num config")
    config_path.write_text(
        config_source.replace("    sim.human_num = 20\n", "    sim.human_num = 1\n"),
        encoding="utf-8",
    )
    rc = load_collector().main(
        [
            "--out",
            str(out),
            "--episodes",
            "6",
            "--seed",
            "7",
            "--env-config",
            str(config_path),
            "--behaviours",
            "orca,linear,linear_slow,stop,random",
        ]
    )
    assert rc == 0
    return str(out)
