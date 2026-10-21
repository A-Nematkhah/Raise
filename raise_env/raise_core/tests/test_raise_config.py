"""raise_config.py is the single source for env + budget defaults."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from raise_config import CONFIG, build_upstream_config, config_to_dict, resolve_config
from raise_core.config_factory import (
    ensure_config_snapshot,
    load_config_snapshot,
    make_run_config,
    snapshot_digest,
    write_config_snapshot,
)
from raise_core.config_defaults import upstream_env_config
from raise_core.explore import StageIConfig
from raise_core.pipeline import RaiseRunConfig
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.refine import Stage2Config
from raise_core.validate import Stage3Config

_RAISE_ENV = Path(__file__).resolve().parents[2]
_RUN_RAISE = _RAISE_ENV / "scripts" / "run_raise.py"

GOLDEN = {
    "population": 6,
    "generations": 4,
    "stage2_train_steps": 12_000,
    "stage3_train_steps": 70_000,
    "stage2_eval_episodes": 30,
    "stage3_eval_episodes": 30,
    "final_eval_episodes": 100,
    "stage2_seeds": 1,
    "stage2_paired_seeds": True,
    "kinematics": "unicycle",
    "collect_kinematics": "holonomic",
    "human_num": 10,
    "time_limit": 30.0,
    "time_step": 0.25,
    "val_size": 100,
    "test_size": 500,
}


def test_build_upstream_config_tracks_env_edits():
    base = build_upstream_config(CONFIG.env)
    from dataclasses import replace


    edited = replace(CONFIG, env=replace(CONFIG.env, human_num=12, time_limit=25.0))
    built = build_upstream_config(edited.env, config=edited)
    assert int(built.sim.human_num) == 12
    assert float(built.env.time_limit) == 25.0
    assert int(base.sim.human_num) == GOLDEN["human_num"]


def test_set_type_error():
    with pytest.raises(TypeError, match="expected bool"):
        resolve_config(set_overrides=("budget.stage2_paired_seeds=maybe",))


def test_print_config_cli():
    proc = subprocess.run(
        [sys.executable, str(_RUN_RAISE), "--print-config"],
        cwd=str(_RAISE_ENV),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["budget"]["population_size"] == GOLDEN["population"]


def test_config_snapshot_written_and_drift(tmp_path):
    out = str(tmp_path / "run")
    cfg = deepcopy(CONFIG)
    write_config_snapshot(out, cfg)
    snap = load_config_snapshot(out)
    assert snap is not None
    assert snap["sha256"] == snapshot_digest(cfg)
    ensure_config_snapshot(out, cfg, resume=True, allow_config_drift=False)

    from dataclasses import replace


    drifted = replace(cfg, budget=replace(cfg.budget, population_size=8))
    with pytest.raises(RuntimeError, match="config drift"):
        ensure_config_snapshot(out, drifted, resume=True, allow_config_drift=False)
    ensure_config_snapshot(out, drifted, resume=True, allow_config_drift=True)


def test_golden_frozen_defaults():
    run = make_run_config(CONFIG)
    assert run.population == GOLDEN["population"]
    assert run.generations == GOLDEN["generations"]
    assert run.stage2_train_steps == GOLDEN["stage2_train_steps"]
    assert run.stage3_train_steps == GOLDEN["stage3_train_steps"]
    assert run.stage2_eval_episodes == GOLDEN["stage2_eval_episodes"]
    assert run.stage3_eval_episodes == GOLDEN["stage3_eval_episodes"]
    assert run.final_eval_episodes == GOLDEN["final_eval_episodes"]
    assert run.stage2_seeds == GOLDEN["stage2_seeds"]
    assert run.stage2_paired_seeds is GOLDEN["stage2_paired_seeds"]
    assert run.robot_kinematics == GOLDEN["kinematics"]
    assert CONFIG.budget.stage1.collect_kinematics == GOLDEN["collect_kinematics"]
    upstream = upstream_env_config()
    assert int(upstream.sim.human_num) == GOLDEN["human_num"]
    assert float(upstream.env.time_limit) == GOLDEN["time_limit"]
    assert float(upstream.env.time_step) == GOLDEN["time_step"]
    assert int(upstream.env.val_size) == GOLDEN["val_size"]
    assert int(upstream.env.test_size) == GOLDEN["test_size"]

    assert RaiseRunConfig().population == GOLDEN["population"]
    assert ClosedLoopConfig().population_size == GOLDEN["population"]
    assert Stage2Config().train_env_steps == GOLDEN["stage2_train_steps"]
    assert Stage3Config().train_env_steps == GOLDEN["stage3_train_steps"]
    assert StageIConfig().population_size == GOLDEN["population"]
    assert StageIConfig().generations == GOLDEN["generations"]


def test_config_to_dict_roundtrip():
    assert config_to_dict(CONFIG)["seed"] == CONFIG.seed
