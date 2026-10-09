"""lane_index must match highway-env's lane geometry (lane i centred at y = 4*i)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from domains.highway.stage1 import (
    STATE_SCHEMA_VERSION,
    migrate_highway_dataset,
    migrate_state_dict,
)
from domains.highway.state import LANE_WIDTH_M, lane_index_from_y


@pytest.mark.parametrize("lane", [0, 1, 2, 3])
def test_lane_centres_map_to_their_lane(lane):
    assert lane_index_from_y(lane * LANE_WIDTH_M) == lane


def test_lane_boundaries_and_clamp():
    assert lane_index_from_y(1.9) == 0
    assert lane_index_from_y(2.1) == 1
    assert lane_index_from_y(-3.0) == 0
    assert lane_index_from_y(40.0) == 3


def test_kinematics_to_state_lane_and_relative_velocity():
    pytest.importorskip("gymnasium")
    from domains.highway.env_wrapper import kinematics_to_state

    obs = np.array(
        [
            [1.0, 100.0, 8.0, 25.0, 0.0, 0.0],
            [1.0, 120.0, 4.0, 20.0, 0.0, 0.0],
        ]
    )
    state, _ = kinematics_to_state(
        obs,
        action=1,
        collision=False,
        off_road=False,
        timeout=False,
        time_step=0.2,
        global_time=0.2,
        time_limit=40.0,
        prev_ego_x=None,
    )
    assert state.ego.lane_index == 2.0
    assert state.others[0].vx == pytest.approx(-5.0)
    assert state.others[0].y == pytest.approx(-4.0)


def test_lane_index_matches_simulator():
    pytest.importorskip("gymnasium")
    pytest.importorskip("highway_env")
    from domains.highway.env_wrapper import make_base_env

    env = make_base_env(seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(150):
        vehicle = env.unwrapped.vehicle
        if vehicle.on_road and abs(vehicle.heading) < 0.05:
            assert lane_index_from_y(float(obs[0][2])) == int(vehicle.lane_index[2])
            checked += 1
        obs, _r, term, trunc, _info = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    assert checked > 20


def test_migrate_state_dict_fixes_lane_and_offroad_timeout():
    state = {
        "ego": {"x": 0.0, "y": 8.0, "lane_index": 3.0},
        "off_road": True,
        "timeout": True,
    }
    out = migrate_state_dict(state)
    assert out["ego"]["lane_index"] == 2.0
    assert out["timeout"] is False
    assert state["ego"]["lane_index"] == 3.0


def test_migrate_dataset_is_idempotent(tmp_path):
    root = tmp_path / "stage1_dataset"
    root.mkdir()
    rec = {
        "trajectory_id": "hw_0000",
        "scenario_id": "s",
        "seed": 1,
        "label": "timeout",
        "behavior": "synth_timeout",
        "metadata": {},
        "states": [
            {"ego": {"y": 4.0, "lane_index": 2.0}, "off_road": True, "timeout": True}
        ],
    }
    (root / "trajectories.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"domain": "highway"}), encoding="utf-8")

    assert migrate_highway_dataset(str(root), backup=True) == 1
    migrated = json.loads((root / "trajectories.jsonl").read_text(encoding="utf-8"))
    assert migrated["states"][0]["ego"]["lane_index"] == 1.0
    assert migrated["states"][0]["timeout"] is False
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_schema_version"] == STATE_SCHEMA_VERSION
    assert any(name.startswith("stage1_dataset_backup_") for name in os.listdir(tmp_path))

    assert migrate_highway_dataset(str(root), backup=False) == 0
