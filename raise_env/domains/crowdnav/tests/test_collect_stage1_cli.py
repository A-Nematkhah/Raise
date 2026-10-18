"""Stage I dataset collector command-line options."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_collector_accepts_and_records_custom_env_config(
    crowdnav_collector, monkeypatch, tmp_path
):
    collector = crowdnav_collector
    config_path = str(tmp_path / "crowdnav_hard.py")
    captured = {}

    def collect(**kwargs):
        captured.update(kwargs)
        return []

    def save_dataset(out, trajectories, *, extra_manifest):
        captured["saved_to"] = out
        captured["trajectory_count"] = len(trajectories)
        captured["manifest"] = extra_manifest

    monkeypatch.setattr(collector, "collect", collect)
    monkeypatch.setattr(collector, "save_crowdnav_dataset", save_dataset)

    rc = collector.main(
        [
            "--env-config",
            config_path,
            "--out",
            str(tmp_path / "dataset"),
            "--behaviours",
            "stop",
            "--kinematics",
            "unicycle",
        ]
    )

    assert rc == 0
    assert captured["env_config_path"] == config_path
    assert captured["kinematics"] == "unicycle"
    assert captured["saved_to"] == str(tmp_path / "dataset")
    assert captured["trajectory_count"] == 0
    assert captured["manifest"]["env_config"] == config_path
    assert captured["manifest"]["kinematics"] == "unicycle"
    assert captured["manifest"]["upstream_commit"] == collector.UPSTREAM_COMMIT
    assert captured["manifest"]["simulator"].endswith("CrowdSimVarNum")


def test_orca_joint_state_contains_only_fov_visible_humans(crowdnav_collector):
    collector = crowdnav_collector

    class Agent:
        def __init__(self, px):
            self.px = px

        def get_observable_state(self):
            return [self.px, 0.0, 0.0, 0.0, 0.3]

    robot = SimpleNamespace(get_full_state=lambda: [0.0] * 9)
    env = SimpleNamespace(
        sim=SimpleNamespace(
            robot=robot,
            humans=[Agent(1.0), Agent(2.0)],
            human_visibility=[False, True],
        )
    )

    state = collector._joint_state(env)
    assert state.self_state.px == 0.0
    assert len(state.human_states) == 1
    assert state.human_states[0].px == 2.0


def test_controllers_produce_unicycle_controls(crowdnav_collector):
    collector = crowdnav_collector
    env = SimpleNamespace(
        settings=SimpleNamespace(kinematics="unicycle"),
        sim=SimpleNamespace(
            robot=SimpleNamespace(px=0.0, py=0.0, gx=1.0, gy=0.0, theta=0.0, v_pref=1.0),
            desiredVelocity=[0.5],
        ),
    )
    rng = np.random.default_rng(3)

    slow = collector._make_controller("linear_slow", env, rng)()
    stop = collector._make_controller("stop", env, rng)()
    random_action = collector._make_controller("random", env, rng)()

    assert isinstance(slow, collector.ActionRot)
    assert slow.v == pytest.approx(collector.UNICYCLE_SPEED_DELTA_MIN)
    assert isinstance(stop, collector.ActionRot)
    assert stop.v == pytest.approx(collector.UNICYCLE_SPEED_DELTA_MIN)
    assert isinstance(random_action, collector.ActionRot)
    assert collector.UNICYCLE_SPEED_DELTA_MIN <= random_action.v <= collector.UNICYCLE_SPEED_DELTA_MAX
    assert abs(random_action.r) <= collector.UNICYCLE_ROTATION_DELTA_MAX


def test_collector_runs_behaviours_with_unicycle_kinematics(
    crowdnav_collector, tmp_path
):
    collector = crowdnav_collector
    config_source = Path(collector.DEFAULT_ENV_CONFIG_PATH).read_text(encoding="utf-8")
    assert "    sim.human_num = 20\n" in config_source
    config_path = tmp_path / "one_human_config.py"
    config_path.write_text(
        config_source.replace("    sim.human_num = 20\n", "    sim.human_num = 1\n"),
        encoding="utf-8",
    )

    trajectories = collector.collect(
        episodes=1,
        seed=7,
        behaviours=["orca", "linear", "linear_slow", "stop", "random"],
        env_config_path=str(config_path),
        kinematics="unicycle",
    )

    assert len(trajectories) == 5
    assert {trajectory.behavior for trajectory in trajectories} == {
        "orca",
        "linear",
        "linear_slow",
        "stop",
        "random",
    }
    assert all(state.robot.kinematics == "unicycle" for t in trajectories for state in t.states)
