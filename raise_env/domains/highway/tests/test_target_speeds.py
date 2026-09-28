"""Action-mode coverage: meta_default / meta_fine / continuous."""

from __future__ import annotations

import numpy as np
import pytest

from domains.highway.action_config import (
    META_DEFAULT_SPEEDS_MPS,
    ActionModeSettings,
    check_target_speed_literals,
    configure_action_mode_from_config,
    physical_speed_range,
    resolve_action_dict,
    set_action_settings,
)
from domains.highway.env_wrapper import (
    TARGET_SPEEDS_MPS,
    default_env_config,
    make_base_env,
    training_env_config,
)


def test_meta_default_six_gears_and_discrete5():
    set_action_settings(ActionModeSettings(mode="meta_default"))
    action = default_env_config()["action"]
    assert action["type"] == "DiscreteMetaAction"
    assert action["target_speeds"] == list(TARGET_SPEEDS_MPS)
    assert action["target_speeds"] == list(META_DEFAULT_SPEEDS_MPS)
    assert len(action["target_speeds"]) == 6
    assert action["target_speeds"] == pytest.approx(list(np.linspace(20.0, 30.0, 6)))

    env = make_base_env(seed=0)
    try:
        veh = env.unwrapped.vehicle
        gears = np.asarray(veh.target_speeds, dtype=np.float64)
        assert gears.shape == (6,)
        assert gears.tolist() == pytest.approx(list(TARGET_SPEEDS_MPS))
        assert env.action_space.n == 5
    finally:
        env.close()


def test_meta_fine_linspace_gears():
    set_action_settings(
        ActionModeSettings(
            mode="meta_fine",
            meta_fine_low=15.0,
            meta_fine_high=35.0,
            meta_fine_n=21,
        )
    )
    action = resolve_action_dict()
    assert action["type"] == "DiscreteMetaAction"
    assert len(action["target_speeds"]) == 21
    assert action["target_speeds"][0] == pytest.approx(15.0)
    assert action["target_speeds"][-1] == pytest.approx(35.0)
    env = make_base_env(seed=0, config=training_env_config())
    try:
        assert env.action_space.n == 5
        gears = list(env.unwrapped.vehicle.target_speeds)
        assert len(gears) == 21
        assert "target_speeds" in env.unwrapped.config["action"]
    finally:
        env.close()


def test_continuous_box_no_lingering_target_speeds():
    set_action_settings(
        ActionModeSettings(mode="continuous", continuous_lateral=True)
    )
    action = resolve_action_dict()
    assert action["type"] == "ContinuousAction"
    assert "target_speeds" not in action
    assert action["lateral"] is True
    env = make_base_env(seed=0, config=training_env_config())
    try:
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
        assert "target_speeds" not in env.unwrapped.config["action"]
    finally:
        env.close()


def test_continuous_lateral_false_box1():
    set_action_settings(
        ActionModeSettings(mode="continuous", continuous_lateral=False)
    )
    env = make_base_env(seed=0, config=training_env_config())
    try:
        assert env.action_space.shape == (1,)
    finally:
        env.close()
    set_action_settings(ActionModeSettings(mode="meta_default"))


def test_continuous_speeds_take_intermediate_values():
    set_action_settings(
        ActionModeSettings(mode="continuous", continuous_lateral=False)
    )
    env = make_base_env(seed=0, config=training_env_config())
    try:
        obs, _ = env.reset(seed=0)
        speeds = []
        for _ in range(40):
            a = np.array([0.5], dtype=np.float32)
            obs, _r, te, tr, _info = env.step(a)
            speeds.append(float(env.unwrapped.vehicle.speed))
            if te or tr:
                break
        uniq = sorted({round(s, 1) for s in speeds})
        assert len(uniq) >= 3, uniq
        # Not stuck on the old meta gear grid alone.
        assert any(u not in (20.0, 22.0, 24.0, 25.0, 26.0, 28.0, 30.0) for u in uniq) or (
            max(speeds) - min(speeds) > 2.0
        )
    finally:
        env.close()
        set_action_settings(ActionModeSettings(mode="meta_default"))


def test_reject_target_speed_below_meta_floor():
    set_action_settings(ActionModeSettings(mode="meta_default"))
    code = """
def compute_reward(state, memory):
    target_speed = 16.0
    return float(state.progress)
"""
    with pytest.raises(Exception) as ei:
        check_target_speed_literals(code)
    assert "16" in str(ei.value)


def test_physical_speed_range_meta_matches_gears():
    set_action_settings(ActionModeSettings(mode="meta_default"))
    lo, hi = physical_speed_range()
    assert lo == pytest.approx(20.0)
    assert hi == pytest.approx(30.0)


def test_configure_from_config_object():
    class C:
        highway_action_mode = "meta_fine"
        highway_action_continuous_lateral = True
        highway_meta_fine_n = 11
        highway_meta_fine_low = 10.0
        highway_meta_fine_high = 30.0

    configure_action_mode_from_config(C())
    assert len(resolve_action_dict()["target_speeds"]) == 11
    set_action_settings(ActionModeSettings(mode="meta_default"))
