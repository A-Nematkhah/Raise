"""state.action is a plain int (meta) or tuple of floats (continuous) everywhere."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from domains.highway.action_config import ActionModeSettings, set_action_settings
from domains.highway.env_wrapper import (
    RewardInjectedHighwayEnv,
    action_for_state,
    training_env_config,
)
from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states

_CONTINUOUS_REWARD = """
def compute_reward(state, memory):
    accel = float(state.action[0])
    return float(state.progress) - 0.01 * accel * accel
"""

_INT_REWARD = """
def compute_reward(state, memory):
    return state.action - 1
"""


@pytest.fixture
def restore_action_mode():
    yield
    set_action_settings(ActionModeSettings())


def test_action_for_state_plain_types():
    assert action_for_state(np.int64(3), spaces.Discrete(5)) == 3
    assert type(action_for_state(np.array(2), spaces.Discrete(5))) is int
    box = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    out = action_for_state(np.array([0.5, -0.25], dtype=np.float32), box)
    assert out == (0.5, -0.25)
    assert all(type(a) is float for a in out)


def test_smoke_states_follow_action_mode(restore_action_mode):
    assert all(type(s.action) is int for s in default_smoke_states())
    set_action_settings(
        ActionModeSettings(mode="continuous", continuous_speed_range=(0.0, 40.0))
    )
    assert all(len(s.action) == 2 for s in default_smoke_states())
    set_action_settings(
        ActionModeSettings(
            mode="continuous",
            continuous_lateral=False,
            continuous_speed_range=(0.0, 40.0),
        )
    )
    assert all(len(s.action) == 1 for s in default_smoke_states())


def test_continuous_reward_reading_action_passes_smoke(restore_action_mode):
    set_action_settings(
        ActionModeSettings(mode="continuous", continuous_speed_range=(0.0, 40.0))
    )
    validator = make_highway_validator(smoke_states=default_smoke_states())
    validator.validate_code(_CONTINUOUS_REWARD)


def test_int_returning_reward_survives_numpy_actions():
    reward = make_highway_validator(smoke_states=default_smoke_states()).validate_code(
        _INT_REWARD
    )
    env = RewardInjectedHighwayEnv(reward, seed=0, config=training_env_config())
    try:
        env.reset(seed=0)
        _obs, r, *_ = env.step(np.int64(1))
        assert r == 0.0
        _obs, r, *_ = env.step(np.array(3))
        assert r == 2.0
    finally:
        env.close()
