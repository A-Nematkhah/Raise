"""PPO sees an ego-centric, scaled observation; the reward still sees absolute state."""

from __future__ import annotations

import numpy as np

from domains.highway.env_wrapper import (
    OBS_DV_SCALE,
    OBS_X_SCALE,
    OBS_Y_SCALE,
    RewardInjectedHighwayEnv,
    policy_observation,
    training_env_config,
)
from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states

_PROGRESS_REWARD = """
def compute_reward(state, memory):
    return float(state.progress)
"""


def test_policy_observation_relative_and_scaled():
    raw = np.array(
        [
            [1.0, 500.0, 4.0, 25.0, 0.0, 0.0],
            [1.0, 530.0, 8.0, 21.0, 0.0, 0.01],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = policy_observation(raw)
    assert out.shape == raw.shape and out.dtype == np.float32
    assert out[0, 1] == 0.0  # world x dropped
    assert np.isclose(out[0, 2], 4.0 / OBS_Y_SCALE)
    assert np.isclose(out[1, 1], 30.0 / OBS_X_SCALE)
    assert np.isclose(out[1, 2], 4.0 / OBS_Y_SCALE)
    assert np.isclose(out[1, 3], -4.0 / OBS_DV_SCALE)
    assert np.all(out[2] == 0.0)


def test_wrapper_obs_bounded_and_info_lane():
    reward = make_highway_validator(smoke_states=default_smoke_states()).validate_code(
        _PROGRESS_REWARD
    )
    env = RewardInjectedHighwayEnv(reward, seed=0, config=training_env_config())
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert "raise_lane_index" in info
        for _ in range(60):
            obs, _r, term, trunc, info = env.step(3)
            assert np.all(np.abs(obs) <= 3.0)
            assert obs[0, 1] == 0.0
            assert 0 <= info["raise_lane_index"] <= 3
            assert info["raise_ego_x"] > 0.0
            if term or trunc:
                break
    finally:
        env.close()
