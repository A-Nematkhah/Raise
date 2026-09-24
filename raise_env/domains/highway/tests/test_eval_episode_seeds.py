"""Eval episodes must use distinct seeds (not 20 copies of one traj)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("highway_env")

from domains.highway.env_wrapper import RewardInjectedHighwayEnv
from domains.highway.prompts import D5_SEED_FUNCTION
from raise_core.domains import load_domain, make_validator_for_domain


def test_reset_without_reseeding_advances_scenario():
    pack = load_domain("highway")
    reward = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    env = RewardInjectedHighwayEnv(reward, seed=7)
    obs1, _ = env.reset()  # consumes construction seed
    obs2, _ = env.reset()  # must NOT re-apply seed=7
    a1 = np.asarray(obs1, dtype=np.float64)
    a2 = np.asarray(obs2, dtype=np.float64)
    # Traffic spawn should differ; if identical, the freeze bug is back.
    assert a1.shape == a2.shape
    assert not np.allclose(a1, a2), "consecutive resets reused the same seed"
    env.close()


def test_explicit_episode_seeds_differ():
    pack = load_domain("highway")
    reward = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    env = RewardInjectedHighwayEnv(reward, seed=0)
    obs_a, _ = env.reset(seed=100)
    obs_b, _ = env.reset(seed=101)
    assert not np.allclose(
        np.asarray(obs_a, dtype=np.float64),
        np.asarray(obs_b, dtype=np.float64),
    )
    env.close()
