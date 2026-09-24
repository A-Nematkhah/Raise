"""Highway env wrapper smoke (requires highway-env extras)."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("highway_env")

from crowd_nav.domains import load_domain, make_validator_for_domain
from crowd_nav.domains.highway.env_wrapper import RewardInjectedHighwayEnv
from crowd_nav.domains.highway.prompts import D5_SEED_FUNCTION


@pytest.mark.slow
def test_highway_wrapper_one_episode():
    pack = load_domain("highway")
    reward = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    env = RewardInjectedHighwayEnv(reward, seed=0)
    obs, _info = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 50:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert isinstance(r, float)
        assert "raise_collision" in info
        done = bool(term or trunc)
        steps += 1
    env.close()
    assert steps >= 1
