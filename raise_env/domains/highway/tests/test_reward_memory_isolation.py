"""Concurrent envs must not share one reward object's per-episode memory."""

from __future__ import annotations

import pytest

from domains.highway.adapter import clone_reward_fn
from domains.highway.state import default_smoke_states
from raise_core.domains import load_domain, make_validator_for_domain

_COUNTER_REWARD = """
def compute_reward(state, memory):
    memory["n"] = memory.get("n", 0) + 1
    return float(memory["n"])
"""


def _counter_reward():
    pack = load_domain("highway")
    return make_validator_for_domain(pack).validate_code(_COUNTER_REWARD)


def test_clone_has_independent_memory():
    state = default_smoke_states()[0]
    original = _counter_reward()
    clone = clone_reward_fn(original)
    assert clone is not original

    assert original.compute(state) == 1.0
    assert original.compute(state) == 2.0
    assert clone.compute(state) == 1.0
    clone.reset()
    assert original.compute(state) == 3.0


def test_train_vec_env_gives_each_env_its_own_reward():
    pytest.importorskip("gymnasium")
    pytest.importorskip("highway_env")
    pytest.importorskip("stable_baselines3")
    from domains.highway.adapter import _make_train_vec_env

    reward = _counter_reward()
    vec = _make_train_vec_env(reward, seed=0, n_envs=3)
    try:
        rewards = [vec.envs[i].env.reward_fn for i in range(3)]
        assert rewards[0] is reward
        assert len({id(r) for r in rewards}) == 3
    finally:
        vec.close()
