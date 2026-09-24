"""Highway domain pack registration and contracts."""

from __future__ import annotations

import os

import pytest

from crowd_nav.domains import available_domains, load_domain, make_validator_for_domain
from crowd_nav.domains.highway.prompts import D5_SEED_FUNCTION


def test_highway_registered():
    assert "highway" in available_domains()


def test_load_highway_pack():
    pack = load_domain("highway")
    assert pack.name == "highway"
    assert os.path.isfile(pack.spec_path)
    assert "compute_reward" in pack.seed_reward_source
    assert pack.stage1_dataset_default == "data/highway_stage1_dataset"
    assert pack.smoke_states_fn is not None
    assert pack.make_stage2_trainer is not None
    assert pack.make_score_fn is not None
    assert pack.metadata.get("pipeline_profile") == "highway"


def test_highway_smoke_score_fn():
    pack = load_domain("highway")
    score_fn, dataset = pack.make_score_fn(mode="smoke")
    assert dataset is None
    validator = make_validator_for_domain(pack)
    reward = validator.validate_code(D5_SEED_FUNCTION)
    result = score_fn(reward, candidate_id="seed")
    assert hasattr(result, "score")
    assert float(result.score) == float(result.score)  # finite


def test_highway_d5_sandbox():
    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    reward = validator.validate_code(D5_SEED_FUNCTION)
    assert reward is not None
    # One compute on a smoke state
    states = pack.smoke_states_fn()
    val = reward.compute(states[0])
    assert isinstance(val, float)
