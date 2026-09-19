"""Domain pack baseline: CrowdNav pack must preserve Algorithm 1 prompt/seed."""

from __future__ import annotations

import os

import pytest

from crowd_nav.domains import (
    DEFAULT_DOMAIN,
    DomainPack,
    available_domains,
    load_domain,
)
from crowd_nav.domains.crowdnav import prompts as pack_prompts
from crowd_nav.reward_search import prompts as legacy_prompts


def test_default_domain_is_crowdnav():
    assert DEFAULT_DOMAIN == "crowdnav"
    assert "crowdnav" in available_domains()


def test_load_crowdnav_pack():
    pack = load_domain("crowdnav")
    assert isinstance(pack, DomainPack)
    assert pack.name == "crowdnav"
    assert os.path.isfile(pack.spec_path)
    assert "compute_reward" in pack.seed_reward_source
    assert pack.stage1_dataset_default == "data/stage1_dataset"


def test_unknown_domain_raises():
    with pytest.raises(KeyError, match="Unknown domain"):
        load_domain("not_a_real_domain")


def test_prompt_module_identity_via_reexport():
    """reward_search.prompts must re-export the same CrowdNav pack objects."""
    assert legacy_prompts.D5_SEED_FUNCTION is pack_prompts.D5_SEED_FUNCTION
    assert legacy_prompts.D1_SYSTEM_PROMPT is pack_prompts.D1_SYSTEM_PROMPT
    assert legacy_prompts.D3_SYSTEM_PROMPT is pack_prompts.D3_SYSTEM_PROMPT
    assert legacy_prompts.format_d1_initial is pack_prompts.format_d1_initial
    assert legacy_prompts.format_d2_crossover is pack_prompts.format_d2_crossover
    assert legacy_prompts.format_d3_repair is pack_prompts.format_d3_repair


def test_pack_formatters_match_legacy_strings():
    pack = load_domain("crowdnav")
    assert pack.d1_system_prompt == legacy_prompts.D1_SYSTEM_PROMPT
    assert pack.d3_system_prompt == legacy_prompts.D3_SYSTEM_PROMPT
    assert pack.d5_seed_function == legacy_prompts.D5_SEED_FUNCTION
    assert pack.format_d1_initial(include_seed=False, include_external_knowledge=False) == (
        legacy_prompts.format_d1_initial(
            include_seed=False, include_external_knowledge=False
        )
    )
    assert pack.format_d1_initial_batch(4, include_seed=False) == (
        legacy_prompts.format_d1_initial_batch(4, include_seed=False)
    )
    cross = pack.format_d2_crossover("codeA", "codeB", "note")
    assert cross == legacy_prompts.format_d2_crossover("codeA", "codeB", "note")
    mut = pack.format_d2_mutation("elitist", "weakness")
    assert mut == legacy_prompts.format_d2_mutation("elitist", "weakness")


def test_pack_exposes_stage1_make_score_fn():
    pack = load_domain("crowdnav")
    assert pack.make_score_fn is not None
    assert pack.stage1_dataset_default == "data/stage1_dataset"
    smoke_fn, dataset = pack.make_score_fn(mode="smoke")
    assert dataset is None
    assert callable(smoke_fn)


def test_make_score_fn_for_domain_smoke_matches_direct():
    from crowd_nav.domains import make_score_fn_for_domain
    from crowd_nav.reward_search.scoring import make_smoke_score_fn
    from crowd_nav.reward_search.state import RewardFunction, RewardState

    class _Const(RewardFunction):
        def reset(self) -> None:
            return None

        def compute(self, state: RewardState) -> float:
            return float(state.robot.px)

    pack = load_domain("crowdnav")
    via_pack = make_score_fn_for_domain(pack, mode="smoke")
    direct = make_smoke_score_fn()
    reward = _Const()
    assert float(via_pack(reward, candidate_id="a")) == float(
        direct(reward, candidate_id="a")
    )


def test_make_score_fn_for_domain_unknown_mode():
    from crowd_nav.domains import make_score_fn_for_domain

    pack = load_domain("crowdnav")
    with pytest.raises(ValueError, match="Unknown score1_mode"):
        make_score_fn_for_domain(pack, mode="not_a_mode")


def test_pipeline_default_loads_crowdnav_pack():
    from crowd_nav.reward_search.pipeline import EvoNavPipeline, EvoNavRunConfig

    pipe = EvoNavPipeline(EvoNavRunConfig(fast=True))
    assert pipe.domain_pack.name == "crowdnav"
    assert pipe.domain_pack.seed_reward_source == legacy_prompts.D5_SEED_FUNCTION
    assert pipe.domain_pack.adapter is not None


def test_make_stage_trainers_use_adapter_bridge():
    from crowd_nav.domains import (
        load_domain,
        make_stage2_trainer_for_domain,
        make_stage3_trainer_for_domain,
    )
    from crowd_nav.domains.crowdnav.adapter import (
        CrowdNavStage2Trainer,
        CrowdNavStage3Trainer,
    )

    pack = load_domain("crowdnav")
    s2 = make_stage2_trainer_for_domain(pack, use_stub=True)
    s3 = make_stage3_trainer_for_domain(pack, use_stub=True)
    assert isinstance(s2, CrowdNavStage2Trainer)
    assert isinstance(s3, CrowdNavStage3Trainer)


def test_stub_bridge_matches_direct_stub_metrics():
    """Adapter stub path must rank identically to raw StubPolicyTrainer."""
    from crowd_nav.domains.crowdnav.adapter import make_stage2_trainer
    from crowd_nav.reward_search.evolver import RewardCandidate
    from crowd_nav.reward_search.stage2 import Stage2Config, StubPolicyTrainer

    code = "def compute_reward(state, memory):\n    return 1.0\n"
    cand = RewardCandidate(candidate_id="t", code=code, valid=True)
    cfg = Stage2Config(rounds=1, train_env_steps=8, eval_episodes=1)
    direct = StubPolicyTrainer().train_and_eval(cand, round_index=0, config=cfg)
    bridged = make_stage2_trainer(use_stub=True).train_and_eval(
        cand, round_index=0, config=cfg
    )
    assert bridged.as_dict() == direct.as_dict()


def test_crowdnav_adapter_rejects_bad_stage():
    from crowd_nav.domains.crowdnav.adapter import CrowdNavAdapter
    from crowd_nav.reward_search.evolver import RewardCandidate

    adapter = CrowdNavAdapter()
    cand = RewardCandidate(
        candidate_id="x",
        code="def compute_reward(state, memory):\n    return 0.0\n",
    )
    with pytest.raises(ValueError, match="Unknown stage"):
        adapter.train_and_eval(cand, round_index=0, config=object(), stage="stage99")
