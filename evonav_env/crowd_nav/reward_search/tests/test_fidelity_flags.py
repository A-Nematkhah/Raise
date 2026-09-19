"""Tests for final ranking + K2 unit resolution + mutation prompt alignment."""

from __future__ import annotations

from crowd_nav.domains.crowdnav.prompts import format_d2_mutation
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.ranking import (
    multiobjective_lex_rank,
    pick_candidate_by_ranking,
    produce_final_ranking,
    rank_ids_by_scalar,
)
from crowd_nav.reward_search.stage2 import Stage2Config, resolve_stage2_env_steps


def test_mutation_prompt_targets_underperforming_parent():
    text = format_d2_mutation("def compute_reward(state, memory):\n    return 0.0\n", "weak")
    assert "Underperforming Parent" in text
    assert "high-performing" not in text.lower()
    assert "individual weaknesses" in text


def test_resolve_k2_gradient_steps_matches_update_count():
    cfg = Stage2Config(train_env_steps=8000, k2_unit="gradient_steps", num_steps=5)
    nproc = 4
    env_steps = resolve_stage2_env_steps(cfg, nproc)
    assert env_steps == 8000 * 5 * 4
    updates = env_steps // cfg.num_steps // nproc
    assert updates == 8000


def test_resolve_k2_env_steps_passthrough():
    cfg = Stage2Config(train_env_steps=50_000, k2_unit="env_steps", num_steps=5)
    assert resolve_stage2_env_steps(cfg, 16) == 50_000


def _cand(cid: str, sr: float, cr: float = 0.0, tr: float = 0.0, **extra) -> RewardCandidate:
    md = {"SR": sr, "CR": cr, "TR": tr, "NT": 10.0, "PL": 12.0, "ITR": 5.0, "SD": 0.3}
    md.update(extra)
    return RewardCandidate(
        candidate_id=cid,
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={"last_metrics": md},
    )


def test_scalar_vs_lex_ranking_can_differ():
    # High SR but worse secondary metrics vs slightly lower SR better rest.
    a = _cand("a", sr=0.90, cr=0.20, tr=0.10, NT=30.0, PL=40.0, ITR=20.0, SD=0.1)
    b = _cand("b", sr=0.88, cr=0.02, tr=0.02, NT=12.0, PL=15.0, ITR=3.0, SD=0.5)
    # scalar: a = 0.90-0.20-0.05=0.65; b=0.88-0.02-0.01=0.85 → b first
    assert rank_ids_by_scalar([a, b])[0] == "b"
    lex = multiobjective_lex_rank([a, b])
    assert lex[0] == "a"  # SR dominates lex


def test_produce_final_ranking_llm_seed_falls_back_to_lex():
    pop = [_cand("x", 0.5), _cand("y", 0.9)]
    out = produce_final_ranking(pop, mode="llm", llm=None)
    assert out["mode"] == "llm"
    assert out["fallback"] == "multiobjective_lex"
    assert out["ranking"][0] == "y"


def test_pick_candidate_by_ranking_best_first():
    pop = [_cand("x", 0.5), _cand("y", 0.9)]
    ranking = {"ranking": ["y", "x"]}
    best = pick_candidate_by_ranking(ranking, pop)
    assert best is not None
    assert best.candidate_id == "y"
