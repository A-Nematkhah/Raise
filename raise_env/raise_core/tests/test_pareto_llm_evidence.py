"""LLM evidence / reflection must track Pareto ranking, not highway_fitness."""

from __future__ import annotations

import domains.highway.prompts as hwy_prompts
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from raise_core.explore import RewardCandidate, StageIEvolver
from raise_core.llm import ScriptedLLMClient
from raise_core.raise_loop.evolve_rank import rank_population_pareto
from raise_core.raise_loop.proxy_feedback import (
    evidence_block,
    refresh_highway_evidence_after_pareto,
)
from raise_core.sandbox.validator import RewardValidator


def _metrics(**extra):
    m = {
        "domain": "highway",
        "SR": 0.9,
        "CR": 0.05,
        "TR": 0.0,
        "mean_speed": 20.0,
        "PL": 700.0,
        "mean_progress": 700.0,
        "soft_success": 0.2,
        "lane_change_rate": 0.01,
        "speed_p10": 19.0,
        "speed_p90": 21.0,
        "n_eval_episodes": 20,
    }
    m.update(extra)
    return m


def _cand(cid: str, *, score: float, metrics: dict, fitness: float) -> RewardCandidate:
    fn = RewardValidator().validate_code(D5_SEED_FUNCTION)
    return RewardCandidate(
        candidate_id=cid,
        code=D5_SEED_FUNCTION,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=score,
        metadata={
            "last_metrics": dict(metrics),
            "fitness": float(fitness),
            "selection_scalar": float(fitness),
        },
    )


def test_evidence_block_pareto_not_legacy_fitness():
    md = {
        "pareto_rank": 2,
        "pareto_n": 6,
        "pareto_feasible": True,
        "pareto_v_floor": 17.3,
        "pareto_cr_ceiling": 0.25,
        "pareto_tr_ceiling": 0.05,
    }
    block = evidence_block(_metrics(), score1=0.4, metadata=md)
    low = block.lower()
    assert "Pareto rank" in block
    assert "Auto-calibrated feasibility" in block
    assert "fitness" not in low
    assert "gate" not in low
    assert "v_target" not in low


def test_evidence_block_without_pareto_is_raw_metrics_only():
    block = evidence_block(_metrics(), score1=0.4)
    low = block.lower()
    assert "SR=0.90" in block
    assert "Pareto rank" not in block
    assert "fitness" not in low
    assert "gate" not in low
    assert "v_target" not in low


def test_highway_reflection_orders_by_pareto_not_fitness():
    """Construct a case where fitness order != pareto_rank order."""
    # High fitness but worse Pareto rank (2), vs lower fitness but Pareto-best (0).
    a = _cand(
        "fit_champ",
        score=0.9,
        metrics=_metrics(mean_speed=20.0, SR=0.95),
        fitness=1.5,
    )
    b = _cand(
        "pareto_champ",
        score=0.2,
        metrics=_metrics(mean_speed=24.0, SR=0.85, soft_success=0.8),
        fitness=0.4,
    )
    a.metadata["pareto_rank"] = 2
    a.metadata["pareto_n"] = 2
    a.metadata["pareto_feasible"] = True
    b.metadata["pareto_rank"] = 0
    b.metadata["pareto_n"] = 2
    b.metadata["pareto_feasible"] = True

    evo = StageIEvolver(
        ScriptedLLMClient([D5_SEED_FUNCTION]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.0, "rejected": False})(),
        validator=RewardValidator(),
        prompts=hwy_prompts,
    )
    # Pass in fitness-descending order; reflection must re-sort by pareto_rank.
    note = evo._build_highway_evidence_reflection([a, b], generation=1)
    assert "Population(pareto_rank asc):" in note
    assert "best_speed_among_feasible=" in note
    assert "n_feasible_trend=" in note
    assert "fitness" not in note.lower()
    pos_p = note.find("pareto_champ")
    pos_f = note.find("fit_champ")
    assert 0 <= pos_p < pos_f


def test_refresh_and_reflection_match_pareto_evo_order():
    """After stamp + refresh, reflection order matches ranked_for_evo ids."""
    from domains.highway.pareto_rank import ReferenceStats

    # Feasible survivor vs crash attractor under a fixed lenient ref.
    cruise = _cand(
        "cruise",
        score=0.3,
        metrics=_metrics(
            SR=1.0,
            CR=0.0,
            TR=0.0,
            mean_speed=22.0,
            speed_p10=21.5,
            soft_success=0.5,
            PL=850.0,
        ),
        fitness=1.0,
    )
    crash = _cand(
        "crash",
        score=0.95,
        metrics=_metrics(
            SR=0.0,
            CR=1.0,
            TR=0.0,
            mean_speed=28.0,
            speed_p10=27.0,
            soft_success=0.0,
            PL=50.0,
        ),
        fitness=-0.5,
    )
    score1_ranked = [crash, cruise]
    ref = ReferenceStats(v_floor=10.0, cr_ceiling=0.5, tr_ceiling=0.2)
    ranked_for_evo = rank_population_pareto(score1_ranked, ref=ref)
    refresh_highway_evidence_after_pareto(ranked_for_evo)

    evo_ids = [c.candidate_id for c in ranked_for_evo]
    assert evo_ids[0] == "cruise", evo_ids
    assert cruise.metadata.get("pareto_feasible") is True
    assert crash.metadata.get("pareto_feasible") is False
    fb0 = (ranked_for_evo[0].metadata or {}).get("proxy_feedback", "")
    assert "Pareto rank" in fb0
    assert "fitness" not in fb0.lower()

    evo = StageIEvolver(
        ScriptedLLMClient([D5_SEED_FUNCTION]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.0, "rejected": False})(),
        validator=RewardValidator(),
        prompts=hwy_prompts,
    )
    note = evo._build_reflection(ranked_for_evo, generation=2)
    table = note.split("Population(pareto_rank asc):")[-1]
    assert "cruise:" in table.split(";")[0]


def test_prompts_describe_pareto_not_gate_formula():
    assert "Pareto dominance" in hwy_prompts.D1_SYSTEM_PROMPT
    assert "sigmoid" not in hwy_prompts.D1_SYSTEM_PROMPT.lower()
    assert "V_TARGET" not in hwy_prompts.D1_SYSTEM_PROMPT
    assert "K_GATE" not in hwy_prompts.D1_SYSTEM_PROMPT
    assert "Pareto dominance" in hwy_prompts.D4_EXTERNAL_KNOWLEDGE
    # Seed may still use a numeric traffic heuristic internally — allowed.
    assert "traffic_speed" in hwy_prompts.D5_SEED_FUNCTION


def test_runner_builds_reflection_after_ranked_for_evo():
    """Source-order guard: reflection must use ranked_for_evo, not early ranked."""
    import inspect

    from raise_core.raise_loop import runner as runner_mod

    src = inspect.getsource(runner_mod)
    assert "evolver.reflection = evolver._build_reflection(ranked, generation=g)" not in src
    i_rank = src.index("ranked_for_evo = rank_population_for_evolution")
    i_refl = src.index("evolver.reflection = evolver._build_reflection")
    assert i_rank < i_refl
    assert "ranked_for_evo, generation=g" in src
    assert "refresh_highway_evidence_after_pareto" in src
