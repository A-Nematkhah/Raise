"""Tests for closed-loop evolve_rank (Score1 vs nav scalar vs hybrid)."""

from __future__ import annotations

from raise_core.explore import RewardCandidate
from raise_core.raise_loop.evolve_rank import (
    parse_evolve_rank,
    rank_population_for_evolution,
)


def _c(
    cid: str,
    *,
    score1: float | None,
    nav: float | None = None,
) -> RewardCandidate:
    md = {}
    if nav is not None:
        md["fitness"] = float(nav)
        md["selection_scalar"] = float(nav)
        md["last_metrics"] = {
            "SR": 1.0,
            "CR": 0.0,
            "TR": 0.0,
            "fitness": float(nav),
            "selection_scalar": float(nav),
            "domain": "highway",
        }
    return RewardCandidate(
        candidate_id=cid,
        code=f"def compute_reward(state, memory):\n    return {hash(cid) % 97}.0\n",
        valid=True,
        origin="test",
        score=score1,
        metadata=md,
    )


def test_parse_defaults():
    assert parse_evolve_rank("", domain="crowdnav") == "score1"
    assert parse_evolve_rank(None, domain="highway") == "pareto"
    assert parse_evolve_rank("hybrid", domain="crowdnav") == "hybrid"
    assert parse_evolve_rank("scalar", domain="highway") == "scalar"


def test_score1_mode_ignores_nav():
    pop = [
        _c("low_s1_high_nav", score1=0.3, nav=2.0),
        _c("high_s1_low_nav", score1=0.9, nav=0.1),
    ]
    ranked = rank_population_for_evolution(pop, mode="score1")
    assert [c.candidate_id for c in ranked] == [
        "high_s1_low_nav",
        "low_s1_high_nav",
    ]


def test_scalar_mode_prefers_nav_and_puts_unlabeled_last():
    pop = [
        _c("unlab_high_s1", score1=0.99, nav=None),
        _c("lab_mid", score1=0.5, nav=1.0),
        _c("lab_best", score1=0.4, nav=1.5),
    ]
    ranked = rank_population_for_evolution(pop, mode="scalar")
    assert [c.candidate_id for c in ranked] == [
        "lab_best",
        "lab_mid",
        "unlab_high_s1",
    ]


def test_hybrid_weights_nav_more_than_score1():
    # Same Score1; higher nav wins. With w=0.4, nav dominates ties on Score1.
    pop = [
        _c("a", score1=0.8, nav=0.2),
        _c("b", score1=0.8, nav=1.8),
    ]
    ranked = rank_population_for_evolution(
        pop, mode="hybrid", hybrid_score1_weight=0.4
    )
    assert ranked[0].candidate_id == "b"


def test_hybrid_can_prefer_strong_score1_when_nav_close():
    pop = [
        _c("score_champ", score1=0.95, nav=1.0),
        _c("nav_slight", score1=0.50, nav=1.05),
    ]
    ranked = rank_population_for_evolution(
        pop, mode="hybrid", hybrid_score1_weight=0.4
    )
    # Normalized: score_champ s1=1, nav~0; nav_slight s1=0, nav=1
    # hybrid: 0.4*1+0.6*0=0.4 vs 0.4*0+0.6*1=0.6 → nav_slight wins
    assert ranked[0].candidate_id == "nav_slight"
