"""Tests for closed-loop parent order (fitness first, Score1 for the rest)."""

from __future__ import annotations

from raise_core.candidate import RewardCandidate
from raise_core.raise_loop.evolve_rank import rank_population_for_evolution


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
            "domain": "crowdnav",
        }
    return RewardCandidate(
        candidate_id=cid,
        code=f"def compute_reward(state, memory):\n    return {hash(cid) % 97}.0\n",
        valid=True,
        origin="test",
        score=score1,
        metadata=md,
    )


def test_prefers_cruise_fitness_over_crash():
    """Bimodal run fingerprint: cruise@20 soft=1 must beat crash@25 for parents."""
    cruise = _c("cruise", score1=0.3, nav=1.039)
    crash = _c("crash", score1=0.92, nav=-0.632)
    ranked = rank_population_for_evolution([crash, cruise])
    assert [c.candidate_id for c in ranked] == ["cruise", "crash"]


def test_fitness_first_and_unlabeled_last():
    pop = [
        _c("unlab_high_s1", score1=0.99, nav=None),
        _c("lab_mid", score1=0.5, nav=1.0),
        _c("lab_best", score1=0.4, nav=1.5),
    ]
    ranked = rank_population_for_evolution(pop)
    assert [c.candidate_id for c in ranked] == [
        "lab_best",
        "lab_mid",
        "unlab_high_s1",
    ]


def test_score1_breaks_fitness_ties_and_orders_unlabeled():
    pop = [
        _c("tie_low_s1", score1=0.2, nav=1.0),
        _c("unlab_low", score1=0.1),
        _c("tie_high_s1", score1=0.8, nav=1.0),
        _c("unlab_high", score1=0.9),
        _c("unlab_rejected", score1=None),
    ]
    ranked = rank_population_for_evolution(pop)
    assert [c.candidate_id for c in ranked] == [
        "tie_high_s1",
        "tie_low_s1",
        "unlab_high",
        "unlab_low",
        "unlab_rejected",
    ]
