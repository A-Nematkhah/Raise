"""Pareto ranking: auto thresholds + NSGA-II order (best → worst)."""

from __future__ import annotations

from domains.highway.pareto_rank import (
    Metrics,
    calibrate_from_population,
    crowding_distance,
    is_feasible,
    rank_population,
)
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.evolve_rank import rank_population_for_evolution


def test_demo_ranking_order_best_to_worst():
    """balanced → fast_risky → crawler (no hand weights)."""
    pop = [
        Metrics(
            "crawler",
            sr=1.0,
            cr=0.0,
            tr=0.0,
            progress=15.0,
            mean_speed=1.5,
            soft_success=1.0,
        ),
        Metrics(
            "fast_risky",
            sr=0.7,
            cr=0.15,
            tr=0.10,
            progress=700.0,
            mean_speed=23.0,
            soft_success=0.8,
        ),
        Metrics(
            "balanced",
            sr=0.9,
            cr=0.04,
            tr=0.05,
            progress=600.0,
            mean_speed=20.0,
            soft_success=0.85,
        ),
    ]
    ordered = rank_population(pop)
    assert [m.candidate_id for m in ordered] == [
        "balanced",
        "fast_risky",
        "crawler",
    ]


def test_speed_p10_preferred_over_mean_for_feasibility():
    from domains.highway.pareto_rank import Metrics, effective_speed

    m = Metrics(
        "spike",
        sr=1.0,
        cr=0.0,
        tr=0.0,
        progress=100.0,
        mean_speed=20.0,
        soft_success=0.0,
        speed_p10=1.5,
    )
    assert effective_speed(m) == 1.5


def test_stamp_does_not_overwrite_fitness():
    from domains.highway.pareto_rank import Metrics, rank_population, stamp_pareto_ranks
    from raise_core.explore import RewardCandidate

    c = RewardCandidate(
        candidate_id="balanced",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        valid=True,
        metadata={"fitness": 2.5, "selection_scalar": 2.5},
    )
    pop = [
        Metrics("balanced", 0.9, 0.04, 0.05, 600.0, 20.0, 0.85),
        Metrics("crawler", 1.0, 0.0, 0.0, 15.0, 1.5, 1.0),
    ]
    ordered = rank_population(pop)
    stamp_pareto_ranks([c], ordered)
    assert c.metadata["fitness"] == 2.5
    assert c.metadata["pareto_rank"] == 0
    assert "pareto_score" in c.metadata


def test_crowding_does_not_scramble_caller_front():
    front = [
        Metrics("a", 1.0, 0.0, 0.0, 100.0, 20.0, 0.5),
        Metrics("b", 0.9, 0.0, 0.0, 200.0, 10.0, 0.5),
        Metrics("c", 0.8, 0.0, 0.0, 150.0, 15.0, 0.5),
    ]
    before = [m.candidate_id for m in front]
    _ = crowding_distance(front)
    assert [m.candidate_id for m in front] == before


def test_crawler_infeasible_under_population_calib():
    pop = [
        Metrics("crawler", 1.0, 0.0, 0.0, 15.0, 1.5, 1.0),
        Metrics("ok", 0.9, 0.02, 0.02, 600.0, 20.0, 0.85),
        Metrics("ok2", 0.85, 0.03, 0.02, 550.0, 19.0, 0.8),
        Metrics("ok3", 0.88, 0.02, 0.03, 580.0, 21.0, 0.82),
    ]
    ref = calibrate_from_population(pop)
    assert not is_feasible(pop[0], ref)
    assert sum(1 for m in pop[1:] if is_feasible(m, ref)) >= 1


def test_evolve_rank_pareto_orders_candidates():
    def _cand(cid, *, sr, cr, tr, progress, speed, soft, score1=0.5):
        return RewardCandidate(
            candidate_id=cid,
            code=f"def compute_reward(state, memory):\n    return {hash(cid) % 50}.0\n",
            valid=True,
            score=score1,
            metadata={
                "last_metrics": {
                    "SR": sr,
                    "CR": cr,
                    "TR": tr,
                    "PL": progress,
                    "mean_speed": speed,
                    "soft_success": soft,
                    "domain": "highway",
                }
            },
        )

    pop = [
        _cand("crawler", sr=1.0, cr=0.0, tr=0.0, progress=15.0, speed=1.5, soft=1.0),
        _cand(
            "fast_risky",
            sr=0.7,
            cr=0.15,
            tr=0.10,
            progress=700.0,
            speed=23.0,
            soft=0.8,
        ),
        _cand(
            "balanced",
            sr=0.9,
            cr=0.04,
            tr=0.05,
            progress=600.0,
            speed=20.0,
            soft=0.85,
        ),
        _cand("unlab", sr=0, cr=0, tr=0, progress=0, speed=0, soft=0, score1=0.99),
    ]
    # unlabeled has empty-ish metrics — strip last_metrics for true unlabeled
    pop[3].metadata = {}
    ranked = rank_population_for_evolution(pop, mode="pareto")
    ids = [c.candidate_id for c in ranked]
    assert ids[:3] == ["balanced", "fast_risky", "crawler"]
    assert ids[-1] == "unlab"
    assert ranked[0].metadata.get("pareto_rank") == 0
