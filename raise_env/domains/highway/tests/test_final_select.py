"""Final highway pick: --pareto-front lists front; --best-ever scalar differs."""

from __future__ import annotations

from domains.highway.final_select import (
    crowdnav_history_scalar,
    format_front_table,
    front0_candidates,
    raw_objective_row,
)
from domains.highway.pareto_rank import ReferenceStats
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.evolve_rank import rank_population_pareto


def _cand(cid: str, *, metrics: dict, score: float = 0.5) -> RewardCandidate:
    return RewardCandidate(
        candidate_id=cid,
        code="def compute_reward(state, memory):\n    return 0.0\n",
        valid=True,
        origin="test",
        score=score,
        metadata={
            "last_metrics": {"domain": "highway", **metrics},
            "fitness": 0.0,
        },
    )


def test_best_ever_scalar_picks_crawler_not_balanced():
    crawler = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 2.0,
        "PL": 40.0,
        "mean_progress": 40.0,
        "soft_success": 0.0,
        "speed_p10": 1.8,
    }
    balanced = {
        "SR": 0.85,
        "CR": 0.05,
        "TR": 0.05,
        "mean_speed": 22.0,
        "PL": 800.0,
        "mean_progress": 800.0,
        "soft_success": 0.4,
        "speed_p10": 20.0,
    }
    # Legacy --best-ever scalar ignores speed/progress → crawler wins.
    assert crowdnav_history_scalar(crawler) > crowdnav_history_scalar(balanced)
    assert crowdnav_history_scalar(crawler) == 1.0


def test_pareto_front_lists_balanced_no_silent_winner():
    crawler = _cand(
        "crawler",
        metrics={
            "SR": 1.0,
            "CR": 0.0,
            "TR": 0.0,
            "mean_speed": 2.0,
            "PL": 40.0,
            "mean_progress": 40.0,
            "soft_success": 0.0,
            "speed_p10": 1.8,
        },
        score=0.99,
    )
    balanced = _cand(
        "balanced",
        metrics={
            "SR": 0.85,
            "CR": 0.05,
            "TR": 0.05,
            "mean_speed": 22.0,
            "PL": 800.0,
            "mean_progress": 800.0,
            "soft_success": 0.4,
            "speed_p10": 20.0,
        },
        score=0.2,
    )
    ref = ReferenceStats(v_floor=5.0, cr_ceiling=0.5, tr_ceiling=0.2)
    ranked = rank_population_pareto([crawler, balanced], ref=ref)
    front = front0_candidates(ranked, ref=ref)
    front_ids = {c.candidate_id for c in front}

    assert "balanced" in front_ids
    # Crawler is typically infeasible under v_floor=5 with speed_p10=1.8,
    # or dominated — must not be the sole silent "winner".
    rows = [raw_objective_row(c) for c in front]
    table = format_front_table(rows)
    assert "No automatic winner" in table
    assert "balanced" in table
    # Do not declare a single champion id.
    assert "winner=" not in table.lower()
    assert "best=" not in table.lower()

    # Metadata exposes front membership (not just flat ordinal).
    bal = next(c for c in ranked if c.candidate_id == "balanced")
    assert bal.metadata.get("pareto_front") == 0
    assert bal.metadata.get("pareto_front0") is True


def test_stamp_sets_pareto_front_index():
    from domains.highway.pareto_rank import Metrics, rank_population, stamp_pareto_ranks

    pop_m = [
        Metrics("a", 0.9, 0.05, 0.05, 600.0, 20.0, 0.5, speed_p10=19.0),
        Metrics("b", 0.8, 0.05, 0.05, 700.0, 22.0, 0.6, speed_p10=21.0),
    ]
    ref = ReferenceStats(v_floor=10.0, cr_ceiling=0.5, tr_ceiling=0.2)
    ordered = rank_population(pop_m, ref=ref)
    cands = [
        RewardCandidate(
            candidate_id=m.candidate_id,
            code="def compute_reward(state, memory):\n    return 0.0\n",
            valid=True,
            metadata={"last_metrics": {"domain": "highway"}},
        )
        for m in pop_m
    ]
    stamp_pareto_ranks(cands, ordered, ref=ref)
    for c in cands:
        assert "pareto_front" in c.metadata
        assert "pareto_front0" in c.metadata
        if c.metadata.get("pareto_feasible"):
            assert c.metadata["pareto_front"] is not None
