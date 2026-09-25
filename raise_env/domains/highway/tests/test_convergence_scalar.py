"""Smoke: scalar breeding prefers fitness; best_ever is monotonic in archive logic."""

from __future__ import annotations

from raise_core.explore import RewardCandidate
from raise_core.raise_loop.evolve_rank import (
    parse_evolve_rank,
    rank_population_for_evolution,
)
from raise_core.selection import candidate_fitness


def _lab(cid: str, *, fit: float, score1: float = 0.5) -> RewardCandidate:
    return RewardCandidate(
        candidate_id=cid,
        code=f"def compute_reward(state, memory):\n    return {hash(cid) % 50}.0\n",
        valid=True,
        score=score1,
        metadata={
            "fitness": fit,
            "selection_scalar": fit,
            "last_metrics": {
                "SR": 1.0 if fit > 0 else 0.0,
                "CR": 0.0 if fit > 0 else 1.0,
                "TR": 0.0,
                "mean_speed": 24.0 if fit > 0 else 25.0,
                "soft_success": 1.0 if fit > 0.5 else 0.0,
                "fitness": fit,
                "domain": "highway",
            },
        },
    )


def test_highway_default_evolve_is_scalar():
    assert parse_evolve_rank("", domain="highway") == "scalar"


def test_bimodal_parent_order_and_best_ever_archive():
    e0 = [_lab("crash", fit=-0.6, score1=0.9), _lab("cruise", fit=0.8, score1=0.3)]
    r0 = rank_population_for_evolution(e0, mode="scalar")
    assert r0[0].candidate_id == "cruise"
    best_ever = candidate_fitness(r0[0])
    assert best_ever == 0.8

    # Next gen: all worse → breeding still puts higher fitness first; archive
    # logic in runner keeps the previous elite when keep_runtime_elite is on.
    e1 = [_lab("crash2", fit=-0.7, score1=0.2), _lab("mid", fit=0.4, score1=0.4)]
    r1 = rank_population_for_evolution(e1, mode="scalar")
    assert r1[0].candidate_id == "mid"
    assert candidate_fitness(r1[0]) < best_ever
