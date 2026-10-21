"""paired_difference duplicate averaging and cross-seed statistical ties."""

from __future__ import annotations

import math

from raise_core.candidate import RewardCandidate
from raise_core.selection import statistical_ties
from raise_core.stats import paired_difference


def test_paired_difference_averages_duplicate_cases():
    # Case 0 appears twice in each list; average before differencing.
    eps_a = [
        {"case": 0, "selection_score": 0.0},
        {"case": 0, "selection_score": 2.0},  # mean 1.0
        {"case": 1, "selection_score": 1.0},
        {"case": 2, "selection_score": 2.0},
    ]
    eps_b = [
        {"case": 0, "selection_score": 1.0},
        {"case": 0, "selection_score": 3.0},  # mean 2.0
        {"case": 1, "selection_score": 2.0},
        {"case": 2, "selection_score": 3.0},
    ]
    paired = paired_difference(eps_a, eps_b, key="selection_score")
    assert paired is not None
    assert paired["n"] == 3.0
    # Diffs: (2-1), (2-1), (3-2) → mean 1.0
    assert math.isclose(paired["mean"], 1.0, rel_tol=1e-9)


def _cand(
    cid: str,
    fitness: float,
    se: float,
    *,
    se_source: str = "episodes_only",
    n_seeds: int = 1,
    episodes=None,
) -> RewardCandidate:
    last: dict = {
        "fitness": fitness,
        "fitness_se": se,
        "fitness_se_source": se_source,
    }
    if episodes is not None:
        last["holdout_episodes"] = list(episodes)
    md = {
        "fitness": fitness,
        "last_metrics": last,
        "stage3_n_seeds": n_seeds,
    }
    return RewardCandidate(
        candidate_id=cid,
        code="def compute_reward(state, memory):\n    return 0.0\n",
        valid=True,
        metadata=md,
    )


def test_statistical_ties_use_cross_seed_se_band():
    # Multi-seed: must not use paired episodes; band = z * sqrt(se_a^2 + se_b^2).
    best = _cand("best", 1.0, 0.1, se_source="across_seeds", n_seeds=3)
    near = _cand("near", 0.85, 0.1, se_source="across_seeds", n_seeds=3)
    far = _cand("far", 0.0, 0.1, se_source="across_seeds", n_seeds=3)
    # z=2 → band = 2 * sqrt(0.02) ≈ 0.282; |1-0.85|=0.15 ≤ band; |1-0|=1 > band.
    ties = statistical_ties([best, near, far], best, z=2.0)
    assert ties == ["near"]


def test_statistical_ties_paired_when_single_seed_matching_cases():
    # Varying per-case gaps → positive paired SE; gap stays inside z=2 band.
    eps_best = [
        {"case": 0, "selection_score": 1.0},
        {"case": 1, "selection_score": 0.0},
        {"case": 2, "selection_score": 1.0},
        {"case": 3, "selection_score": 0.0},
    ]
    eps_near = [
        {"case": 0, "selection_score": 0.7},
        {"case": 1, "selection_score": 0.2},
        {"case": 2, "selection_score": 0.8},
        {"case": 3, "selection_score": 0.1},
    ]
    # diffs (best-near): 0.3, -0.2, 0.2, -0.1 → mean 0.05, se > 0
    best = _cand(
        "best", 0.5, 0.5, se_source="episodes_only", n_seeds=1, episodes=eps_best
    )
    near = _cand(
        "near", 0.45, 0.5, se_source="episodes_only", n_seeds=1, episodes=eps_near
    )
    ties = statistical_ties([best, near], best, z=2.0)
    assert ties == ["near"]
