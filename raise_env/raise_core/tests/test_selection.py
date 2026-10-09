"""Unit tests for best-trained selection."""

from __future__ import annotations

from raise_core.explore import RewardCandidate
from raise_core.selection import (
    candidate_fitness,
    candidate_nav_scalar,
    navigation_scalar,
    pick_best_trained,
)


def test_navigation_scalar():
    assert navigation_scalar(0.68, 0.267, 0.053) == 0.68 - 0.267 - 0.5 * 0.053


def test_pick_best_trained_prefers_higher_scalar():
    weak = RewardCandidate(
        candidate_id="weak",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={
            "last_metrics": {"SR": 0.1, "CR": 0.5, "TR": 0.4},
            "trained_snapshot": True,
        },
    )
    strong = RewardCandidate(
        candidate_id="strong",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        metadata={
            "last_metrics": {"SR": 0.68, "CR": 0.267, "TR": 0.053},
            "trained_snapshot": True,
            "checkpoint_path": "ckpt.pt",
        },
    )
    best = pick_best_trained([weak, strong])
    assert best is not None
    assert best.candidate_id == "strong"
    assert candidate_nav_scalar(best) > candidate_nav_scalar(weak)


def test_candidate_fitness_prefers_stamped_fitness():
    c = RewardCandidate(
        candidate_id="c",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={"fitness": 0.42, "last_metrics": {"SR": 0.1, "CR": 0.5, "TR": 0.4}},
    )
    assert abs(candidate_fitness(c) - 0.42) < 1e-12
