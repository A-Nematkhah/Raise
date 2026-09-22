"""Unit tests for best-ever navigation scalar selection."""

from __future__ import annotations

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.pipeline import EvoNavPipeline
from crowd_nav.reward_search.selection import (
    candidate_nav_scalar,
    navigation_scalar,
    pick_best_trained,
)
from crowd_nav.reward_search.stage2 import ProxyMetrics, Stage2RoundRecord


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


def test_pipeline_best_by_ever_uses_snapshots_not_last_round_only():
    """Regression: R0 peak must beat a worse final-round candidate."""
    r0 = RewardCandidate(
        candidate_id="mut_0060_v2",
        code="def compute_reward(state, memory):\n    return 0.5\n",
        metadata={
            "last_metrics": {"SR": 0.68, "CR": 0.267, "TR": 0.053},
            "checkpoint_path": "r00.pt",
            "trained_snapshot": True,
        },
    )
    r1 = RewardCandidate(
        candidate_id="mut_0060_v3",
        code="def compute_reward(state, memory):\n    return 0.1\n",
        metadata={
            "last_metrics": {"SR": 0.467, "CR": 0.453, "TR": 0.08},
            "checkpoint_path": "r01.pt",
            "trained_snapshot": True,
        },
    )
    history = [
        Stage2RoundRecord(
            round_index=0,
            candidate_id="mut_0060_v2",
            metrics=ProxyMetrics(sr=0.68, cr=0.267, tr=0.053),
            refined=False,
            kept_previous=True,
        ),
        Stage2RoundRecord(
            round_index=1,
            candidate_id="mut_0060_v2",
            metrics=ProxyMetrics(sr=0.467, cr=0.453, tr=0.08),
            refined=True,
            kept_previous=False,
        ),
    ]
    best = EvoNavPipeline._best_by_ever_metrics(
        [r1], history, trained_snapshots=[r0, r1]
    )
    assert best.candidate_id == "mut_0060_v2"
    assert abs(candidate_nav_scalar(best) - navigation_scalar(0.68, 0.267, 0.053)) < 1e-9
