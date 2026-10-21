"""Stage III G3-style refine helpers."""

from __future__ import annotations

from raise_core.candidate import RewardCandidate
from raise_core.stage3_refine import (
    merge_refine_populations,
    multi_objective_feedback,
    refine_round_count,
)


def test_refine_round_count_clamps():
    assert refine_round_count(None) == 1
    assert refine_round_count(0) == 1
    assert refine_round_count(3) == 3


def test_multi_objective_feedback_includes_itr_sd():
    cand = RewardCandidate(
        candidate_id="c1",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={
            "last_metrics": {
                "SR": 0.8,
                "CR": 0.1,
                "TR": 0.1,
                "NT": 12.0,
                "PL": 9.0,
                "ITR": 0.05,
                "SD": 0.15,
                "fitness": 0.7,
            }
        },
    )
    text = multi_objective_feedback(cand)
    assert "ITR=0.0500" in text
    assert "SD=0.1500" in text
    assert "danger_freq" in text


def test_merge_keeps_better_fitness():
    parent = RewardCandidate(
        candidate_id="p",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={"fitness": 0.5, "last_metrics": {"fitness": 0.5}},
    )
    child = RewardCandidate(
        candidate_id="p_r1",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        parent_ids=("p",),
        metadata={"fitness": 0.9, "last_metrics": {"fitness": 0.9}},
        origin="stage3_refine",
    )
    worse = RewardCandidate(
        candidate_id="p_r1",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        parent_ids=("p",),
        metadata={"fitness": 0.1, "last_metrics": {"fitness": 0.1}},
        origin="stage3_refine",
    )
    assert merge_refine_populations([parent], [child])[0].candidate_id == "p_r1"
    assert merge_refine_populations([parent], [worse])[0].candidate_id == "p"
