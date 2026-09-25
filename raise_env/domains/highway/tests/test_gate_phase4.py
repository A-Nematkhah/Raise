"""Phase 4: hard-gate readiness + Stage III elite assembly."""

from __future__ import annotations

import pytest

from raise_core.explore import RewardCandidate
from raise_core.raise_loop.gate_policy import (
    assemble_stage3_population,
    epoch_population_stats,
    score1_elite_ok,
    surrogate_hard_gate_ready,
)
from raise_core.raise_loop.report import format_epoch_summary


def _cand(cid: str, *, score=None, soft=None, scalar=None, code=None):
    md = {}
    if soft is not None or scalar is not None:
        fit = float(scalar if scalar is not None else 0.0)
        md["last_metrics"] = {
            "SR": 1.0 if (soft or 0) > 0 else 0.5,
            "CR": 0.0,
            "TR": 0.0,
            "PL": 500.0,
            "mean_speed": 20.0 if (soft or 0) > 0.2 else 8.0,
            "soft_success": float(soft or 0.0),
            "domain": "highway",
            "fitness": fit,
            "selection_scalar": fit,
        }
        md["fitness"] = fit
        md["selection_scalar"] = fit
    return RewardCandidate(
        candidate_id=cid,
        code=code
        or f"def compute_reward(state, memory):\n    return {hash(cid) % 97}.0\n",
        valid=True,
        score=score,
        metadata=md,
    )


def test_hard_gate_ready_by_labels():
    ok, reason = surrogate_hard_gate_ready(n_labeled=16, min_labels=16)
    assert ok
    assert "n_labeled" in reason


def test_hard_gate_ready_by_mae():
    fit = {
        "per_target": {
            "SR": {"mae": 0.1},
            "CR": {"mae": 0.2},
            "TR": {"mae": 0.15},
            # Huge continuous MAE must NOT block the rate-based early gate.
            "mean_speed": {"mae": 20.0},
            "mean_progress": {"mae": 500.0},
        }
    }
    ok, reason = surrogate_hard_gate_ready(
        n_labeled=8, min_labels=16, fit_metrics=fit, max_val_mae=0.35
    )
    assert ok
    assert "val_mae" in reason


def test_hard_gate_soft_when_mae_high():
    fit = {"per_target": {"SR": {"mae": 0.9}, "CR": {"mae": 0.8}}}
    ok, reason = surrogate_hard_gate_ready(
        n_labeled=8, min_labels=16, fit_metrics=fit, max_val_mae=0.35
    )
    assert not ok
    assert "n_labeled" in reason or "val_mae" in reason


def test_assemble_forces_elites_and_filters_weak_score1():
    weak_s1 = _cand("s1", score=0.9, soft=0.0, scalar=-0.2)
    strong_s2 = _cand("s2", score=0.4, soft=0.8, scalar=0.9)
    mid = _cand("m", score=0.5, soft=0.3, scalar=0.4)
    out, report = assemble_stage3_population(
        [mid],
        full_pool=[weak_s1, strong_s2, mid],
        best_s2=strong_s2,
        best_s1=weak_s1,
        domain="highway",
    )
    ids = {c.candidate_id for c in out}
    assert "m" in ids and "s2" in ids
    assert "s1" not in ids
    assert report["score1_best_added"] is False


def test_assemble_adds_strong_score1():
    # High soft_success but lower scalar than s2 → only Score1 path forces it.
    strong_s1 = _cand("s1", score=0.95, soft=0.85, scalar=0.2)
    s2 = _cand("s2", score=0.4, soft=0.5, scalar=0.9)
    out, report = assemble_stage3_population(
        [s2],
        full_pool=[strong_s1, s2],
        best_s2=s2,
        best_s1=strong_s1,
        domain="highway",
    )
    assert any(c.candidate_id == "s1" for c in out)
    assert report["score1_best_added"] is True


def test_score1_elite_ok():
    assert score1_elite_ok(_cand("a", soft=0.5, scalar=0.1))
    assert not score1_elite_ok(
        _cand("b", soft=0.0, scalar=-0.5), soft_success_min=0.25
    )


def test_epoch_stats_and_summary_line():
    pop = [
        _cand("a", score=0.2, soft=0.1, scalar=0.0),
        _cand("b", score=0.8, soft=0.9, scalar=1.0),
    ]
    stats = epoch_population_stats(pop)
    assert stats["score1_spread"] == pytest.approx(0.6)
    line = format_epoch_summary(
        {
            "epoch": 1,
            "best_id": "b",
            "best_score1": 0.8,
            "labeled_ids": ["a"],
            "n_population": 2,
            "gate": {"soft": True, "reason": "n_labeled=3<16"},
            "al": {"enabled": False},
            "n_labeled_dataset": 3,
            "refit": False,
            "n_label_ok": 1,
            "n_label_failed": 0,
            "stats": stats,
        }
    )
    assert "spread=" in line
    assert "softμ=" in line
