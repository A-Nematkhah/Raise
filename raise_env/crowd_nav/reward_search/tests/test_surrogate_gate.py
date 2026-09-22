"""Unit tests for surrogate Stage-III gate + predict helpers."""

from __future__ import annotations

from types import SimpleNamespace

from crowd_nav.reward_search.surrogate.gate import gate_population


def _cand(cid: str):
    return SimpleNamespace(candidate_id=cid, metadata={})


def test_gate_drops_only_confident_weak():
    pop = [_cand("a"), _cand("b"), _cand("c"), _cand("d")]
    preds = [
        {"candidate_id": "a", "quality": 0.9, "uncertainty": 0.01},
        {"candidate_id": "b", "quality": 0.1, "uncertainty": 0.01},  # drop
        {"candidate_id": "c", "quality": 0.05, "uncertainty": 0.9},  # uncertain → keep
        {"candidate_id": "d", "quality": 0.2, "uncertainty": 0.01},  # drop if fraction allows
    ]
    kept, report = gate_population(
        pop,
        preds,
        drop_fraction=0.5,
        max_uncertainty_to_drop=0.15,
        min_keep=2,
    )
    kept_ids = {c.candidate_id for c in kept}
    assert "a" in kept_ids
    assert "c" in kept_ids  # high uncertainty protected
    assert report["n_kept"] >= 2
    assert report["n_dropped"] >= 1
    assert "b" in report["dropped_ids"] or "d" in report["dropped_ids"]


def test_gate_respects_min_keep():
    pop = [_cand("a"), _cand("b")]
    preds = [
        {"candidate_id": "a", "quality": 0.0, "uncertainty": 0.0},
        {"candidate_id": "b", "quality": 0.1, "uncertainty": 0.0},
    ]
    kept, report = gate_population(
        pop, preds, drop_fraction=1.0, max_uncertainty_to_drop=1.0, min_keep=2
    )
    assert len(kept) == 2
    assert report["n_dropped"] == 0
