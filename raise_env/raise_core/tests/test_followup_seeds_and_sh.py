"""Seed SE threshold, lower median, SH+K>1 rejection."""

from __future__ import annotations

import logging

import pytest

from raise_core.seed_aggregate import (
    MIN_SEEDS_FOR_ACROSS_SEEDS_SE,
    aggregate_seed_fits,
    median_seed_checkpoint,
)


def test_across_seeds_requires_at_least_three(caplog):
    with caplog.at_level(logging.WARNING):
        agg2 = aggregate_seed_fits([0.1, 0.5], episode_se=0.2)
    assert agg2["fitness_se_source"] == "episodes_only"
    assert agg2["n_ok_seeds"] == 2
    assert any("episodes_only" in r.message for r in caplog.records)

    agg3 = aggregate_seed_fits([0.1, 0.5, 0.9], episode_se=0.99)
    assert agg3["fitness_se_source"] == "across_seeds"
    assert agg3["n_ok_seeds"] >= MIN_SEEDS_FOR_ACROSS_SEEDS_SE


def test_lower_median_even_k_records_index():
    runs = [
        {"status": "ok", "fitness": 0.1, "seed_index": 0, "checkpoint_path": "a.zip"},
        {"status": "ok", "fitness": 0.2, "seed_index": 1, "checkpoint_path": "b.zip"},
        {"status": "ok", "fitness": 0.3, "seed_index": 2, "checkpoint_path": "c.zip"},
        {"status": "ok", "fitness": 0.4, "seed_index": 3, "checkpoint_path": "d.zip"},
    ]
    path, idx = median_seed_checkpoint(runs)
    # Sorted by fitness: 0.1,0.2,0.3,0.4 → lower median index (4-1)//2 = 1 → seed 1
    assert path == "b.zip"
    assert idx == 1


def test_successive_halving_rejects_stage2_seeds_gt_1():
    from raise_core.raise_loop.runner import reject_successive_halving_with_multi_seed

    reject_successive_halving_with_multi_seed(False, 3)  # ok
    reject_successive_halving_with_multi_seed(True, 1)  # ok
    with pytest.raises(ValueError, match="successive_halving"):
        reject_successive_halving_with_multi_seed(True, 3)
