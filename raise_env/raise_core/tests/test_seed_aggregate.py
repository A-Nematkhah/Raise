"""K-seed fitness aggregation and median checkpoint selection."""

from __future__ import annotations

import math
import statistics

from raise_core.seed_aggregate import (
    aggregate_seed_fits,
    median_seed_checkpoint,
    partial_seeds_flag,
)
from raise_core.stats import fitness_standard_error


def test_aggregate_seed_fits_k3_mean_and_across_seeds_se():
    fits = [0.1, 0.4, 0.7]
    agg = aggregate_seed_fits(fits, episode_se=0.99)
    assert agg["n_ok_seeds"] == 3
    assert agg["fitness"] == statistics.fmean(fits)
    assert agg["fitness_se_source"] == "across_seeds"
    expected_se = fitness_standard_error(fits)
    assert math.isclose(agg["fitness_se"], expected_se, rel_tol=1e-9)
    assert agg["seed_fits"] == fits


def test_aggregate_seed_fits_single_keeps_episode_se():
    agg = aggregate_seed_fits([0.42], episode_se=0.15)
    assert agg["fitness"] == 0.42
    assert agg["fitness_se"] == 0.15
    assert agg["fitness_se_source"] == "episodes_only"


def test_median_seed_checkpoint_picks_middle_not_max():
    runs = [
        {"seed_index": 0, "status": "ok", "fitness": 0.9, "checkpoint_path": "max.zip"},
        {"seed_index": 1, "status": "ok", "fitness": 0.2, "checkpoint_path": "lo.zip"},
        {"seed_index": 2, "status": "ok", "fitness": 0.5, "checkpoint_path": "mid.zip"},
    ]
    path, idx = median_seed_checkpoint(runs)
    assert path == "mid.zip"
    assert idx == 2


def test_median_seed_checkpoint_even_count_uses_lower_middle():
    runs = [
        {"seed_index": 0, "status": "ok", "fitness": 0.1, "checkpoint_path": "a.zip"},
        {"seed_index": 1, "status": "ok", "fitness": 0.2, "checkpoint_path": "b.zip"},
        {"seed_index": 2, "status": "ok", "fitness": 0.3, "checkpoint_path": "c.zip"},
        {"seed_index": 3, "status": "ok", "fitness": 0.4, "checkpoint_path": "d.zip"},
    ]
    path, idx = median_seed_checkpoint(runs)
    # Sorted by fitness: a,b,c,d → lower middle is b (index 1 of 0..3).
    assert path == "b.zip"
    assert idx == 1


def test_partial_seeds_flag():
    assert partial_seeds_flag(n_ok=2, n_seeds=3) is True
    assert partial_seeds_flag(n_ok=3, n_seeds=3) is False
    assert partial_seeds_flag(n_ok=1, n_seeds=1) is False
