"""Episode aggregation and bootstrap standard error of highway fitness."""

from __future__ import annotations

from domains.highway.metrics import (
    bootstrap_fitness_se,
    episode_fitness_fields,
    highway_fitness,
)


def _episodes(outcomes):
    n = len(outcomes)
    dists = [700.0 if o == "success" else 300.0 for o in outcomes]
    speeds = [22.0 + 0.1 * i for i in range(n)]
    lane_changes = [2] * n
    steps = [40] * n
    return outcomes, dists, speeds, lane_changes, steps


def test_fields_aggregate_rates_and_lane_change_per_step():
    f = episode_fitness_fields(*_episodes(["success", "collision", "off_road", "success"]))
    assert f["SR"] == 0.5 and f["CR"] == 0.25 and f["TR"] == 0.25
    assert f["PL"] == 500.0
    assert abs(f["lane_change_rate"] - 0.05) < 1e-12
    assert f["outcome_unique"] == 3.0


def test_bootstrap_is_reproducible_and_reflects_outcome_mix():
    mixed = _episodes(["success", "collision"] * 15)
    se1 = bootstrap_fitness_se(*mixed, seed=7)
    se2 = bootstrap_fitness_se(*mixed, seed=7)
    assert se1 == se2 > 0.05
    clean = _episodes(["success"] * 30)
    assert bootstrap_fitness_se(*clean, seed=7) < se1


def test_single_episode_has_zero_se():
    assert bootstrap_fitness_se(*_episodes(["success"]), seed=0) == 0.0


def test_fields_feed_highway_fitness():
    f = episode_fitness_fields(*_episodes(["success"] * 9 + ["collision"]))
    assert highway_fitness(f) > 0.0
