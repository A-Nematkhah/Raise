"""Anti-hacking fitness: speed gate, floor disqualify, holdout preference."""

from __future__ import annotations

from domains.highway.env_wrapper import HOLDOUT_VEHICLES, holdout_env_config
from domains.highway.metrics import (
    HACK_PENALTY,
    V_FLOOR,
    V_MIN,
    V_TARGET,
    attach_fitness,
    fitness_components,
    highway_fitness,
    highway_navigation_scalar,
)


def test_holdout_env_denser_than_default():
    cfg = holdout_env_config()
    assert int(cfg["vehicles_count"]) == int(HOLDOUT_VEHICLES)
    assert int(cfg["vehicles_count"]) > 20


def test_speed_thresholds_match_spec():
    assert abs(V_MIN - 0.4 * V_TARGET) < 1e-9
    assert abs(V_FLOOR - 0.15 * V_TARGET) < 1e-9


def test_fitness_alias_matches_navigation_scalar():
    m = {
        "SR": 0.8,
        "CR": 0.1,
        "TR": 0.0,
        "mean_speed": 22.0,
        "PL": 700.0,
        "soft_success": 0.7,
        "lane_change_rate": 0.05,
        "speed_p10": 20.0,
        "speed_p90": 24.0,
        "outcome_unique": 2,
    }
    assert highway_fitness(m) == highway_navigation_scalar(m)
    out = attach_fitness(dict(m))
    assert "fitness" in out and "selection_scalar" in out
    assert out["fitness"] == out["selection_scalar"] == highway_fitness(m)
    assert out["fitness_gate"] > 0.5
    assert out["fitness_disqualified"] == 0.0


def test_below_floor_hard_disqualify():
    crawl = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 2.0,
        "PL": 80.0,
        "soft_success": 0.0,
        "lane_change_rate": 0.0,
        "speed_p10": 1.5,
        "speed_p90": 2.5,
        "outcome_unique": 1,
    }
    assert crawl["speed_p10"] < V_FLOOR
    assert highway_fitness(crawl) == -HACK_PENALTY
    assert fitness_components(crawl)["disqualified"] == 1.0


def test_gate_suppresses_low_speed_survival():
    # Same perfect SR, but slow vs cruise — gate + low_speed should dominate.
    slow = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 8.0,
        "PL": 300.0,
        "soft_success": 0.0,
        "lane_change_rate": 0.05,
        "speed_p10": 7.5,  # below v_min=10, above v_floor
        "speed_p90": 8.5,
        "outcome_unique": 2,
    }
    cruise = {
        **slow,
        "mean_speed": 22.0,
        "PL": 800.0,
        "soft_success": 1.0,
        "speed_p10": 20.0,
        "speed_p90": 24.0,
    }
    assert V_FLOOR < slow["speed_p10"] < V_MIN
    assert fitness_components(slow)["gate"] < 0.5
    assert fitness_components(cruise)["gate"] > 0.9
    assert highway_fitness(cruise) > highway_fitness(slow) + 0.5


def test_scalar_prefers_holdout_when_nested():
    metrics = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 22.0,
        "PL": 800.0,
        "soft_success": 1.0,
        "lane_change_rate": 0.1,
        "speed_p10": 21.0,
        "speed_p90": 23.0,
        "outcome_unique": 2,
        "holdout": {
            "SR": 0.4,
            "CR": 0.5,
            "TR": 0.1,
            "mean_speed": 18.0,
            "PL": 400.0,
            "soft_success": 0.1,
            "lane_change_rate": 0.0,
            "speed_p10": 17.5,
            "speed_p90": 18.2,
            "outcome_unique": 2,
        },
    }
    train_only = {k: v for k, v in metrics.items() if k != "holdout"}
    assert highway_fitness(metrics) < highway_fitness(train_only)


def test_scalar_penalizes_lag_behind_traffic():
    cruise = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 22.0,
        "PL": 850.0,
        "soft_success": 1.0,
        "lane_change_rate": 0.05,
        "speed_p10": 20.0,
        "speed_p90": 24.0,
        "outcome_unique": 2,
    }
    lag = {
        **cruise,
        "mean_speed": 14.0,
        "PL": 500.0,
        "soft_success": 0.0,
        "speed_p10": 13.5,
        "speed_p90": 14.5,
    }
    assert highway_fitness(cruise) > highway_fitness(lag) + 0.15


def test_scalar_penalizes_constant_cruise_degeneracy():
    varied = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 22.0,
        "PL": 800.0,
        "soft_success": 1.0,
        "lane_change_rate": 0.08,
        "speed_p10": 18.0,
        "speed_p90": 26.0,
        "outcome_unique": 2,
    }
    flat = {
        **varied,
        "lane_change_rate": 0.0,
        "speed_p10": 22.0,
        "speed_p90": 22.05,
        "outcome_unique": 1,
    }
    assert highway_fitness(varied) > highway_fitness(flat)
