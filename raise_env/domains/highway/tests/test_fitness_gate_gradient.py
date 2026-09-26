"""Recentered speed gate must keep a real gradient across [V_MIN, V_TARGET]."""

from __future__ import annotations

from domains.highway.metrics import (
    HACK_PENALTY,
    V_FLOOR,
    V_MIN,
    V_TARGET,
    _speed_gate,
    fitness_components,
    highway_fitness,
)
from domains.highway.objective_constants import K_GATE, V_TARGET as VT_CONST


def test_objective_constants_shared():
    assert V_TARGET == VT_CONST
    assert K_GATE == 6.0


def test_gate_gradient_20_vs_25():
    g20 = _speed_gate(20.0)
    g25 = _speed_gate(25.0)
    assert g25 - g20 >= 0.10


def test_gate_not_saturated_at_bounds():
    g_min = _speed_gate(V_MIN)
    g_tgt = _speed_gate(V_TARGET)
    assert abs(g_min - 0.0) < 0.08  # ≈0 at V_MIN
    assert g_tgt <= 0.97  # must not already be saturated at V_TARGET


def test_floor_disqualify_unchanged():
    crawl = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 2.0,
        "PL": 80.0,
        "soft_success": 0.0,
        "speed_p10": 1.5,
        "speed_p90": 2.5,
    }
    assert crawl["speed_p10"] < V_FLOOR
    assert highway_fitness(crawl) == -HACK_PENALTY
    assert fitness_components(crawl)["disqualified"] == 1.0
