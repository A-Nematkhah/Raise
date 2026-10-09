"""Overtake counting + overtake metrics in the LLM evidence block."""

from __future__ import annotations

from domains.highway.overtake import OvertakeTracker
from raise_core.raise_loop.proxy_feedback import evidence_block


def test_evidence_lists_overtake_metrics_and_fitness():
    text = evidence_block(
        {
            "SR": 1.0,
            "CR": 0.0,
            "TR": 0.0,
            "mean_speed": 20.0,
            "PL": 100.0,
            "lane_change_rate": 0.01,
            "overtakes_per_km": 1.25,
            "overtake_episode_frac": 0.4,
            "fitness": 0.75,
        },
        score1=0.5,
    )
    assert "overtakes_per_km=1.25" in text
    assert "overtake_episode_frac=0.40" in text
    assert "lane_change_rate=0.010" in text
    assert "fitness=0.750" in text
    assert "Score1=0.500" in text


def test_overtake_tracker_counts_pass():
    class _V:
        def __init__(self, x: float) -> None:
            self.position = (x, 0.0)

    class _Road:
        def __init__(self, vehicles):
            self.vehicles = vehicles

    class _Env:
        def __init__(self):
            self.vehicle = _V(0.0)
            self.road = _Road([self.vehicle, _V(10.0)])
            self.unwrapped = self

    env = _Env()
    other = env.road.vehicles[1]
    tr = OvertakeTracker()
    tr.observe(env)
    assert tr.passes == 0
    # Ego moves past the lead vehicle.
    env.vehicle.position = (15.0, 0.0)
    other.position = (10.0, 0.0)
    tr.observe(env)
    assert tr.passes == 1

    # Swapping back and forth must not count the same vehicle repeatedly.
    for _ in range(3):
        other.position = (env.vehicle.position[0] + 5.0, 0.0)
        tr.observe(env)
        assert tr.passes == 0
        other.position = (env.vehicle.position[0] - 5.0, 0.0)
        tr.observe(env)
        assert tr.passes == 1
