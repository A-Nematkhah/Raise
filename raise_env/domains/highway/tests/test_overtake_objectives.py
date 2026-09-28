"""Overtake counting + lane/overtake Pareto objectives."""

from __future__ import annotations

from domains.highway.pareto_rank import (
    Metrics,
    ParetoObjectives,
    dominates,
    rank_population,
    survival_only_reference,
    _objectives,
)
from raise_core.raise_loop.proxy_feedback import evidence_block


def _m(
    cid: str,
    *,
    sr: float = 1.0,
    cr: float = 0.0,
    progress: float = 100.0,
    speed: float = 20.0,
    lane: float = 0.0,
    ovt: float = 0.0,
) -> Metrics:
    return Metrics(
        candidate_id=cid,
        sr=sr,
        cr=cr,
        tr=0.0,
        progress=progress,
        mean_speed=speed,
        soft_success=0.0,
        speed_p10=speed,
        lane_change_rate=lane,
        overtakes_per_km=ovt,
    )


def test_objectives_include_lane_and_overtake_when_enabled():
    m = _m("x", lane=0.02, ovt=1.5)
    bare = ParetoObjectives(progress=False, lane_change=False, overtake=False)
    assert list(_objectives(m, bare)) == [1.0, -0.0, -0.0, 20.0]
    full = ParetoObjectives(progress=True, lane_change=True, overtake=True)
    assert list(_objectives(m, full)) == [1.0, -0.0, -0.0, 100.0, 20.0, 0.02, 1.5]
    assert full.names()[-2:] == ("lane_change_rate", "overtakes_per_km")


def test_crawler_dominated_by_safer_overtaker_on_overtake_objective():
    crawl = _m("crawl", sr=1.0, cr=0.0, speed=20.0, progress=800.0, lane=0.0, ovt=0.0)
    pass_safe = _m(
        "pass", sr=1.0, cr=0.0, speed=20.0, progress=800.0, lane=0.01, ovt=2.0
    )
    objs = ParetoObjectives(progress=True, lane_change=True, overtake=True)
    assert dominates(pass_safe, crawl, objs)
    assert not dominates(crawl, pass_safe, objs)


def test_rank_prefers_overtaker_when_otherwise_tied():
    crawl = _m("crawl", sr=1.0, speed=20.0, progress=800.0, ovt=0.0)
    pass_safe = _m("pass", sr=1.0, speed=20.0, progress=800.0, ovt=3.0)
    ref = survival_only_reference()
    objs = ParetoObjectives(progress=True, lane_change=True, overtake=True)
    ordered = rank_population([crawl, pass_safe], ref=ref, objectives=objs)
    assert ordered[0].candidate_id == "pass"


def test_evidence_lists_overtake_metrics_and_objectives():
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
        },
        metadata={
            "pareto_rank": 0,
            "pareto_n": 2,
            "pareto_feasible": True,
            "pareto_require_survival": True,
            "pareto_calibration_source": "no_speed_floor",
            "pareto_objectives": [
                "SR",
                "-CR",
                "-TR",
                "progress",
                "mean_speed",
                "lane_change_rate",
                "overtakes_per_km",
            ],
        },
    )
    assert "overtakes_per_km=1.25" in text
    assert "overtake_episode_frac=0.40" in text
    assert "lane_change_rate" in text
    assert "overtakes_per_km" in text.split("Pareto objectives:")[-1]


def test_overtake_tracker_counts_pass():
    from domains.highway.overtake import OvertakeTracker

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
    assert tr.passed_by == 0
