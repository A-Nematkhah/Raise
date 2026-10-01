"""Fixes for population gear-endpoint ref, SR drop, statistical dominance."""

from __future__ import annotations

import numpy as np

from domains.highway.pareto_rank import (
    Metrics,
    ParetoObjectives,
    bernoulli_se,
    calibrate_from_gear_endpoints,
    calibrate_from_population,
    dominates,
    is_feasible,
    rank_population,
    reference_for_mode,
    _objectives,
)


def _m(
    cid: str,
    *,
    sr: float,
    cr: float = 0.0,
    tr: float = 0.0,
    progress: float = 100.0,
    speed: float = 20.0,
    n: float = 20.0,
    lane: float = 0.0,
    ovt: float = 0.0,
) -> Metrics:
    return Metrics(
        candidate_id=cid,
        sr=sr,
        cr=cr,
        tr=tr,
        progress=progress,
        mean_speed=speed,
        soft_success=0.0,
        speed_p10=speed,
        lane_change_rate=lane,
        overtakes_per_km=ovt,
        n_eval_episodes=n,
    )


def test_default_objectives_drop_redundant_sr():
    m = _m("x", sr=1.0, cr=0.0, progress=500.0, speed=20.0)
    # −CR, −TR, progress, mean_speed
    assert len(_objectives(m)) == 4
    assert list(_objectives(m))[0] == -0.0
    with_sr = ParetoObjectives(include_sr=True)
    assert len(_objectives(m, with_sr)) == 5
    assert _objectives(m, with_sr)[0] == 1.0


def test_noise_margin_blocks_tiny_sr_difference():
    # Same CR/TR/speed/progress; SR 0.50 vs 0.55 with n=20 → within 2×SE.
    objs = ParetoObjectives(include_sr=True, progress=True)
    # Without meaningful gap beyond SE, neither should dominate on SR alone
    # when CR also differs only slightly — use identical CR and only SR gap.
    a2 = _m("a2", sr=0.55, cr=0.2, speed=20.0, progress=100.0, n=20)
    b2 = _m("b2", sr=0.50, cr=0.2, speed=20.0, progress=100.0, n=20)
    se = 2.0 * max(bernoulli_se(0.55, 20), bernoulli_se(0.50, 20))
    assert abs(0.55 - 0.50) < se  # gap inside margin
    assert not dominates(a2, b2, objs)
    assert not dominates(b2, a2, objs)


def test_large_cr_gap_still_dominates():
    safe = _m("safe", sr=1.0, cr=0.0, speed=20.0, progress=800.0, n=20)
    crash = _m("crash", sr=0.2, cr=0.8, speed=20.0, progress=800.0, n=20)
    objs = ParetoObjectives(progress=True)
    assert dominates(safe, crash, objs)


def test_se_margin_keeps_noisy_survivor_feasible():
    ref = calibrate_from_population(
        [
            _m("a", sr=1.0, cr=0.0, speed=20.1, n=20),
            _m("b", sr=1.0, cr=0.0, speed=20.1, n=20),
            _m("c", sr=0.9, cr=0.1, speed=22.0, n=20),
        ]
    )
    assert ref.se_margin is True
    # Candidate CR slightly above ceiling but within 2×SE.
    cand = _m("noisy", sr=0.85, cr=float(ref.cr_ceiling) + 0.02, speed=22.0, n=20)
    assert is_feasible(cand, ref)


def test_gear_endpoint_calibration_usable():
    # Synthetic: SLOWER slower+safer, FASTER faster+crashier.
    slow = np.full(50, 20.0)
    fast = np.full(50, 28.0)
    ref = calibrate_from_gear_endpoints(slow, 0.1, 0.0, fast, 0.6, 0.0)
    assert ref.source == "gear_endpoints_slower_faster"
    assert ref.se_margin is True
    assert ref.v_floor == 20.0
    assert ref.cr_ceiling == 0.6
    crawl = _m("crawl", sr=1.0, cr=0.0, speed=20.0, n=20)
    assert is_feasible(crawl, ref)


def test_population_mode_prefers_gear_ref_over_percentiles():
    gear = calibrate_from_gear_endpoints(
        np.full(20, 20.0), 0.15, 0.0, np.full(20, 28.0), 0.7, 0.0
    )
    pop = [_m("c", sr=1.0, cr=0.0, speed=20.1, n=20)]
    ref = reference_for_mode("population", pop, gear_ref=gear)
    assert ref.source == "gear_endpoints_slower_faster"
    assert ref.v_floor == 20.0


def test_no_speed_floor_unchanged():
    ref = reference_for_mode("no_speed_floor", [])
    assert ref.require_survival and ref.v_floor is None
    assert is_feasible(_m("ok", sr=0.1, cr=0.9, speed=1.0), ref)
    assert not is_feasible(_m("dead", sr=0.0, cr=0.0, speed=30.0), ref)


def test_rank_with_gear_ref_keeps_crawlers_above_crashers():
    """Regression: under gear-endpoint ref, SR=1 crawlers beat always-crash."""
    gear = calibrate_from_gear_endpoints(
        np.full(30, 20.0), 0.2, 0.0, np.full(30, 28.0), 0.8, 0.0
    )
    pop = [
        _m("crawl", sr=1.0, cr=0.0, speed=20.1, progress=800.0, n=20),
        _m("crash", sr=0.0, cr=1.0, speed=24.0, progress=300.0, n=20),
        _m("fast", sr=0.2, cr=0.8, speed=24.0, progress=500.0, n=20),
    ]
    ordered = rank_population(pop, ref=gear, objectives=ParetoObjectives())
    assert ordered[0].candidate_id == "crawl"
    assert is_feasible(pop[0], gear)
    assert not is_feasible(pop[1], gear)
