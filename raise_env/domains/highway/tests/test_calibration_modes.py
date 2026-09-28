"""Calibration modes, truthful feedback text, progress objective, reachable range."""

from __future__ import annotations

from domains.highway.action_config import (
    ActionModeSettings,
    set_action_settings,
    warn_inline_speed_literals,
)
from domains.highway.pareto_rank import (
    Metrics,
    ReferenceStats,
    is_feasible,
    parse_calibration_mode,
    rank_population,
    reference_for_mode,
    survival_only_reference,
    _objectives,
)
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.evolve_rank import (
    resolve_elite_archive_mode,
    rank_population_pareto,
)
from raise_core.raise_loop.proxy_feedback import evidence_block


def _m(
    cid: str,
    *,
    sr: float,
    cr: float = 0.0,
    tr: float = 0.0,
    progress: float = 100.0,
    speed: float = 20.0,
    soft: float = 0.0,
    p10: float | None = None,
) -> Metrics:
    return Metrics(
        candidate_id=cid,
        sr=sr,
        cr=cr,
        tr=tr,
        progress=progress,
        mean_speed=speed,
        soft_success=soft,
        speed_p10=p10 if p10 is not None else speed,
    )


def test_parse_calibration_mode_default():
    assert parse_calibration_mode("") == "no_speed_floor"
    assert parse_calibration_mode(None) == "no_speed_floor"
    assert parse_calibration_mode("population") == "population"
    assert parse_calibration_mode("env_measured") == "env_measured"


def test_no_speed_floor_survival_only():
    ref = reference_for_mode("no_speed_floor", [])
    assert ref.require_survival is True
    assert ref.v_floor is None
    assert ref.cr_ceiling is None
    assert is_feasible(_m("ok", sr=0.1, cr=0.9, speed=1.0), ref)
    assert not is_feasible(_m("dead", sr=0.0, cr=0.0, speed=30.0), ref)


def test_env_measured_attaches_ambient_not_as_floor():
    ambient = {
        "traffic_speed_mean": 20.5,
        "traffic_speed_p10": 18.0,
        "traffic_speed_p90": 23.0,
    }
    ref = reference_for_mode("env_measured", [], ambient=ambient)
    assert ref.require_survival is True
    assert ref.v_floor is None
    assert ref.ambient == ambient


def test_population_mode_still_sets_floor():
    pop = [
        _m("a", sr=1.0, speed=20.0, p10=20.0),
        _m("b", sr=1.0, speed=22.0, p10=22.0),
        _m("c", sr=1.0, speed=24.0, p10=24.0),
    ]
    ref = reference_for_mode("population", pop)
    assert ref.v_floor is not None
    assert ref.source == "population_percentile"
    assert not ref.require_survival


def test_progress_objective_toggle():
    from domains.highway.pareto_rank import ParetoObjectives

    m = _m("x", sr=1.0, progress=500.0, speed=20.0)
    with_p = ParetoObjectives(progress=True)
    without = ParetoObjectives(progress=False)
    assert len(_objectives(m, with_p)) == 5
    assert len(_objectives(m, without)) == 4
    assert list(_objectives(m, with_p))[3] == 500.0


def test_crawler_and_faster_both_feasible_without_floor():
    crawler = _m("crawl", sr=1.0, cr=0.0, progress=50.0, speed=5.0, p10=5.0)
    faster = _m("fast", sr=0.8, cr=0.2, progress=600.0, speed=24.0, p10=24.0)
    ref = survival_only_reference()
    ordered = rank_population([crawler, faster], ref=ref)
    assert is_feasible(crawler, ref) and is_feasible(faster, ref)
    # Faster dominates on progress+speed while crawler is safer on CR —
    # both stay on front 0 (non-dominated). Order is crowding within front.
    ids = {m.candidate_id for m in ordered}
    assert ids == {"crawl", "fast"}


def test_feedback_no_false_traffic_claim_population():
    text = evidence_block(
        {"SR": 1.0, "CR": 0.0, "TR": 0.0, "mean_speed": 20.0, "PL": 100.0},
        score1=0.5,
        metadata={
            "pareto_rank": 0,
            "pareto_n": 3,
            "pareto_feasible": True,
            "pareto_front": 0,
            "pareto_v_floor": 20.1,
            "pareto_cr_ceiling": 0.05,
            "pareto_tr_ceiling": 0.0,
            "pareto_calibration_source": "population_percentile",
        },
    )
    assert "measured from the environment's own traffic" not in text
    assert "population statistic" in text
    assert "20.1" in text


def test_feedback_survival_only_text():
    text = evidence_block(
        {"SR": 1.0, "CR": 0.0, "TR": 0.0, "mean_speed": 20.0, "PL": 100.0},
        metadata={
            "pareto_rank": 0,
            "pareto_n": 2,
            "pareto_feasible": True,
            "pareto_front": 0,
            "pareto_require_survival": True,
            "pareto_calibration_source": "no_speed_floor",
        },
    )
    assert "survival only" in text
    assert "measured from the environment's own traffic" not in text
    assert "Reachable action speed range" in text


def test_feedback_ambient_info():
    text = evidence_block(
        {"SR": 0.5, "CR": 0.1, "TR": 0.0, "mean_speed": 21.0, "PL": 200.0},
        metadata={
            "pareto_rank": 0,
            "pareto_n": 1,
            "pareto_feasible": True,
            "pareto_require_survival": True,
            "pareto_calibration_source": "env_measured",
            "pareto_ambient_traffic": {
                "traffic_speed_mean": 20.6,
                "traffic_speed_p10": 18.7,
                "traffic_speed_p90": 22.3,
            },
        },
    )
    assert "non-ego" in text
    assert "informational only" in text
    assert "20.6" in text


def test_rank_population_pareto_no_speed_floor_mode():
    def _cand(cid: str, **kw: float) -> RewardCandidate:
        return RewardCandidate(
            candidate_id=cid,
            code="def compute_reward(state, memory):\n    return 0.0\n",
            valid=True,
            origin="test",
            score=0.5,
            metadata={
                "last_metrics": {
                    "SR": kw["sr"],
                    "CR": kw.get("cr", 0.0),
                    "TR": kw.get("tr", 0.0),
                    "mean_speed": kw.get("speed", 20.0),
                    "mean_progress": kw.get("progress", 100.0),
                    "soft_success": 0.0,
                    "speed_p10": kw.get("speed", 20.0),
                }
            },
        )

    crawler = _cand("crawl", sr=1.0, cr=0.0, speed=5.0, progress=40.0)
    risky = _cand("risky", sr=0.7, cr=0.25, speed=24.0, progress=700.0)
    ranked = rank_population_pareto(
        [crawler, risky], calibration_mode="no_speed_floor"
    )
    assert ranked[0].metadata.get("pareto_require_survival") is True
    assert "pareto_v_floor" not in (ranked[0].metadata or {})
    assert ranked[0].metadata.get("pareto_feasible") is True
    assert ranked[1].metadata.get("pareto_feasible") is True


def test_elite_archive_auto_resolution():
    assert resolve_elite_archive_mode("auto", "population") == "fitness"
    assert resolve_elite_archive_mode("auto", "no_speed_floor") == "pareto"
    assert resolve_elite_archive_mode("pareto", "population") == "pareto"


def test_inline_speed_literal_warning_not_rejection():
    set_action_settings(ActionModeSettings(mode="meta_default"))
    code = (
        "def compute_reward(state, memory):\n"
        "    return -abs(state.speed - 16.0)\n"
    )
    warnings = warn_inline_speed_literals(code)
    assert warnings
    assert any("16.0" in w or "16" in w for w in warnings)
