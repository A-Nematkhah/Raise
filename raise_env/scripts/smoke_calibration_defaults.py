"""Smoke: new calibration defaults + survival-only ranking without LLM/PPO."""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import raise_paths  # noqa: F401

from domains.highway.prompts import _SELECTION_OBJECTIVE
from domains.highway.pareto_rank import parse_calibration_mode, DEFAULT_CALIBRATION_MODE
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.evolve_rank import (
    rank_population_pareto,
    resolve_elite_archive_mode,
    select_pareto_elite,
)
from raise_core.raise_loop.proxy_feedback import evidence_block
from raise_core.presets import CLOSED_LOOP_PROFILES


def _cand(cid: str, *, sr: float, cr: float, speed: float, progress: float) -> RewardCandidate:
    return RewardCandidate(
        candidate_id=cid,
        code="def compute_reward(state, memory):\n    return 0.0\n",
        valid=True,
        origin="smoke",
        score=0.5,
        metadata={
            "last_metrics": {
                "SR": sr,
                "CR": cr,
                "TR": 0.0,
                "mean_speed": speed,
                "mean_progress": progress,
                "soft_success": 0.0,
                "speed_p10": speed,
                "n_eval_episodes": 4,
            }
        },
    )


def main() -> int:
    assert DEFAULT_CALIBRATION_MODE == "no_speed_floor"
    assert parse_calibration_mode("") == "no_speed_floor"
    assert ClosedLoopConfig().highway_calibration_mode == "no_speed_floor"
    assert CLOSED_LOOP_PROFILES["highway_4h"]["highway_calibration_mode"] == "no_speed_floor"
    assert CLOSED_LOOP_PROFILES["highway_7h"]["highway_calibration_mode"] == "no_speed_floor"
    assert CLOSED_LOOP_PROFILES["highway_7h"]["population"] == 8
    assert CLOSED_LOOP_PROFILES["highway_7h"]["generations"] == 7
    assert CLOSED_LOOP_PROFILES["highway_7h"]["k2"] == CLOSED_LOOP_PROFILES["highway_4h"]["k2"]
    assert (
        CLOSED_LOOP_PROFILES["highway_7h"]["stage3_k3"]
        == CLOSED_LOOP_PROFILES["highway_4h"]["stage3_k3"]
    )
    assert "built-in traffic" not in _SELECTION_OBJECTIVE

    crawler = _cand("crawl", sr=1.0, cr=0.0, speed=5.0, progress=40.0)
    faster = _cand("fast", sr=0.75, cr=0.2, speed=24.0, progress=700.0)
    ranked = rank_population_pareto(
        [crawler, faster], calibration_mode="no_speed_floor"
    )
    assert all((c.metadata or {}).get("pareto_feasible") for c in ranked)
    assert "pareto_v_floor" not in (ranked[0].metadata or {})

    text = evidence_block(
        ranked[0].metadata["last_metrics"],
        metadata=ranked[0].metadata,
    )
    assert "survival only" in text
    assert "measured from the environment's own traffic" not in text
    assert "Reachable action speed range" in text

    elite = select_pareto_elite(
        ranked, None, calibration_mode="no_speed_floor"
    )
    assert elite is not None
    assert resolve_elite_archive_mode("auto", "no_speed_floor") == "pareto"
    assert resolve_elite_archive_mode("auto", "population") == "fitness"

    # Population mode still stamps a floor (legacy restore path).
    legacy = rank_population_pareto(
        [crawler, faster], calibration_mode="population"
    )
    assert (legacy[0].metadata or {}).get("pareto_v_floor") is not None
    legacy_text = evidence_block(
        legacy[0].metadata["last_metrics"], metadata=legacy[0].metadata
    )
    assert "population statistic" in legacy_text
    assert "measured from the environment's own traffic" not in legacy_text

    print("SMOKE_OK calibration_mode=no_speed_floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
