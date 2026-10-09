"""
Highway closed-loop evolution hooks (Pareto calibration + elite archive).

:class:`ClosedLoopRunner` calls these through ``DomainPack.make_evolve_hooks``
so highway-only ranking state lives in the pack instead of ``if highway``
branches in the shared runner. One instance spans a whole run and is
round-tripped through the closed-loop checkpoint (``extra.highway_elite``).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from domains.crowdnav.reporting import candidate_to_dict, load_candidate_dict
from domains.highway.pareto_rank import (
    ParetoObjectives,
    ReferenceStats,
    calibrate_from_gear_endpoints,
    collect_ambient_traffic_stats,
    collect_gear_endpoint_reference_stats,
    parse_calibration_mode,
)
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.diversity import diversify_ranking
from raise_core.raise_loop.evolve_rank import (
    resolve_elite_archive_mode,
    select_pareto_elite,
)
from raise_core.raise_loop.logging_io import closed_loop_dir, write_json
from raise_core.raise_loop.proxy_feedback import refresh_highway_evidence_after_pareto
from raise_core.selection import candidate_fitness

logger = logging.getLogger(__name__)

CHECKPOINT_KEY = "highway_elite"


class HighwayEvolveHooks:
    def __init__(self) -> None:
        self.pareto_elite: Optional[RewardCandidate] = None
        self.best_ever_cand: Optional[RewardCandidate] = None
        self.best_ever_fitness: Optional[float] = None
        self._ambient_loaded = False
        self._ambient: Optional[Dict[str, float]] = None
        self._ref_loaded = False
        self._pareto_ref: Optional[ReferenceStats] = None

    # ------------------------------------------------------------ ranking

    def ranking_inputs(self, cfg: Any, evolve_mode: str) -> Dict[str, Any]:
        """Keyword arguments for ``rank_population_for_evolution``."""
        calib_mode = parse_calibration_mode(getattr(cfg, "highway_calibration_mode", None))
        objectives = ParetoObjectives(
            progress=bool(getattr(cfg, "highway_pareto_use_progress", True)),
            lane_change=bool(getattr(cfg, "highway_pareto_use_lane_change", True)),
            overtake=bool(getattr(cfg, "highway_pareto_use_overtake", True)),
        )
        ambient = None
        pareto_ref = None
        if evolve_mode == "pareto" and calib_mode == "env_measured":
            ambient = self._ambient_traffic(cfg)
        if evolve_mode == "pareto" and calib_mode == "population":
            pareto_ref = self._pareto_reference(cfg)
        return {
            "pareto_ref": pareto_ref,
            "calibration_mode": calib_mode,
            "pareto_objectives": objectives,
            "ambient": ambient,
        }

    def after_ranking(
        self,
        ranked_for_evo: List[RewardCandidate],
        cfg: Any,
        inputs: Dict[str, Any],
    ) -> Tuple[List[RewardCandidate], Optional[str]]:
        """Demote metric clones, update elite archive, pin the elite first."""
        ranked_for_evo = diversify_ranking(ranked_for_evo)
        calib_mode = inputs["calibration_mode"]
        elite_mode = resolve_elite_archive_mode(
            getattr(cfg, "highway_elite_archive", "auto"), calib_mode
        )
        # Legacy fitness is still tracked for epoch logs.
        for c in ranked_for_evo:
            fit = float(candidate_fitness(c))
            best = self.best_ever_fitness
            if fit > (float("-inf") if best is None else float(best)):
                self.best_ever_fitness = fit
                self.best_ever_cand = c
        if elite_mode == "pareto":
            self.pareto_elite = select_pareto_elite(
                ranked_for_evo,
                self.pareto_elite,
                calibration_mode=str(calib_mode),
                objectives=inputs["pareto_objectives"],
                ambient=inputs["ambient"],
            )
            elite = self.pareto_elite
        else:
            elite = self.best_ever_cand
        elite_id: Optional[str] = None
        if elite is not None and bool(cfg.keep_runtime_elite):
            eid = str(elite.candidate_id)
            rest = [c for c in ranked_for_evo if str(c.candidate_id) != eid]
            # Prefer live copy from this epoch if still present.
            live = next((c for c in ranked_for_evo if str(c.candidate_id) == eid), elite)
            ranked_for_evo = [live] + rest
            elite_id = eid
        # Refresh LLM evidence with stamped pareto_* (label-time attach runs
        # before stamp_pareto_ranks); the runner builds reflection afterwards.
        refresh_highway_evidence_after_pareto(ranked_for_evo)
        return ranked_for_evo, elite_id

    def pareto_summary(
        self,
        ranked_for_evo: List[RewardCandidate],
        evolve_mode: str,
        inputs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if evolve_mode != "pareto":
            return None
        front0_ids = []
        n_feas = 0
        for c in ranked_for_evo:
            md = c.metadata or {}
            if md.get("pareto_feasible"):
                n_feas += 1
            if md.get("pareto_front0") or md.get("pareto_front") == 0:
                front0_ids.append(str(c.candidate_id))
        objectives = inputs.get("pareto_objectives")
        return {
            "calibration_mode": inputs.get("calibration_mode"),
            "objectives": list(objectives.names()) if objectives is not None else None,
            "elite_id": (
                str(self.pareto_elite.candidate_id) if self.pareto_elite is not None else None
            ),
            "n_feasible": n_feas,
            "n_population": len(ranked_for_evo),
            "front0_ids": front0_ids,
            "front0_n": len(front0_ids),
            "evolve_order": [str(c.candidate_id) for c in ranked_for_evo],
        }

    # --------------------------------------------------------- checkpoint

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if self.pareto_elite is None and self.best_ever_cand is None:
            return None
        fit = self.best_ever_fitness
        return {
            "pareto_elite": (
                candidate_to_dict(self.pareto_elite) if self.pareto_elite is not None else None
            ),
            "best_ever_cand": (
                candidate_to_dict(self.best_ever_cand)
                if self.best_ever_cand is not None
                else None
            ),
            "best_ever_fitness": (
                float(fit) if fit is not None and math.isfinite(float(fit)) else None
            ),
        }

    def load_state_dict(self, saved: Optional[Dict[str, Any]], *, validator: Any) -> None:
        if not isinstance(saved, dict):
            return
        if isinstance(saved.get("pareto_elite"), dict):
            self.pareto_elite = load_candidate_dict(
                dict(saved["pareto_elite"]), validator=validator
            )
        if isinstance(saved.get("best_ever_cand"), dict):
            self.best_ever_cand = load_candidate_dict(
                dict(saved["best_ever_cand"]), validator=validator
            )
        if saved.get("best_ever_fitness") is not None:
            self.best_ever_fitness = float(saved["best_ever_fitness"])

    # ------------------------------------------------------- calibration

    def _ambient_traffic(self, cfg: Any) -> Optional[Dict[str, float]]:
        """Cached non-ego traffic speed stats (``env_measured``; informational)."""
        if self._ambient_loaded:
            return self._ambient
        self._ambient_loaded = True
        path = os.path.join(closed_loop_dir(cfg.output_dir), "ambient_traffic.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._ambient = dict(json.load(fh) or {}) or None
                return self._ambient
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load ambient_traffic.json (%s)", exc)
        if bool(cfg.use_stub):
            return None
        try:
            stats = collect_ambient_traffic_stats(seed=int(cfg.seed))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ambient traffic measurement failed (%s)", exc)
            return None
        if stats:
            write_json(path, stats)
            self._ambient = stats
        return self._ambient

    def _pareto_reference(self, cfg: Any) -> Optional[ReferenceStats]:
        """One-shot SLOWER/FASTER env reference (cached on disk and in memory).

        IDLE is useless here (CR≈1); gear endpoints come from the action space.
        """
        if self._ref_loaded:
            return self._pareto_ref
        self._ref_loaded = True
        ref_path = os.path.join(closed_loop_dir(cfg.output_dir), "pareto_reference.json")
        if os.path.isfile(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
                self._pareto_ref = ReferenceStats(
                    v_floor=float(payload["v_floor"]),
                    cr_ceiling=float(payload["cr_ceiling"]),
                    tr_ceiling=float(payload["tr_ceiling"]),
                    source=str(payload.get("source") or "gear_endpoints_slower_faster"),
                    se_margin=bool(payload.get("se_margin", True)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load pareto_reference.json (%s)", exc)
                self._pareto_ref = None
        if self._pareto_ref is None and not bool(cfg.use_stub):
            try:
                (
                    slow_s,
                    slow_cr,
                    slow_tr,
                    fast_s,
                    fast_cr,
                    fast_tr,
                ) = collect_gear_endpoint_reference_stats(n_episodes=4, seed=int(cfg.seed))
                ref = calibrate_from_gear_endpoints(
                    slow_s, slow_cr, slow_tr, fast_s, fast_cr, fast_tr
                )
                self._pareto_ref = ref
                write_json(
                    ref_path,
                    {
                        "v_floor": ref.v_floor,
                        "cr_ceiling": ref.cr_ceiling,
                        "tr_ceiling": ref.tr_ceiling,
                        "source": ref.source,
                        "se_margin": ref.se_margin,
                        "slower_cr": float(slow_cr),
                        "faster_cr": float(fast_cr),
                        "n_slower_samples": int(len(slow_s)),
                        "n_faster_samples": int(len(fast_s)),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Pareto SLOWER/FASTER reference failed (%s); "
                    "falling back to per-generation percentiles",
                    exc,
                )
                self._pareto_ref = None
        return self._pareto_ref
