"""
Paper-scale multi-seed Algorithm 1 orchestration.

Only invoked by ``scripts/run_raise_paper_scale.py`` (never by pytest).
Aggregates final-policy metrics as mean±std **across seeds**, with explicit
methodology text when the paper omits its seed count.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from crowd_nav.reward_search.checkpointing import CheckpointStore, CostLogger, Timer
from crowd_nav.reward_search.pipeline import RaisePipeline, RaiseRunConfig
from crowd_nav.reward_search.presets import (
    PaperScaleSpec,
    apply_paper_scale,
    load_paper_scale_yaml,
    parse_seeds_arg,
)
from crowd_nav.reward_search.reporting import write_json

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("SR", "CR", "TR", "NT", "PL", "ITR", "SD")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_final_metrics(seed_dir: str) -> Dict[str, float]:
    """Load final-policy scalars for one completed seed directory."""
    final_path = os.path.join(seed_dir, "final_candidate.json")
    if os.path.isfile(final_path):
        with open(final_path, encoding="utf-8") as f:
            cand = json.load(f)
        md = (cand.get("metadata") or {}).get("last_metrics") or {}
        if md:
            return {k: float(md[k]) for k in _METRIC_KEYS if k in md}

    pop_path = os.path.join(seed_dir, "stage3_population.json")
    if os.path.isfile(pop_path):
        with open(pop_path, encoding="utf-8") as f:
            pop = json.load(f)
        hist = pop.get("history") or []
        if hist:

            def _score(h: Dict[str, Any]) -> float:
                m = h.get("metrics") or {}
                return float(
                    m.get("SR", 0) - m.get("CR", 0) - 0.5 * m.get("TR", 0)
                )

            # Best-ever across all rounds (not last-round only).
            best = max(hist, key=_score)
            m = best.get("metrics") or {}
            return {k: float(m[k]) for k in _METRIC_KEYS if k in m}

    raise FileNotFoundError(f"No final metrics under {seed_dir}")


def aggregate_across_seeds(per_seed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean ± sample std across seeds (ddof=1 when n>1)."""
    if not per_seed:
        return {"n_seeds": 0, "metrics": {}, "per_seed": []}

    keys = sorted(
        {
            k
            for row in per_seed
            for k in (row.get("metrics") or {})
            if isinstance((row.get("metrics") or {}).get(k), (int, float))
        }
    )
    out_metrics: Dict[str, Any] = {}
    for key in keys:
        vals = [
            float(row["metrics"][key])
            for row in per_seed
            if key in (row.get("metrics") or {})
        ]
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        if n <= 1:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
        out_metrics[key] = {
            "mean": mean,
            "std": std,
            "n": n,
            "format": f"{mean:.4f}+/-{std:.4f}",
        }
    return {
        "n_seeds": len(per_seed),
        "metrics": out_metrics,
        "per_seed": list(per_seed),
        "aggregation": "across_seeds",
        "aggregation_note": (
            "mean±std is taken across independent Algorithm-1 seeds, "
            "not across evaluation episodes within a single seed."
        ),
    }


def build_paper_scale_report(
    *,
    spec: PaperScaleSpec,
    seeds: Sequence[int],
    per_seed_rows: Sequence[Dict[str, Any]],
    cost_log_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    agg = aggregate_across_seeds(per_seed_rows)
    cost_totals: Dict[str, Any] = {}
    if os.path.isfile(cost_log_path):
        with open(cost_log_path, encoding="utf-8") as f:
            cost_payload = json.load(f)
        cost_totals = cost_payload.get("totals") or {}
    return {
        "created_utc": _utc_now(),
        "scale": "paper",
        "hyperparameters": {
            "N": spec.N,
            "G1": spec.G1,
            "M": spec.M,
            "N_traj": spec.N_traj,
            "K2": spec.K2,
            "G2": spec.G2,
            "E2": spec.E2,
            "T_short": spec.T_short,
            "K3": spec.K3,
            "G3": spec.G3,
            "E3": spec.E3,
            "human_counts": list(spec.human_counts),
        },
        "methodology": {
            "seed_count_default": len(seeds),
            "seeds": list(seeds),
            "seed_count_rationale": spec.methodology_note(),
            "variance_source": "across_seeds",
            "tables": (
                "Tables 3–6 hyper-parameters; "
                "Table 1-style mean±std across seeds"
            ),
        },
        "table1_style_aggregate": agg,
        "cost": cost_totals,
        "cost_log": cost_log_path,
        "output_dir": output_dir,
    }


class PaperScaleRunner:
    """Run Algorithm 1 once per seed at paper budgets, with resume + cost logging."""

    def __init__(
        self,
        spec: Optional[PaperScaleSpec] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        llm_provider: Optional[str] = None,
        dry_run_stubs: bool = False,
    ) -> None:
        self.spec = spec or load_paper_scale_yaml()
        self.seeds = parse_seeds_arg(seeds, default=self.spec.seeds)
        self.output_dir = output_dir or self.spec.output_dir
        self.device = device or self.spec.device
        self.llm_provider = llm_provider or self.spec.llm_provider
        self.dry_run_stubs = bool(dry_run_stubs)
        os.makedirs(self.output_dir, exist_ok=True)
        self.cost_log_path = self.spec.cost_log or os.path.join(
            self.output_dir, "cost_log.json"
        )
        self.cost_logger = CostLogger(self.cost_log_path)
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.store = CheckpointStore(
            self.checkpoint_root,
            cost_logger=self.cost_logger,
            device=self.device,
        )

    def _config_for_seed(self, seed: int) -> RaiseRunConfig:
        seed_dir = os.path.join(self.output_dir, f"seed_{int(seed):04d}")
        cfg = RaiseRunConfig(output_dir=seed_dir, seed=int(seed))
        apply_paper_scale(cfg, self.spec)
        cfg.seed = int(seed)
        cfg.device = self.device
        cfg.llm_provider = self.llm_provider
        cfg.output_dir = seed_dir
        if self.dry_run_stubs:
            cfg.apply_fast_profile()
            logger.warning(
                "dry_run_stubs=True: using --fast stub budgets (not paper K3=1e7)"
            )
        return cfg

    def run_seed(self, seed: int) -> Dict[str, Any]:
        seed_dir = os.path.join(self.output_dir, f"seed_{int(seed):04d}")
        if self.store.seed_complete(seed) and os.path.isfile(
            os.path.join(seed_dir, "final_candidate.json")
        ):
            logger.info("Seed %s already complete — loading metrics", seed)
            metrics = extract_final_metrics(seed_dir)
            return {"seed": seed, "metrics": metrics, "resumed_complete": True}

        cfg = self._config_for_seed(seed)
        timer = Timer()
        pipeline = RaisePipeline(cfg, checkpoint_store=self.store)
        logger.info(
            "Paper-scale seed=%s → %s (K2=%d G2=%d K3=%d G3=%d)",
            seed,
            cfg.output_dir,
            cfg.stage2_train_steps,
            cfg.stage2_rounds,
            cfg.stage3_train_steps,
            cfg.stage3_rounds,
        )
        artifacts = pipeline.run()
        wall = timer.elapsed()
        best_id = (
            artifacts.best_stage3.candidate_id if artifacts.best_stage3 else None
        )
        self.store.mark_stage_done(seed, "stage1", {"status": "done"})
        self.store.mark_stage_done(seed, "stage2", {"status": "done"})
        self.store.mark_stage_done(
            seed,
            "stage3",
            {"wall_seconds_seed": wall, "best_stage3_id": best_id},
        )
        metrics = extract_final_metrics(seed_dir)
        return {
            "seed": seed,
            "metrics": metrics,
            "wall_seconds": wall,
            "resumed_complete": False,
        }

    def run(self) -> Dict[str, Any]:
        per_seed: List[Dict[str, Any]] = []
        for seed in self.seeds:
            row = self.run_seed(int(seed))
            per_seed.append(row)
            write_json(
                os.path.join(
                    self.output_dir, f"seed_{int(seed):04d}", "seed_metrics.json"
                ),
                row,
            )

        report = build_paper_scale_report(
            spec=self.spec,
            seeds=self.seeds,
            per_seed_rows=per_seed,
            cost_log_path=self.cost_log_path,
            output_dir=self.output_dir,
        )
        report_path = os.path.join(self.output_dir, "paper_scale_report.json")
        write_json(report_path, report)
        meth_path = os.path.join(self.output_dir, "METHODOLOGY.md")
        with open(meth_path, "w", encoding="utf-8") as f:
            f.write("# Paper-scale methodology\n\n")
            f.write(self.spec.methodology_note() + "\n\n")
            f.write("## Hyper-parameters (Tables 3–6)\n\n")
            f.write(
                f"- N={self.spec.N}, G1={self.spec.G1}, M={self.spec.M}, "
                f"N_traj={self.spec.N_traj}\n"
            )
            f.write(
                f"- K2={self.spec.K2}, G2={self.spec.G2}, E2={self.spec.E2}, "
                f"T_short={self.spec.T_short}\n"
            )
            f.write(
                f"- K3={self.spec.K3}, G3={self.spec.G3}, E3={self.spec.E3}\n"
            )
            f.write(f"- Seeds: {list(self.seeds)} (mean±std across seeds)\n")
            f.write(
                f"\nCost log: `{self.cost_log_path}` "
                "(wall / GPU-hours per stage for Table 9 comparisons).\n"
            )
        logger.info("Wrote paper-scale report → %s", report_path)
        return report
