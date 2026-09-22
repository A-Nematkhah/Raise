"""Closed-loop multi-fidelity runner (innovation path). See PLAN.md."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from crowd_nav.reward_search.raise_loop.config import ClosedLoopConfig
from crowd_nav.reward_search.raise_loop.epoch import select_to_label
from crowd_nav.reward_search.raise_loop.logging_io import (
    append_al_step,
    append_epoch_record,
    closed_loop_dir,
    write_json,
)
from crowd_nav.reward_search.explore import RewardCandidate, StageIConfig, StageIEvolver
from crowd_nav.reward_search.llm import LLMClient, make_llm_client
from crowd_nav.reward_search.surrogate.bootstrap import label_and_append_candidate
from crowd_nav.reward_search.surrogate.dataset_io import (
    existing_example_ids,
    load_table,
    read_manifest,
    write_manifest,
)
from crowd_nav.reward_search.surrogate.features import FEATURE_SCHEMA_VERSION
from crowd_nav.reward_search.surrogate.gate import (
    predict_population,
    surrogate_model_ready,
)
from crowd_nav.reward_search.surrogate.model import SurrogateModel

logger = logging.getLogger(__name__)


@dataclass
class ClosedLoopResult:
    population: List[RewardCandidate]
    history: List[Dict[str, Any]] = field(default_factory=list)
    n_labeled_total: int = 0
    model_dir: str = ""
    dataset_dir: str = ""
    output_dir: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)


def _refit_surrogate(
    dataset_dir: str,
    model_dir: str,
    *,
    seed: int,
    use_stub: bool,
) -> Dict[str, Any]:
    feats, labs = load_table(dataset_dir)
    if len(feats) < 1:
        raise RuntimeError("closed-loop refit: surrogate dataset is empty")
    model = SurrogateModel(random_seed=int(seed), n_bags=3 if use_stub else 5)
    metrics = model.fit(feats, labs, target_keys=("SR", "CR", "TR"))
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    metrics_path = os.path.join(model_dir, "metrics.json")
    write_json(metrics_path, metrics)
    return metrics


class ClosedLoopRunner:
    """
    Generation epochs: Score1 → (gate+AL) → Stage II short labels → refit Surrogate.
    """

    def __init__(
        self,
        cfg: ClosedLoopConfig,
        *,
        llm: Optional[LLMClient] = None,
        score_fn: Any = None,
        trainer: Any = None,
        validator: Any = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.score_fn = score_fn
        self.trainer = trainer
        self.validator = validator

    def run(self) -> ClosedLoopResult:
        from crowd_nav.domains import load_domain, make_stage2_trainer_for_domain
        from crowd_nav.reward_search.sandbox.validator import RewardValidator
        from crowd_nav.reward_search.refine import Stage2Config

        cfg = self.cfg
        if cfg.use_stub:
            cfg.apply_fast_profile()

        os.makedirs(cfg.output_dir, exist_ok=True)
        closed_loop_dir(cfg.output_dir)
        os.makedirs(cfg.surrogate_dataset, exist_ok=True)
        os.makedirs(cfg.al_root, exist_ok=True)

        pack = load_domain("crowdnav")
        score1_mode = "smoke" if cfg.use_stub else "dataset"
        if self.score_fn is None:
            self.score_fn, _ds = pack.make_score_fn(
                mode=score1_mode,
                dataset_path=None if score1_mode == "smoke" else cfg.stage1_dataset_path,
            )
        if self.trainer is None:
            self.trainer = make_stage2_trainer_for_domain(pack, use_stub=cfg.use_stub)
        if self.validator is None:
            self.validator = RewardValidator()
        if self.llm is None:
            self.llm = make_llm_client(cfg.llm_provider)

        n = int(cfg.population_size)
        n_crossover = int(cfg.n_crossover)
        n_mutation = int(cfg.n_mutation)
        n_random = int(cfg.n_random)
        if n_crossover + n_mutation + n_random != n:
            n_crossover = min(2, n)
            n_mutation = min(4, max(0, n - n_crossover))
            n_random = n - n_crossover - n_mutation

        s1_cfg = StageIConfig(
            population_size=n,
            generations=int(cfg.generations),
            n_crossover=n_crossover,
            n_mutation=n_mutation,
            n_random=n_random,
            keep_runtime_elite=bool(cfg.keep_runtime_elite),
        )
        evolver = StageIEvolver(
            self.llm,
            score_fn=self.score_fn,
            validator=self.validator,
            config=s1_cfg,
            rejection_log_path=os.path.join(
                cfg.output_dir, "closed_loop", "stage1_rejections.jsonl"
            ),
        )

        from crowd_nav.reward_search.regime import env_name_for_predict_method

        stage2_cfg = Stage2Config(
            train_env_steps=int(cfg.stage2_train_steps),
            k2_unit=str(cfg.k2_unit),
            eval_episodes=8 if cfg.use_stub else 50,
            horizon_steps=20 if cfg.use_stub else max(1, int(cfg.horizon_steps)),
            seed=int(cfg.seed),
            device=str(cfg.device),
            output_root=os.path.join(cfg.output_dir, "closed_loop", "stage2_train"),
            num_processes=max(1, int(cfg.num_processes)),
            human_num=max(1, int(cfg.human_num)),
            predict_method=str(cfg.predict_method),
            randomization_regime=str(cfg.randomization_regime),
            env_name=env_name_for_predict_method(str(cfg.predict_method)),
        )

        known_ids = existing_example_ids(cfg.surrogate_dataset)
        n_labeled = len(known_ids)
        labels_since_refit = 0
        history: List[Dict[str, Any]] = []

        # Gen0 propose
        population = evolver.initialize_population()

        for g in range(int(cfg.generations)):
            ranked = evolver.score_population(population)
            evolver.reflection = evolver._build_reflection(ranked, generation=g)
            if evolver.global_best is None or (
                ranked[0].score is not None
                and (
                    evolver.global_best.score is None
                    or float(ranked[0].score) > float(evolver.global_best.score)
                )
            ):
                evolver.global_best = ranked[0]

            model_ready = surrogate_model_ready(cfg.surrogate_model_dir)
            predictions = None
            if model_ready and g > 0:
                try:
                    predictions, _model = predict_population(
                        ranked,
                        cfg.surrogate_model_dir,
                        score_fn=self.score_fn,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("closed-loop predict failed: %s", exc)
                    predictions = None
                    model_ready = False

            score1_results = [
                (c.metadata or {}).get("score1_result") for c in ranked
            ]
            # Prefer attaching raw Score1Result if stored; else None list is fine.
            to_label, gate_report, al_report = select_to_label(
                ranked,
                epoch=g,
                n_labeled=n_labeled,
                min_labels_for_gate=int(cfg.min_labels_for_gate),
                predictions=predictions,
                model_ready=bool(model_ready and predictions is not None),
                drop_fraction=float(cfg.drop_fraction),
                max_uncertainty_to_drop=float(cfg.max_uncertainty_to_drop),
                min_keep=int(cfg.min_keep),
                min_stage2_per_gen=int(cfg.min_stage2_per_gen),
                al_enabled=bool(cfg.al_enabled),
                al_max_per_epoch=int(cfg.al_max_per_epoch),
                al_allow_stage1_requests=bool(cfg.al_allow_stage1_requests),
                al_root=str(cfg.al_root),
                score1_results=score1_results,
            )

            if al_report.get("enabled"):
                append_al_step(
                    cfg.output_dir,
                    {"epoch": g, **al_report},
                )

            n_ok = 0
            n_fail = 0
            for i, cand in enumerate(to_label):
                result = label_and_append_candidate(
                    cand,
                    score_fn=self.score_fn,
                    trainer=self.trainer,
                    stage2_cfg=stage2_cfg,
                    out_dir=cfg.surrogate_dataset,
                    round_index=g * 1000 + i,
                    known_ids=known_ids,
                    label_rejected=bool(cfg.label_rejected),
                    use_stub=bool(cfg.use_stub),
                )
                status = str(result.get("status") or "")
                if status == "ok" and result.get("example_id"):
                    n_ok += 1
                    labels_since_refit += 1
                    n_labeled = len(known_ids)
                elif status == "failed" and result.get("example_id"):
                    # Still appended (ok=False); do not advance refit budget.
                    n_fail += 1
                    n_labeled = len(known_ids)
                elif status == "skipped":
                    pass

            n_labeled = len(existing_example_ids(cfg.surrogate_dataset))
            known_ids = existing_example_ids(cfg.surrogate_dataset)

            should_refit = False
            if g == 0:
                should_refit = n_labeled >= 1
            elif labels_since_refit >= int(cfg.refit_every_new_labels):
                should_refit = True
            elif n_labeled >= int(cfg.min_labels_for_gate) and not surrogate_model_ready(
                cfg.surrogate_model_dir
            ):
                should_refit = True

            fit_metrics: Optional[Dict[str, Any]] = None
            if should_refit and n_labeled >= 1:
                try:
                    fit_metrics = _refit_surrogate(
                        cfg.surrogate_dataset,
                        cfg.surrogate_model_dir,
                        seed=int(cfg.seed),
                        use_stub=bool(cfg.use_stub),
                    )
                    labels_since_refit = 0
                    write_json(
                        os.path.join(closed_loop_dir(cfg.output_dir), "last_refit.json"),
                        {"epoch": g, "n_labeled": n_labeled, "metrics": fit_metrics},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("closed-loop refit failed: %s", exc)

            epoch_rec = {
                "epoch": g,
                "n_population": len(ranked),
                "n_to_label": len(to_label),
                "labeled_ids": [c.candidate_id for c in to_label],
                "ranking": [c.candidate_id for c in ranked],
                "best_id": ranked[0].candidate_id if ranked else None,
                "best_score1": ranked[0].score if ranked else None,
                "gate": gate_report,
                "al": al_report,
                "n_labeled_dataset": n_labeled,
                "refit": fit_metrics is not None,
                "n_label_ok": n_ok,
                "n_label_failed": n_fail,
            }
            from crowd_nav.reward_search import console
            from crowd_nav.reward_search.raise_loop.report import format_epoch_summary

            append_epoch_record(cfg.output_dir, epoch_rec)
            history.append(epoch_rec)
            write_json(
                os.path.join(closed_loop_dir(cfg.output_dir), f"pop_epoch_{g:03d}.json"),
                {
                    "ranking": [c.candidate_id for c in ranked],
                    "to_label": [c.candidate_id for c in to_label],
                },
            )

            console.status(format_epoch_summary(epoch_rec), stage="closed-loop")

            if g + 1 < int(cfg.generations):
                population = evolver._next_generation(ranked)
            else:
                population = ranked

        # Final manifest for dataset
        man = read_manifest(cfg.surrogate_dataset) or {}
        man.update(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "mode": "closed_loop_v1",
                "n_examples": n_labeled,
                "closed_loop_output": cfg.output_dir,
                "generations": int(cfg.generations),
            }
        )
        write_manifest(cfg.surrogate_dataset, man)

        result_manifest = {
            "mode": "closed_loop_v1",
            "n_epochs": len(history),
            "n_labeled_total": n_labeled,
            "surrogate_model_dir": cfg.surrogate_model_dir,
            "surrogate_dataset": cfg.surrogate_dataset,
            "config": asdict(cfg),
            "history": history,
        }
        write_json(
            os.path.join(closed_loop_dir(cfg.output_dir), "manifest.json"),
            result_manifest,
        )
        try:
            from crowd_nav.reward_search.raise_loop.report import write_closed_loop_report

            write_closed_loop_report(cfg.output_dir, manifest=result_manifest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("closed-loop report failed: %s", exc)

        return ClosedLoopResult(
            population=list(population),
            history=history,
            n_labeled_total=n_labeled,
            model_dir=cfg.surrogate_model_dir,
            dataset_dir=cfg.surrogate_dataset,
            output_dir=cfg.output_dir,
            manifest=result_manifest,
        )
