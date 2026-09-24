"""Closed-loop multi-fidelity runner (innovation path). See PLAN.md."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from crowd_nav.reward_search.raise_loop.checkpoint import (
    build_checkpoint,
    deserialize_population,
    load_checkpoint,
    save_checkpoint,
)
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
from crowd_nav.reward_search.reporting import load_candidate_dict
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
    resumed: bool = False


def _is_under(parent: str, child: str) -> bool:
    par = os.path.abspath(parent)
    chi = os.path.abspath(child)
    return chi == par or chi.startswith(par + os.sep)


def isolate_run_paths(cfg: ClosedLoopConfig) -> ClosedLoopConfig:
    """Nest surrogate model / dataset / AL queue under ``output_dir`` in place."""
    out = str(cfg.output_dir)
    if not _is_under(out, cfg.surrogate_model_dir):
        cfg.surrogate_model_dir = os.path.join(out, "surrogate_model")
    if not _is_under(out, cfg.surrogate_dataset):
        cfg.surrogate_dataset = os.path.join(out, "surrogate_dataset")
    if not _is_under(out, cfg.al_root):
        cfg.al_root = os.path.join(out, "active_learning")
    return cfg


def _checkpoint_dirs(output_dir: str) -> List[str]:
    """Checkpoint locations, most specific first (``closed_loop/`` then run root)."""
    return [closed_loop_dir(output_dir), output_dir]


def _save_ckpt(output_dir: str, payload: Dict[str, Any]) -> None:
    for path in _checkpoint_dirs(output_dir):
        save_checkpoint(path, payload)


def _load_ckpt(output_dir: str) -> Optional[Dict[str, Any]]:
    for path in _checkpoint_dirs(output_dir):
        data = load_checkpoint(path)
        if data:
            return data
    return None


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
        if bool(cfg.isolate_run_artifacts):
            isolate_run_paths(cfg)

        os.makedirs(cfg.output_dir, exist_ok=True)
        closed_loop_dir(cfg.output_dir)
        os.makedirs(cfg.surrogate_dataset, exist_ok=True)
        os.makedirs(cfg.al_root, exist_ok=True)

        ckpt = _load_ckpt(cfg.output_dir) if bool(cfg.resume) else None
        resumed = ckpt is not None

        pack = load_domain(str(getattr(cfg, "domain", "crowdnav") or "crowdnav"))
        score1_mode = "smoke" if cfg.use_stub else "dataset"
        if self.score_fn is None:
            self.score_fn, _ds = pack.make_score_fn(
                mode=score1_mode,
                dataset_path=None if score1_mode == "smoke" else cfg.stage1_dataset_path,
            )
        if self.trainer is None:
            self.trainer = make_stage2_trainer_for_domain(pack, use_stub=cfg.use_stub)
        if self.validator is None:
            from crowd_nav.domains import make_validator_for_domain

            self.validator = make_validator_for_domain(pack)
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
            prompts=pack.prompts,
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
        generations = int(cfg.generations)
        start_epoch = 0
        pending: Optional[Dict[str, Any]] = None
        population: List[RewardCandidate] = []

        cfg_dict = asdict(cfg)

        def _checkpoint(
            *,
            status: str,
            phase: str,
            epoch: int,
            next_epoch: int,
            population: Optional[List[RewardCandidate]] = None,
            ranked: Optional[List[RewardCandidate]] = None,
            to_label: Optional[List[RewardCandidate]] = None,
            labeled_ids: Optional[List[str]] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            paths = {
                "output_dir": cfg.output_dir,
                "surrogate_model_dir": cfg.surrogate_model_dir,
                "surrogate_dataset": cfg.surrogate_dataset,
                "al_root": cfg.al_root,
                "generations": generations,
            }
            payload = build_checkpoint(
                status=status,
                phase=phase,
                epoch=int(epoch),
                next_epoch=int(next_epoch),
                population=population,
                ranked=ranked,
                to_label=to_label,
                labeled_ids_this_epoch=labeled_ids or [],
                global_best=evolver.global_best,
                reflection=evolver.reflection,
                labels_since_refit=labels_since_refit,
                n_labeled=n_labeled,
                history=history,
                config=cfg_dict,
                extra={**paths, **(extra or {})},
            )
            # Mirror paths at top level so resume scripts can read them directly.
            payload.update(paths)
            _save_ckpt(cfg.output_dir, payload)

        if ckpt is not None:
            start_epoch = max(0, int(ckpt.get("next_epoch") or 0))
            history = [r for r in (ckpt.get("history") or []) if isinstance(r, dict)]
            labels_since_refit = int(ckpt.get("labels_since_refit") or 0)
            evolver.reflection = str(ckpt.get("reflection") or "")
            saved_best = ckpt.get("global_best")
            if isinstance(saved_best, dict):
                evolver.global_best = load_candidate_dict(dict(saved_best))
            population = deserialize_population(ckpt.get("population"))
            if str(ckpt.get("phase") or "") == "labeling" and start_epoch < generations:
                resumed_to_label = deserialize_population(ckpt.get("to_label"))
                if resumed_to_label:
                    saved_extra = ckpt.get("extra") or {}
                    pending = {
                        "epoch": start_epoch,
                        "ranked": (
                            deserialize_population(ckpt.get("ranked"))
                            or list(resumed_to_label)
                        ),
                        "to_label": resumed_to_label,
                        "done": {
                            str(x) for x in (ckpt.get("labeled_ids_this_epoch") or [])
                        },
                        "gate": dict(
                            saved_extra.get("gate_report")
                            or {"enabled": False, "soft": True, "reason": "resumed"}
                        ),
                        "al": dict(
                            saved_extra.get("al_report")
                            or {
                                "enabled": False,
                                "n_al_stage2": 0,
                                "n_stage1_requests": 0,
                            }
                        ),
                    }
            logger.info(
                "closed-loop resume: status=%s phase=%s next_epoch=%d epochs_done=%d",
                ckpt.get("status"),
                ckpt.get("phase"),
                start_epoch,
                len(history),
            )

        if not population:
            # Gen0 propose
            population = evolver.initialize_population()

        if pending is None and start_epoch < generations:
            _checkpoint(
                status="running",
                phase="init",
                epoch=start_epoch,
                next_epoch=start_epoch,
                population=population,
            )

        for g in range(start_epoch, generations):
            if pending is not None and int(pending["epoch"]) == g:
                # Mid-epoch resume: reuse the selection the interrupted run made.
                ranked = list(pending["ranked"])
                to_label = list(pending["to_label"])
                gate_report = dict(pending["gate"])
                al_report = dict(pending["al"])
                labeled_this_epoch = set(pending["done"])
                pending = None
                epoch_resumed = True
            else:
                labeled_this_epoch = set()
                epoch_resumed = False
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

                _checkpoint(
                    status="running",
                    phase="labeling",
                    epoch=g,
                    next_epoch=g,
                    population=population,
                    ranked=ranked,
                    to_label=to_label,
                    labeled_ids=[],
                    extra={"gate_report": gate_report, "al_report": al_report},
                )

            n_ok = 0
            n_fail = 0
            n_proxy_fb = 0
            # Score1 / scalar pools for mismatch detection (include prior labels on pop).
            pop_s1 = [
                float(c.score)
                for c in ranked
                if c.score is not None and float(c.score) == float(c.score)
            ]
            pop_sc = []
            for c in ranked:
                m = (c.metadata or {}).get("last_metrics")
                if isinstance(m, dict) and "SR" in m:
                    from crowd_nav.reward_search.raise_loop.proxy_feedback import nav_scalar

                    pop_sc.append(nav_scalar(m))

            for i, cand in enumerate(to_label):
                if str(cand.candidate_id) in labeled_this_epoch:
                    continue
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

                metrics = (cand.metadata or {}).get("last_metrics")
                if isinstance(metrics, dict) and metrics:
                    from crowd_nav.reward_search.raise_loop.proxy_feedback import (
                        attach_proxy_feedback,
                        nav_scalar,
                    )

                    pop_sc.append(nav_scalar(metrics))
                    if cand.score is not None:
                        pop_s1.append(float(cand.score))
                    if attach_proxy_feedback(
                        cand,
                        metrics,
                        enabled=bool(getattr(cfg, "proxy_feedback", False)),
                        n_labeled_dataset=int(n_labeled),
                        min_labels=int(getattr(cfg, "proxy_feedback_min_labels", 16)),
                        epoch=int(g),
                        population_score1=pop_s1,
                        population_scalars=pop_sc,
                    ):
                        n_proxy_fb += 1

                # Each label is a Stage II train/eval; checkpoint per candidate so an
                # interrupt costs at most one candidate of compute.
                labeled_this_epoch.add(str(cand.candidate_id))
                _checkpoint(
                    status="running",
                    phase="labeling",
                    epoch=g,
                    next_epoch=g,
                    population=population,
                    ranked=ranked,
                    to_label=to_label,
                    labeled_ids=sorted(labeled_this_epoch),
                    extra={"gate_report": gate_report, "al_report": al_report},
                )

            n_labeled = len(existing_example_ids(cfg.surrogate_dataset))
            known_ids = existing_example_ids(cfg.surrogate_dataset)

            # Optional in-loop D.3 on worst proxy candidates (budget-capped).
            d3_n = int(getattr(cfg, "proxy_feedback_d3_per_epoch", 0) or 0)
            if bool(getattr(cfg, "enable_refine", False)) and d3_n <= 0:
                d3_n = 1
            n_d3 = 0
            if (
                bool(getattr(cfg, "proxy_feedback", False))
                and d3_n > 0
                and int(g) >= 1
                and int(n_labeled) >= int(getattr(cfg, "proxy_feedback_min_labels", 16))
            ):
                from crowd_nav.reward_search.raise_loop.proxy_feedback import (
                    apply_in_loop_d3,
                    select_for_in_loop_d3,
                )

                targets = select_for_in_loop_d3(to_label, max_n=d3_n)
                for old in targets:
                    new_c = apply_in_loop_d3(
                        old, llm=self.llm, validator=self.validator
                    )
                    if new_c is old or new_c.candidate_id == old.candidate_id:
                        continue
                    n_d3 += 1
                    # Swap into ranked / to_label / population lists in-place.
                    for lst in (ranked, to_label, population):
                        for j, c in enumerate(lst):
                            if str(c.candidate_id) == str(old.candidate_id):
                                lst[j] = new_c
                    # Re-score only the rewritten genome so next_generation sees Score1.
                    try:
                        rescored = evolver.score_population([new_c])
                        if rescored:
                            new_c = rescored[0]
                            for lst in (ranked, to_label, population):
                                for j, c in enumerate(lst):
                                    if str(c.candidate_id) == str(new_c.candidate_id) or (
                                        str(c.candidate_id) == str(old.candidate_id)
                                    ):
                                        lst[j] = new_c
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("in-loop D.3 rescore failed: %s", exc)

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
                "n_proxy_feedback": n_proxy_fb,
                "n_in_loop_d3": n_d3,
            }
            if epoch_resumed:
                # Counts cover only the labels this process produced.
                epoch_rec["resumed"] = True
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

            if g + 1 < generations:
                population = evolver._next_generation(ranked)
            else:
                population = ranked

            _checkpoint(
                status="running",
                phase="epoch_done",
                epoch=g,
                next_epoch=g + 1,
                population=population,
                ranked=ranked,
            )

        n_labeled = len(existing_example_ids(cfg.surrogate_dataset))
        _checkpoint(
            status="running",
            phase="finishing",
            epoch=max(0, generations - 1),
            next_epoch=generations,
            population=population,
        )

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
            "al_root": cfg.al_root,
            "resumed": bool(resumed),
            "config": cfg_dict,
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

        _checkpoint(
            status="completed",
            phase="done",
            epoch=max(0, generations - 1),
            next_epoch=generations,
            population=population,
        )

        return ClosedLoopResult(
            population=list(population),
            history=history,
            n_labeled_total=n_labeled,
            model_dir=cfg.surrogate_model_dir,
            dataset_dir=cfg.surrogate_dataset,
            output_dir=cfg.output_dir,
            manifest=result_manifest,
            resumed=bool(resumed),
        )
