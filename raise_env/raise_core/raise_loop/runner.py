"""Closed-loop reward evolution runner.

Each generation: Score1 → Stage II PPO train+eval of every new candidate →
rank by fitness → LLM crossover/mutation (with evaluation feedback).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from raise_core import console
from raise_core.domains import get_pack, make_stage2_trainer, make_validator
from raise_core.candidate import RewardCandidate, evaluation_fields, genome_fingerprint
from raise_core.explore import StageIConfig, StageIEvolver
from raise_core.feedback import ArchiveRecord, CandidateArchive, component_diversity
from raise_core.evo_selection import LLMRanker
from raise_core.evo_selection.consistency import append_consistency_csv
from raise_core.evo_selection.guards import candidate_guard_flags
from raise_core.llm import LLMClient, make_llm_client
from raise_core.raise_loop.checkpoint import (
    build_checkpoint,
    deserialize_population,
    load_checkpoint,
    save_checkpoint,
)
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.evolve_rank import rank_population_for_evolution
from raise_core.raise_loop.logging_io import (
    append_epoch_record,
    closed_loop_dir,
    write_json,
)
from raise_core.raise_loop.parallel_label import (
    attach_warm_start_checkpoints,
    label_candidates_parallel,
)
from raise_core.raise_loop.proxy_feedback import attach_proxy_feedback
from raise_core.raise_loop.report import format_epoch_summary, write_closed_loop_report
from raise_core.refine import ProxyMetrics, Stage2Config
from raise_core.seed_aggregate import aggregate_seed_fits
from raise_core.selection import candidate_fitness, candidate_fitness_se, metrics_fitness
from raise_core.validate import _accepts_kwarg

logger = logging.getLogger(__name__)


@dataclass
class ClosedLoopResult:
    population: List[RewardCandidate]
    history: List[Dict[str, Any]] = field(default_factory=list)
    n_labeled_total: int = 0
    output_dir: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    resumed: bool = False


def _checkpoint_dirs(output_dir: str) -> List[str]:
    """Checkpoint locations, most specific first (``closed_loop/`` then run root)."""
    return [closed_loop_dir(output_dir), output_dir]


def _save_ckpt(output_dir: str, payload: Dict[str, Any]) -> None:
    for path in _checkpoint_dirs(output_dir):
        save_checkpoint(path, payload)


def _load_ckpt(output_dir: str) -> Optional[Dict[str, Any]]:
    corrupt: List[str] = []
    for path in _checkpoint_dirs(output_dir):
        try:
            data = load_checkpoint(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Unreadable closed-loop checkpoint in %s (%s)", path, exc)
            corrupt.append(path)
            continue
        if data:
            return data
    if corrupt:
        raise RuntimeError(
            "closed-loop resume: every checkpoint copy is unreadable "
            f"({corrupt}); refusing to silently restart from scratch"
        )
    return None


def _resume_known_ids(ckpt: Dict[str, Any], stage2_root: str) -> List[str]:
    """Every candidate id an interrupted run may already have issued."""
    ids: List[str] = []
    for key in ("population", "to_label"):
        ids.extend(
            str(row.get("candidate_id"))
            for row in (ckpt.get(key) or [])
            if isinstance(row, dict)
        )
    for rec in ckpt.get("history") or []:
        if not isinstance(rec, dict):
            continue
        for key in ("ranking", "labeled_ids", "evolve_ranking"):
            ids.extend(str(x) for x in (rec.get(key) or []))
    if os.path.isdir(stage2_root):
        for name in os.listdir(stage2_root):
            # Trainer folders are ``rXX_<candidate_id>``.
            head, sep, tail = name.partition("_")
            if sep and head.startswith("r") and head[1:].isdigit():
                ids.append(tail)
    return ids


def _store_stage2_metrics(candidate: RewardCandidate, metrics: ProxyMetrics) -> None:
    """Keep Stage II metrics on the candidate (domain trainer already wrote them)."""
    md = dict(candidate.metadata or {})
    if md.get("fitness") is None or not isinstance(md.get("last_metrics"), dict):
        md["last_metrics"] = metrics.as_dict()
        md["fitness"] = metrics_fitness(md["last_metrics"])
    candidate.metadata = md


_COMPACT_METRIC_KEYS = (
    "SR",
    "CR",
    "TR",
    "NT",
    "PL",
    "task_return",
    "fitness",
    "fitness_se",
)


def _compact_metrics(candidate: RewardCandidate) -> Dict[str, Any]:
    """Headline Stage II numbers for epoch records / plots."""
    md = candidate.metadata or {}
    metrics = md.get("last_metrics") if isinstance(md.get("last_metrics"), dict) else {}
    out = {k: metrics.get(k) for k in _COMPACT_METRIC_KEYS if metrics.get(k) is not None}
    # None (not -inf) for a candidate without a Stage II evaluation.
    out["fitness"] = _finite_or_none(candidate_fitness(candidate))
    return out


def _needs_training(candidate: RewardCandidate) -> bool:
    """Untrained executable genomes that Score1 did not hard-reject."""
    md = candidate.metadata or {}
    return (
        candidate.reward_fn is not None
        and md.get("fitness") is None
        and not md.get("score1_rejected")
    )


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


# Settings whose change mid-run mixes incomparable generations.
_RESUME_CONFIG_KEYS = (
    "domain",
    "population_size",
    "generations",
    "stage2_train_steps",
    "eval_episodes",
    "seed",
    "keep_runtime_elite",
    "n_crossover",
    "n_mutation",
    "n_random",
    "mutation_parents",
    "mutation_exclude_collapsed",
    "collapse_cr",
    "action_mode",
    "robot_kinematics",
    "action_speed_samples",
    "action_rotation_samples",
    "env_config_path",
    "eval_mode",
    "stage2_eval_split",
    "stage2_successive_halving",
    "stage2_paired_seeds",
    "stage2_seeds",
    "ppo_normalize_reward",
    "selection",
)

# Subset that changes numerical results; resume aborts unless allow_config_drift.
_RESULT_CHANGING_KEYS = (
    "stage2_train_steps",
    "eval_episodes",
    "seed",
    "population_size",
    "action_mode",
    "robot_kinematics",
    "stage2_eval_split",
    "stage2_successive_halving",
    "stage2_paired_seeds",
    "stage2_seeds",
    "ppo_normalize_reward",
    "collapse_cr",
    "env_config_path",
)

# Keys added by the patch: missing from a pre-patch checkpoint means LEGACY
# behaviour (not "current defaults"), and must not trigger drift-abort.
_LEGACY_ABSENT_CONFIG_DEFAULTS: Dict[str, Any] = {
    "stage2_paired_seeds": False,
    "stage2_seeds": 1,
    "ppo_normalize_reward": False,
}


def reject_successive_halving_with_multi_seed(use_sh: bool, n_stage2_seeds: int) -> None:
    """Successive-halving + K>1 shared seeds is unsupported."""
    if use_sh and int(n_stage2_seeds) > 1:
        raise ValueError(
            "stage2_successive_halving cannot be combined with "
            f"stage2_seeds={int(n_stage2_seeds)} (>1); use one or the other"
        )


def _normalize_resume_config(saved: Dict[str, Any]) -> Dict[str, Any]:
    """Fill absent patch keys with legacy values; warn once per missing key."""
    out = dict(saved)
    missing = [k for k in _LEGACY_ABSENT_CONFIG_DEFAULTS if k not in out]
    for key in missing:
        out[key] = _LEGACY_ABSENT_CONFIG_DEFAULTS[key]
    if missing:
        detail = ", ".join(
            f"{k}={_LEGACY_ABSENT_CONFIG_DEFAULTS[k]!r}" for k in missing
        )
        console.warn(
            f"pre-patch checkpoint missing keys; using legacy values ({detail})",
            stage="closed-loop",
        )
    return out


def _config_drift(
    saved: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, tuple]:
    return {
        key: (saved[key], current[key])
        for key in _RESUME_CONFIG_KEYS
        if key in saved and key in current and saved[key] != current[key]
    }


def _check_config_drift(
    saved: Dict[str, Any],
    current: Dict[str, Any],
    *,
    allow_config_drift: bool,
    already_normalized: bool = False,
) -> Optional[Dict[str, Any]]:
    """Abort on result-changing drift unless ``allow_config_drift``.

    Absent patch-era keys should already be filled with legacy values on both
    sides (see resume path) so a pre-patch checkpoint does not drift-abort.

    Returns ``config_at_start`` payload when drift was allowed (for the
    manifest); otherwise ``None``.
    """
    normalized = saved if already_normalized else _normalize_resume_config(saved)
    drift = _config_drift(normalized, current)
    if not drift:
        return None
    detail = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in drift.items())
    result_drift = {k: v for k, v in drift.items() if k in _RESULT_CHANGING_KEYS}
    if result_drift and not allow_config_drift:
        raise RuntimeError(
            "closed-loop resume aborted: result-changing config drift "
            f"({detail}). Re-run with matching settings or pass "
            "--allow-config-drift to continue anyway."
        )
    console.warn(
        f"resuming with settings that differ from the checkpoint ({detail})",
        stage="closed-loop",
    )
    return dict(normalized)


def _needs_reevaluation(candidate: RewardCandidate, generation: int) -> bool:
    """Trained genomes carried in from an earlier generation's evaluation."""
    md = candidate.metadata or {}
    return (
        candidate.reward_fn is not None
        and md.get("fitness") is not None
        and md.get("eval_generation") != int(generation)
    )


def _carried(candidate: RewardCandidate, generation: int) -> bool:
    """Carried genome of this generation, before or after its eval-only pass."""
    md = candidate.metadata or {}
    return (
        _needs_reevaluation(candidate, generation)
        or bool(md.get("fitness_history"))
        or bool(md.get("fitness_stale"))
    )


def _same_genome_copies(candidates: List[RewardCandidate]) -> Dict[str, RewardCandidate]:
    """Later candidates whose genome equals an earlier one's: ``copy id -> original``."""
    first: Dict[str, RewardCandidate] = {}
    copies: Dict[str, RewardCandidate] = {}
    for cand in candidates:
        fp = genome_fingerprint(cand.code)
        if fp in first:
            copies[str(cand.candidate_id)] = first[fp]
        else:
            first[fp] = cand
    return copies


_EVAL_KEYS = ("fitness", "selection_scalar", "last_metrics")


def _past_evaluation(md: Dict[str, Any]) -> Dict[str, Any]:
    metrics = md.get("last_metrics") if isinstance(md.get("last_metrics"), dict) else {}
    return {
        "generation": md.get("eval_generation"),
        "fitness": md.get("fitness"),
        "fitness_se": metrics.get("fitness_se"),
        "eval_seed_base": md.get("eval_seed_base"),
    }


class ClosedLoopRunner:
    """Generation epochs: Score1 → Stage II train+eval → fitness rank → LLM breeding."""

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
        cfg = self.cfg
        if cfg.use_stub:
            cfg.apply_fast_profile()

        os.makedirs(cfg.output_dir, exist_ok=True)
        closed_loop_dir(cfg.output_dir)

        ckpt = _load_ckpt(cfg.output_dir) if bool(cfg.resume) else None
        resumed = ckpt is not None

        pack = get_pack()
        pack.apply_config(cfg)

        score1_mode = "smoke" if cfg.use_stub else "dataset"
        if self.score_fn is None:
            self.score_fn, _ds = pack.make_score_fn(
                mode=score1_mode,
                dataset_path=None if score1_mode == "smoke" else cfg.stage1_dataset_path,
            )
        if self.trainer is None:
            self.trainer = make_stage2_trainer(pack, use_stub=cfg.use_stub)
        if self.validator is None:
            self.validator = make_validator(pack)
        if self.llm is None:
            self.llm = make_llm_client(
                cfg.llm_provider,
                base_code=getattr(pack, "seed_reward_source", None),
                variant_fn=getattr(pack.prompts, "seed_variant", None),
            )

        s1_cfg = StageIConfig(
            population_size=int(cfg.population_size),
            generations=int(cfg.generations),
            n_crossover=cfg.n_crossover,
            n_mutation=cfg.n_mutation,
            n_random=cfg.n_random,
            keep_runtime_elite=bool(cfg.keep_runtime_elite),
            mutation_parents=str(cfg.mutation_parents),
            mutation_exclude_collapsed=bool(
                getattr(cfg, "mutation_exclude_collapsed", False)
            ),
            collapse_cr=float(getattr(cfg, "collapse_cr", 0.8) or 0.8),
            seed=int(cfg.seed),
            max_fallback_fraction=float(
                getattr(cfg, "max_fallback_fraction", 0.5) or 0.5
            ),
            strict_fallback=bool(getattr(cfg, "strict_fallback", False)),
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
            feedback=cfg.feedback,
            feedback_card=pack.feedback_card,
            llm_log_dir=closed_loop_dir(cfg.output_dir),
        )
        selection_cfg = cfg.selection
        if selection_cfg.mode == "llm_rank" and not cfg.feedback.enabled:
            raise ValueError("selection.mode='llm_rank' requires feedback cards enabled")
        ranker = (
            LLMRanker(
                self.llm,
                selection_cfg,
                log_path=os.path.join(
                    closed_loop_dir(cfg.output_dir), "llm_selection.jsonl"
                ),
            )
            if selection_cfg.mode == "llm_rank"
            else None
        )
        card_on = bool(cfg.feedback.enabled)

        eval_eps = getattr(cfg, "eval_episodes", None)
        if eval_eps is None:
            eval_eps = 8 if cfg.use_stub else 30
        else:
            eval_eps = max(1, int(eval_eps))

        stage2_cfg = Stage2Config(
            train_env_steps=int(cfg.stage2_train_steps),
            eval_episodes=int(eval_eps),
            seed=int(cfg.seed),
            device=str(cfg.device),
            output_root=os.path.join(cfg.output_dir, "closed_loop", "stage2_train"),
            n_envs=max(1, int(cfg.n_envs or 1)),
            warm_start=bool(cfg.warm_start),
            eval_mode=str(cfg.eval_mode or "both"),
            eval_split=str(getattr(cfg, "stage2_eval_split", None) or "val"),
            action_mode=str(cfg.action_mode or "continuous"),
            robot_kinematics=str(cfg.robot_kinematics or "unicycle"),
            action_speed_samples=int(cfg.action_speed_samples),
            action_rotation_samples=int(cfg.action_rotation_samples),
            env_config_path=cfg.env_config_path,
            record_training_dynamics=card_on and bool(cfg.feedback.training_dynamics),
            record_reward_components=card_on and bool(cfg.feedback.component_instrumentation),
        )

        can_reevaluate = callable(getattr(self.trainer, "reevaluate", None))
        if not can_reevaluate:
            logger.warning(
                "Stage II trainer %s has no reevaluate(); carried candidates keep "
                "the fitness of the generation that evaluated them",
                type(self.trainer).__name__,
            )

        n_labeled = 0
        history: List[Dict[str, Any]] = []
        selection_notes: List[str] = []
        generations = int(cfg.generations)
        start_epoch = 0
        pending: Optional[Dict[str, Any]] = None
        population: List[RewardCandidate] = []

        cfg_dict = asdict(cfg)
        cfg_dict.pop("env_config_obj", None)
        config_at_start: Optional[Dict[str, Any]] = None
        # Patch-era keys filled from LEGACY defaults because they were absent in
        # the checkpoint. Omit them when persisting so a later resume still
        # treats them as absent (no false drift against post-patch defaults).
        legacy_absent_keys: set[str] = set()

        def _checkpoint(
            *,
            status: str,
            phase: str,
            epoch: int,
            next_epoch: int,
            population: Optional[List[RewardCandidate]] = None,
            to_label: Optional[List[RewardCandidate]] = None,
            labeled_ids: Optional[List[str]] = None,
            sh_phase: Optional[int] = None,
            sh_survivor_ids: Optional[List[str]] = None,
        ) -> None:
            extra: Dict[str, Any] = {
                "id_counter": evolver.id_counter,
                "n_labeled": n_labeled,
                "generations": generations,
                "selection_notes": selection_notes,
            }
            if sh_phase is not None:
                extra["sh_phase"] = int(sh_phase)
            if sh_survivor_ids is not None:
                extra["sh_survivor_ids"] = [str(x) for x in sh_survivor_ids]
            config_payload = {
                k: v for k, v in cfg_dict.items() if k not in legacy_absent_keys
            }
            payload = build_checkpoint(
                status=status,
                phase=phase,
                epoch=int(epoch),
                next_epoch=int(next_epoch),
                population=population,
                to_label=to_label,
                labeled_ids_this_epoch=labeled_ids or [],
                reflection=evolver.reflection,
                history=history,
                config=config_payload,
                extra=extra,
            )
            payload["output_dir"] = cfg.output_dir
            _save_ckpt(cfg.output_dir, payload)

        if ckpt is not None:
            raw_saved = dict(ckpt.get("config") or {})
            saved_cfg = _normalize_resume_config(raw_saved)
            # Keys absent from the pre-patch checkpoint take LEGACY values on
            # the live run so resume is continuous and does not drift-abort.
            for key, legacy in _LEGACY_ABSENT_CONFIG_DEFAULTS.items():
                if key not in raw_saved:
                    setattr(cfg, key, legacy)
                    cfg_dict[key] = legacy
                    legacy_absent_keys.add(key)
            config_at_start = _check_config_drift(
                saved_cfg,
                cfg_dict,
                allow_config_drift=bool(getattr(cfg, "allow_config_drift", False)),
                already_normalized=True,
            )
            start_epoch = max(0, int(ckpt.get("next_epoch") or 0))
            history = [r for r in (ckpt.get("history") or []) if isinstance(r, dict)]
            evolver.reflection = str(ckpt.get("reflection") or "")
            population = deserialize_population(
                ckpt.get("population"), validator=self.validator
            )
            evolver.archive = CandidateArchive.from_epoch_files(closed_loop_dir(cfg.output_dir))
            ckpt_extra = ckpt.get("extra") or {}
            n_labeled = int(ckpt_extra.get("n_labeled") or 0)
            saved_notes = [
                str(note)
                for note in (ckpt_extra.get("selection_notes") or [])
                if str(note).strip()
            ]
            selection_notes = (
                saved_notes[-selection_cfg.reflection_note_cap :]
                if selection_cfg.reflection_note_cap
                else []
            )
            evolver.restore_id_counter(
                ckpt_extra.get("id_counter"),
                existing_ids=_resume_known_ids(ckpt, stage2_cfg.output_root),
            )
            if str(ckpt.get("phase") or "") == "labeling" and start_epoch < generations:
                resumed_to_label = deserialize_population(
                    ckpt.get("to_label"), validator=self.validator
                )
                if population:
                    pending = {
                        "epoch": start_epoch,
                        "to_label": resumed_to_label,
                        "done": {
                            str(x) for x in (ckpt.get("labeled_ids_this_epoch") or [])
                        },
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
                # Mid-epoch resume: population was already scored; finish its labels.
                ranked = list(population)
                labeled_ids = {str(c.candidate_id) for c in pending["to_label"]}
                to_label = [c for c in ranked if str(c.candidate_id) in labeled_ids]
                labeled_this_epoch = set(pending["done"])
                pending = None
                epoch_resumed = True
            else:
                labeled_this_epoch = set()
                epoch_resumed = False
                ranked = evolver.score_population(population)
                to_label = [
                    c for c in ranked
                    if _needs_training(c) or (can_reevaluate and _needs_reevaluation(c, g))
                ]
                # ``population`` is now the scored list so a labeling resume
                # skips re-scoring.
                population = ranked
                _checkpoint(
                    status="running",
                    phase="labeling",
                    epoch=g,
                    next_epoch=g,
                    population=population,
                    to_label=to_label,
                    labeled_ids=[],
                )

            n_ok = 0
            n_fail = 0
            n_pending = sum(
                1 for c in to_label if str(c.candidate_id) not in labeled_this_epoch
            )
            console.status(
                f"epoch {g}: Stage II labeling {n_pending} candidate(s) "
                f"(workers={max(1, int(cfg.label_workers or 1))})",
                stage="closed-loop",
            )

            n_stage2_seeds = max(1, int(getattr(cfg, "stage2_seeds", 1) or 1))
            paired_seeds = bool(getattr(cfg, "stage2_paired_seeds", True))
            infra_policy = str(
                getattr(cfg, "infra_error_policy", "retry_then_abort") or "retry_then_abort"
            )
            epoch_infra_fails = 0
            epoch_reward_fails = 0

            def _call_train(cand: RewardCandidate, round_index: int, train_seed_index: Optional[int]):
                kwargs: Dict[str, Any] = {}
                if train_seed_index is not None and _accepts_kwarg(
                    self.trainer.train_and_eval, "train_seed_index"
                ):
                    kwargs["train_seed_index"] = int(train_seed_index)
                return self.trainer.train_and_eval(
                    cand, round_index=int(round_index), config=stage2_cfg, **kwargs
                )

            def _label_one(cand: RewardCandidate, round_index: int) -> Dict[str, Any]:
                if not _needs_reevaluation(cand, g):
                    seed_fits: List[float] = []
                    seed_runs: List[Dict[str, Any]] = []
                    last_episode_se = 0.0
                    for slot in range(n_stage2_seeds):
                        tsi = (int(g) * 1000 + int(slot)) if paired_seeds else None
                        metrics = _call_train(cand, int(round_index), tsi)
                        _store_stage2_metrics(cand, metrics)
                        md = dict(cand.metadata or {})
                        fit = float(md.get("fitness") or metrics_fitness(md.get("last_metrics")))
                        last = md.get("last_metrics") if isinstance(md.get("last_metrics"), dict) else {}
                        last_episode_se = float(last.get("fitness_se") or 0.0)
                        seed_fits.append(fit)
                        seed_runs.append(
                            {
                                "seed_slot": slot,
                                "train_seed_index": tsi,
                                "train_seed": md.get("train_seed"),
                                "status": "ok",
                                "fitness": fit,
                                "checkpoint_path": md.get("checkpoint_path"),
                                "actual_train_steps": md.get("actual_train_steps"),
                            }
                        )
                        # Clear fitness so the next seed's store is not skipped.
                        if n_stage2_seeds > 1 and slot + 1 < n_stage2_seeds:
                            md.pop("fitness", None)
                            md.pop("last_metrics", None)
                            cand.metadata = md
                    agg = aggregate_seed_fits(seed_fits, episode_se=last_episode_se)
                    md = dict(cand.metadata or {})
                    last = dict(md.get("last_metrics") or {})
                    last["fitness"] = float(agg["fitness"])
                    last["fitness_se"] = float(agg["fitness_se"])
                    last["fitness_se_source"] = str(agg["fitness_se_source"])
                    last["stage2_seed_fits"] = list(seed_fits)
                    md["last_metrics"] = last
                    md["fitness"] = float(agg["fitness"])
                    md["selection_scalar"] = float(agg["fitness"])
                    md["fitness_se_source"] = str(agg["fitness_se_source"])
                    md["stage2_n_seeds"] = n_stage2_seeds
                    md["stage2_seed_runs"] = seed_runs
                    md["stage2_paired_seeds"] = bool(paired_seeds)
                    md["eval_generation"] = g
                    cand.metadata = md
                    return {"status": "ok", "kind": "trained"}
                # Carried genome: same scenarios as this generation, no training.
                prev = dict(cand.metadata or {})
                hist = list(prev.get("fitness_history") or []) + [_past_evaluation(prev)]
                cand.metadata = {k: v for k, v in prev.items() if k not in _EVAL_KEYS}
                try:
                    re_kwargs: Dict[str, Any] = {}
                    if paired_seeds and _accepts_kwarg(
                        self.trainer.reevaluate, "train_seed_index"
                    ):
                        re_kwargs["train_seed_index"] = int(g) * 1000
                    metrics = self.trainer.reevaluate(
                        cand, round_index=int(round_index), config=stage2_cfg, **re_kwargs
                    )
                except Exception:
                    cand.metadata = {**prev, "fitness_stale": True}
                    raise
                _store_stage2_metrics(cand, metrics)
                cand.metadata = {
                    **cand.metadata,
                    "eval_generation": g,
                    "fitness_history": hist,
                    "fitness_stale": False,
                }
                return {"status": "ok", "kind": "reevaluated"}

            def _on_done(cand: RewardCandidate, result: Dict[str, Any], _i: int) -> None:
                nonlocal n_ok, n_fail, n_labeled, epoch_infra_fails, epoch_reward_fails
                status = str(result.get("status") or "")
                if status == "ok":
                    if result.get("kind") != "reevaluated":
                        n_ok += 1
                        n_labeled += 1
                    attach_proxy_feedback(
                        cand,
                        (cand.metadata or {}).get("last_metrics") or {},
                        evidence_line=pack.evidence_line,
                    )
                    labeled_this_epoch.add(str(cand.candidate_id))
                else:
                    # Not marked done (except abort checkpoint): resume retries.
                    if not _carried(cand, g):
                        n_fail += 1
                    kind = str(result.get("failure_kind") or "reward")
                    if kind in ("infra", "unknown"):
                        epoch_infra_fails += 1
                    else:
                        epoch_reward_fails += 1
                    md = dict(cand.metadata or {})
                    md["failure_kind"] = kind
                    if status == "aborted_infra":
                        md["abort_traceback"] = result.get("traceback")
                    cand.metadata = md
                    logger.warning(
                        "Stage II failed for %s (%s): %s",
                        cand.candidate_id,
                        kind,
                        result.get("reason"),
                    )
                # Write checkpoint BEFORE LabelAbortError propagates.
                _checkpoint(
                    status="aborted_infra" if status == "aborted_infra" else "running",
                    phase="labeling",
                    epoch=g,
                    next_epoch=g,
                    population=population,
                    to_label=to_label,
                    labeled_ids=sorted(labeled_this_epoch),
                    sh_phase=sh_phase_state,
                    sh_survivor_ids=sh_survivor_ids_state,
                )

            stage2_cfg.eval_seed_base = int(cfg.seed) + int(g) * 1000
            stage2_cfg.ppo_normalize_reward = bool(
                getattr(cfg, "ppo_normalize_reward", False)
            )
            to_train = [c for c in to_label if not _needs_reevaluation(c, g)]
            copies = _same_genome_copies(to_train)
            attach_warm_start_checkpoints(to_train, list(population))
            use_sh = bool(getattr(cfg, "stage2_successive_halving", False))
            reject_successive_halving_with_multi_seed(use_sh, n_stage2_seeds)
            # Immutable K2 from the run config (never from a mutated stage2_cfg).
            k2_full = int(cfg.stage2_train_steps)
            warm_default = bool(stage2_cfg.warm_start)
            sh_phase_state: Optional[int] = None
            sh_survivor_ids_state: Optional[List[str]] = None
            ckpt_extra = (ckpt or {}).get("extra") or {} if epoch_resumed else {}
            resume_sh_phase = int(ckpt_extra.get("sh_phase") or 0) if epoch_resumed else 0
            resume_survivors = [
                str(x) for x in (ckpt_extra.get("sh_survivor_ids") or [])
            ] if epoch_resumed else []

            def _run_labels(cands, *, already_done, round_index_fn):
                return label_candidates_parallel(
                    cands,
                    already_done=already_done,
                    workers=max(1, int(cfg.label_workers or 1)),
                    label_one=_label_one,
                    on_done=_on_done,
                    round_index_fn=round_index_fn,
                    infra_error_policy=infra_policy,
                )

            phase1_steps = max(1, k2_full // 3) if use_sh else k2_full
            phase2_steps = max(1, k2_full - phase1_steps) if use_sh else 0
            try:
                if use_sh and to_train and resume_sh_phase < 2:
                    sh_phase_state = 1
                    stage2_cfg.train_env_steps = phase1_steps
                    console.status(
                        f"epoch {g}: successive-halving phase 1 "
                        f"({phase1_steps:,} of {k2_full:,} steps)",
                        stage="closed-loop",
                    )
                    _run_labels(
                        to_label,
                        already_done=labeled_this_epoch | set(copies),
                        round_index_fn=lambda i: g * 1000 + i,
                    )
                elif not use_sh or not to_train:
                    _run_labels(
                        to_label,
                        already_done=labeled_this_epoch | set(copies),
                        round_index_fn=lambda i: g * 1000 + i,
                    )
                if use_sh and to_train:
                    # Continue the top half from their own phase-1 checkpoints.
                    if resume_sh_phase >= 2 and resume_survivors:
                        by_id = {str(c.candidate_id): c for c in to_train}
                        survivors = [by_id[i] for i in resume_survivors if i in by_id]
                        trained_ok = survivors
                    else:
                        trained_ok = [
                            c
                            for c in to_train
                            if str(c.candidate_id) not in copies
                            and (c.metadata or {}).get("fitness") is not None
                        ]
                        trained_ok.sort(
                            key=lambda c: float(
                                (c.metadata or {}).get("fitness") or float("-inf")
                            ),
                            reverse=True,
                        )
                        n_keep = max(1, (len(trained_ok) + 1) // 2)
                        survivors = trained_ok[:n_keep]
                    sh_phase_state = 2
                    sh_survivor_ids_state = [str(c.candidate_id) for c in survivors]
                    console.status(
                        f"epoch {g}: successive-halving phase 2 "
                        f"({len(survivors)}/{len(trained_ok)} survivors, "
                        f"{phase2_steps:,} steps, warm-start from own ckpt)",
                        stage="closed-loop",
                    )
                    stage2_cfg.train_env_steps = phase2_steps
                    stage2_cfg.warm_start = True
                    for c in survivors:
                        md = dict(c.metadata or {})
                        # Preserve phase-1 metrics before clearing live fitness.
                        md["sh_phase1"] = {
                            "fitness": md.get("fitness"),
                            "last_metrics": md.get("last_metrics"),
                            "eval_generation": md.get("eval_generation"),
                            "actual_train_steps": md.get("actual_train_steps"),
                            "checkpoint_path": md.get("checkpoint_path"),
                            "train_seed": md.get("train_seed"),
                        }
                        ckpt_path = md.get("checkpoint_path")
                        if ckpt_path:
                            md["warm_start_checkpoint"] = Path(str(ckpt_path)).as_posix()
                        md.pop("fitness", None)
                        md.pop("last_metrics", None)
                        md.pop("eval_generation", None)
                        c.metadata = md
                        labeled_this_epoch.discard(str(c.candidate_id))
                    _checkpoint(
                        status="running",
                        phase="labeling",
                        epoch=g,
                        next_epoch=g,
                        population=population,
                        to_label=to_label,
                        labeled_ids=sorted(labeled_this_epoch),
                        sh_phase=2,
                        sh_survivor_ids=sh_survivor_ids_state,
                    )
                    # Bookkeeping index uses +500; train seeds stay shared slots.
                    _run_labels(
                        survivors,
                        already_done=set(),
                        round_index_fn=lambda i: g * 1000 + 500 + i,
                    )
                    for c in survivors:
                        md = dict(c.metadata or {})
                        phase1 = md.get("sh_phase1") if isinstance(md.get("sh_phase1"), dict) else {}
                        if md.get("fitness") is None:
                            # Phase 2 failed: restore phase-1 metrics.
                            if phase1.get("fitness") is not None:
                                md["fitness"] = phase1.get("fitness")
                                md["last_metrics"] = phase1.get("last_metrics")
                                md["eval_generation"] = phase1.get("eval_generation")
                                md["checkpoint_path"] = phase1.get("checkpoint_path")
                                md["actual_train_steps"] = phase1.get("actual_train_steps")
                            md["sh_phase2_failed"] = True
                            console.warn(
                                f"{c.candidate_id}: SH phase 2 failed; "
                                "keeping phase-1 fitness",
                                stage="closed-loop",
                            )
                        else:
                            p1_steps = int(phase1.get("actual_train_steps") or phase1_steps)
                            p2_steps = int(md.get("actual_train_steps") or phase2_steps)
                            md["actual_train_steps"] = p1_steps + p2_steps
                            md["sh_phase2_failed"] = False
                        md["successive_halving"] = {
                            "phase1_steps": phase1_steps,
                            "phase2_steps": phase2_steps,
                            "k2": k2_full,
                            "optimizer_reset_at_phase2": True,
                            "note": (
                                "Phase 2 builds a fresh PPO instance and warm-starts "
                                "policy weights only; optimizer state is reset."
                            ),
                        }
                        md["train_steps"] = k2_full
                        c.metadata = md
            finally:
                stage2_cfg.train_env_steps = k2_full
                stage2_cfg.warm_start = warm_default
                sh_phase_state = None
                sh_survivor_ids_state = None
            for cand in to_train:
                original = copies.get(str(cand.candidate_id))
                if original is not None and (original.metadata or {}).get("fitness") is not None:
                    cand.metadata = {
                        **(cand.metadata or {}),
                        **evaluation_fields(original.metadata),
                        "duplicate_of": original.candidate_id,
                    }

            selection_payload: Dict[str, Any] = {}
            rank_consistency = None
            if ranker is not None:
                evolver.archive.add_population(ranked, generation=g)
                fallback_population = rank_population_for_evolution(ranked)
                # Fitness-first render; may drop weak cards to fit input budget.
                cards = evolver.render_selection_cards(fallback_population)
                card_ids = {cid for cid, _ in cards}
                fallback_order = [
                    c.candidate_id
                    for c in fallback_population
                    if c.candidate_id in card_ids
                ]
                result = ranker.rank(
                    cards,
                    fallback_order=fallback_order,
                    seed=int(selection_cfg.seed) + g,
                    stage="stage2",
                    generation=g,
                    flags={
                        candidate.candidate_id: candidate_guard_flags(
                            candidate, selection_cfg
                        )
                        for candidate in ranked
                        if candidate.candidate_id in card_ids
                    },
                )
                by_id = {candidate.candidate_id: candidate for candidate in ranked}
                ranked_llm = [
                    by_id[candidate_id]
                    for candidate_id in result.ranking
                    if candidate_id in by_id
                ]
                # Candidates dropped from the ranking prompt stay at the tail.
                ranked_tail = [
                    c for c in fallback_population if c.candidate_id not in card_ids
                ]
                ranked = ranked_llm + ranked_tail
                note_candidates = [
                    f"{candidate_id}: {result.rationale[candidate_id]}"
                    for candidate_id in result.ranking[: min(3, len(result.ranking))]
                    if candidate_id in result.rationale
                ]
                if note_candidates and selection_cfg.reflection_note_cap:
                    selection_notes = (
                        selection_notes
                        + [f"Gen {g} reflection: " + " | ".join(note_candidates)]
                    )[-selection_cfg.reflection_note_cap :]
                for candidate in ranked_llm:
                    candidate.metadata = {
                        **(candidate.metadata or {}),
                        "llm_selection_rank": (
                            result.ranking.index(candidate.candidate_id) + 1
                        ),
                        "llm_selection_score": result.scores.get(
                            candidate.candidate_id
                        ),
                        "selection_weaknesses": result.weaknesses.get(
                            candidate.candidate_id, []
                        ),
                        "selection_rationale": result.rationale.get(
                            candidate.candidate_id, ""
                        ),
                        "selection_reflection_notes": list(selection_notes),
                    }
                if history and history[-1].get("ranking"):
                    rank_consistency = append_consistency_csv(
                        os.path.join(
                            closed_loop_dir(cfg.output_dir), "rank_consistency.csv"
                        ),
                        stage="stage2_generation",
                        generation=g,
                        previous=history[-1]["ranking"],
                        current=result.ranking,
                        top_k=selection_cfg.top_k,
                    )
                selection_payload = {
                    "ranking": result.ranking,
                    "scores": result.scores,
                    "weaknesses": result.weaknesses,
                    "rationale": result.rationale,
                    "fallback": result.fallback,
                    "fallback_reason": result.fallback_reason,
                    "rank_consistency": rank_consistency,
                }
            ranked_for_evo = (
                list(ranked)
                if ranker is not None
                else rank_population_for_evolution(ranked)
            )
            evolver.archive.add_population(ranked_for_evo, generation=g)
            evolver.reflection = evolver._build_reflection(
                ranked_for_evo,
                generation=g,
                fitness_trend=[rec.get("evolve_best_fitness") for rec in history],
            )
            if selection_notes:
                evolver.reflection += " || LLM selection notes: " + " | ".join(
                    selection_notes
                )

            trained = [
                c for c in to_label
                if not _carried(c, g) and str(c.candidate_id) not in copies
            ]
            trained_ok = [c for c in trained if (c.metadata or {}).get("fitness") is not None]
            carried = [c for c in to_label if _carried(c, g)]
            previous_ids = set(history[-1].get("ranking") or []) if history else set()
            epoch_rec = {
                "epoch": g,
                "n_population": len(ranked),
                "n_unique_genomes": len({genome_fingerprint(c.code) for c in ranked}),
                "component_diversity": component_diversity([c.code for c in ranked]),
                # How this generation was actually built (fallback clones show
                # up as ``*_fallback``); ``n_carried`` came from the last one.
                "origin_counts": dict(
                    Counter(c.origin for c in ranked if c.candidate_id not in previous_ids)
                ),
                "n_carried": sum(1 for c in ranked if c.candidate_id in previous_ids),
                "n_to_label": len(trained),
                # ``labeled_ids`` trained successfully; ``failed_ids`` did not.
                "labeled_ids": [c.candidate_id for c in trained_ok],
                "failed_ids": [c.candidate_id for c in trained if c not in trained_ok],
                # Score1 hard-rejected: never sent to PPO.
                "skipped_score1_ids": [
                    c.candidate_id
                    for c in ranked
                    if (c.metadata or {}).get("score1_rejected")
                    and (c.metadata or {}).get("fitness") is None
                ],
                # Same genome as a candidate trained this generation: took over
                # its evaluation instead of a second training run.
                "duplicate_ids": [c.candidate_id for c in to_train if str(c.candidate_id) in copies],
                # Carried genomes evaluated again on this generation's seeds
                # (no training); ``stale_fitness_ids`` failed that pass.
                "reevaluated_ids": [
                    c.candidate_id for c in carried if not (c.metadata or {}).get("fitness_stale")
                ],
                "stale_fitness_ids": [
                    c.candidate_id for c in carried if (c.metadata or {}).get("fitness_stale")
                ],
                "label_metrics": {
                    str(c.candidate_id): _compact_metrics(c) for c in to_label
                },
                "ranking": [c.candidate_id for c in ranked],
                "selection": selection_payload,
                "best_score1": ranked[0].score if ranked else None,
                "evolve_ranking": [c.candidate_id for c in ranked_for_evo],
                "evolve_best_id": (
                    ranked_for_evo[0].candidate_id if ranked_for_evo else None
                ),
                "evolve_best_score1": (
                    ranked_for_evo[0].score if ranked_for_evo else None
                ),
                "evolve_best_fitness": (
                    _finite_or_none(candidate_fitness(ranked_for_evo[0]))
                    if ranked_for_evo
                    else None
                ),
                # Carried elites are not in label_metrics, so the plots read it here.
                "evolve_best_fitness_se": (
                    candidate_fitness_se(ranked_for_evo[0]) if ranked_for_evo else None
                ),
                "n_label_ok": n_ok,
                "n_label_failed": n_fail,
                "n_infra_fail": epoch_infra_fails,
                "n_reward_fail": epoch_reward_fails,
                "n_labeled_total": n_labeled,
                "fitness_se_source": next(
                    (
                        str((c.metadata or {}).get("fitness_se_source"))
                        for c in ranked_for_evo
                        if (c.metadata or {}).get("fitness_se_source")
                    ),
                    "episodes_only",
                ),
            }
            # Fallback contamination counters from the evolver (item 6).
            fb = getattr(evolver, "last_fallback_stats", None) or {}
            if fb:
                epoch_rec["n_fallback_slots"] = fb.get("n_fallback_slots", 0)
                epoch_rec["fallback_fraction"] = fb.get("fallback_fraction", 0.0)
            if epoch_resumed:
                # Counts cover only the labels this process produced.
                epoch_rec["resumed"] = True
            append_epoch_record(cfg.output_dir, epoch_rec)
            history.append(epoch_rec)
            write_json(
                os.path.join(closed_loop_dir(cfg.output_dir), f"pop_epoch_{g:03d}.json"),
                {
                    "to_label": [c.candidate_id for c in to_label],
                    # Archive fields (generation, lineage, code, Stage II
                    # metadata) let a resumed run rebuild the feedback archive.
                    "candidates": [
                        {
                            **(
                                evolver.archive.get(c.candidate_id)
                                or ArchiveRecord.from_candidate(c, generation=g)
                            ).to_dict(),
                            "candidate_id": str(c.candidate_id),
                            "score": c.score,
                            "fitness": (c.metadata or {}).get("fitness"),
                            "last_metrics": (c.metadata or {}).get("last_metrics"),
                        }
                        for c in ranked_for_evo
                    ],
                },
            )

            console.status(format_epoch_summary(epoch_rec), stage="closed-loop")

            if g + 1 < generations:
                population = (
                    evolver._next_generation_llm_rank(
                        ranked_for_evo,
                        top_k=selection_cfg.top_k,
                        p_restart=selection_cfg.p_restart,
                        seed=selection_cfg.seed,
                        generation=g,
                    )
                    if ranker is not None
                    else evolver._next_generation(ranked_for_evo)
                )
            else:
                population = ranked_for_evo

            _checkpoint(
                status="running",
                phase="epoch_done",
                epoch=g,
                next_epoch=g + 1,
                population=population,
            )

        best = max(population, key=candidate_fitness) if population else None
        best_origin = str(getattr(best, "origin", "") or "") if best else None
        fallback_origins = {
            "seed_fallback",
            "crossover_fallback",
            "mutation_fallback",
        }
        best_is_fallback = bool(
            best is not None
            and (
                best_origin in fallback_origins
                or bool((best.metadata or {}).get("llm_fallback_clone"))
            )
        )
        # run_quality is separate from checkpoint status (running/completed).
        run_quality = getattr(evolver, "run_status", None) or "ok"
        result_manifest = {
            "mode": "closed_loop",
            "status": "completed",
            "run_quality": run_quality,
            "n_epochs": len(history),
            "n_labeled_total": n_labeled,
            "resumed": bool(resumed),
            "config": cfg_dict,
            "operators": evolver.operators.to_dict(),
            "history": history,
            "selection_notes": list(selection_notes),
            "final_winner_origin": best_origin,
            "final_winner_is_fallback": best_is_fallback,
            "n_infra_fail_total": sum(int(r.get("n_infra_fail") or 0) for r in history),
            "n_reward_fail_total": sum(int(r.get("n_reward_fail") or 0) for r in history),
        }
        if config_at_start is not None and config_at_start != cfg_dict:
            result_manifest["config_at_start"] = config_at_start
        write_json(
            os.path.join(closed_loop_dir(cfg.output_dir), "manifest.json"),
            result_manifest,
        )
        try:
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
            output_dir=cfg.output_dir,
            manifest=result_manifest,
            resumed=bool(resumed),
        )
