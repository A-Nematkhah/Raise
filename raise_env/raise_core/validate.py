"""
RAISE Stage III — long PPO training + evaluation of the final candidates.

Every candidate of the final closed-loop population is trained once for K3
environment steps and evaluated on E3 episodes. The best is the trained
candidate with the highest fitness. Crash-safe: progress is checkpointed
after each candidate under ``{run}/stage3/checkpoint.json``.

Every candidate trains with the same seed and is evaluated on scenarios
(``seed + STAGE3_EVAL_SEED_OFFSET``) that no closed-loop generation saw. The
winner's fitness is a max over noisy estimates, so it is biased upward; the
pipeline re-evaluates the winner alone on a further unseen seed family
(``seed + FINAL_EVAL_SEED_OFFSET``) of the domain's ``test`` split (selection
evals use ``val``) for the number to report.
"""

from __future__ import annotations

import logging
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

from raise_core import console
from raise_core.candidate import RewardCandidate
from raise_core.population_io import deserialize_population, load_candidate_dict
from raise_core.refine import ProxyMetrics, stub_metrics_from_code
from raise_core.sandbox import RewardValidator
from raise_core.selection import (
    candidate_fitness,
    candidate_fitness_se,
    metrics_fitness,
    statistical_ties,
)
from raise_core.stage3_checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    stage3_dir_from_train_root,
)

logger = logging.getLogger(__name__)

# K3 default (environment steps per candidate).
STAGE3_STEPS: int = 70_000

# Eval seed bases relative to the run seed. Closed-loop generation g uses
# ``seed + g * 1000``, so these stay clear of any realistic G.
STAGE3_EVAL_SEED_OFFSET: int = 10_000_000
FINAL_EVAL_SEED_OFFSET: int = 20_000_000
# Scenario split of the final blind evaluation (never used for selection).
FINAL_EVAL_SPLIT: str = "test"

# Stage II results that must not leak into Stage III selection / final eval.
# Values still useful for reports are preserved as ``stage2_*`` in ``_stage3_input``.
_STAGE2_RESULT_KEYS = (
    "last_metrics",
    "fitness",
    "selection_scalar",
    "checkpoint_path",
    "proxy_feedback",
    "training_dynamics",
    "warm_start_checkpoint",
    "eval_generation",
    "fitness_history",
    "duplicate_of",
    "llm_fallback_clone",
)

# Stage II keys kept under a ``stage2_`` prefix for reports (fitness always).
_STAGE2_PRESERVE_AS = (
    "training_dynamics",
    "warm_start_checkpoint",
    "eval_generation",
    "fitness_history",
    "duplicate_of",
    "llm_fallback_clone",
)


@dataclass
class Stage3Config:
    """Stage III budget + domain trainer knobs."""

    train_env_steps: int = STAGE3_STEPS
    eval_episodes: int = 30
    seed: int = 425
    output_root: str = "trained_models/stage3"
    device: str = "cpu"
    # Crash-safe resume (``{run}/stage3/checkpoint.json``).
    resume: bool = True
    action_mode: str = "discrete"
    robot_kinematics: str = "holonomic"
    action_speed_samples: int = 5
    action_rotation_samples: int = 16
    env_config_path: Optional[str] = None
    eval_mode: str = "both"
    # Selection split; ``final_eval`` switches to ``FINAL_EVAL_SPLIT``.
    eval_split: str = "val"
    warm_start: bool = False
    n_envs: int = 1
    # Same PPO rollout / eval policy mode as Stage II so both stages are comparable.
    ppo_n_steps: int = 256
    # PPO per-step discount; None -> domain default (CrowdNav PPO_GAMMA = 0.99).
    ppo_gamma: Optional[float] = None
    eval_deterministic: bool = False
    # None -> ``seed + round_index`` (the closed loop's generation-0 scenarios).
    eval_seed_base: Optional[int] = None
    # Skip Stage-II collision-collapsed candidates (CR >= collapse_cr). Default
    # on: they waste K3 wall-clock and stay collapsed after long training.
    # Timeout-heavy Stage II (high TR, low SR) is NOT filtered — that often
    # reflects under-training rather than a broken reward.
    skip_collapsed: bool = True
    collapse_cr: float = 0.8
    # Independent PPO train seeds per finalist (default 1 = current behaviour).
    # Selection uses mean fitness across seeds; final eval uses the best seed's
    # checkpoint. Pay for K>1 via ``top_k`` (top Stage-II non-collapsed).
    n_seeds: int = 1
    # After the collapse filter, keep only the top-k by Stage II fitness.
    # None / <=0 = keep everyone that survived the collapse filter.
    top_k: Optional[int] = None


@dataclass
class Stage3Record:
    candidate_id: str
    metrics: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "metrics": dict(self.metrics),
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class TrainEvalBundle:
    """Result of one Stage III train+eval."""

    metrics: ProxyMetrics
    checkpoint_path: Optional[str] = None


class PolicyTrainer(ABC):
    """Train + evaluate a full policy under a candidate reward."""

    @abstractmethod
    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
    ) -> TrainEvalBundle:
        raise NotImplementedError


class StubPolicyTrainer(PolicyTrainer):
    """Deterministic no-op trainer for pytest / ``--fast``."""

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        **kwargs: Any,
    ) -> TrainEvalBundle:
        metrics = stub_metrics_from_code(
            candidate.code,
            round_index,
            sr_offset=0.5,
            sr_scale=0.4,
            cr_scale=0.3,
            jitter_mod=3,
            nt_base=12.0,
            nt_scale=4.0,
            pl_base=14.0,
            pl_scale=6.0,
            itr_base=4.0,
            itr_scale=15.0,
            sd_base=0.25,
            sd_scale=0.25,
        )
        return TrainEvalBundle(metrics=metrics, checkpoint_path=None)

    def reevaluate(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
    ) -> ProxyMetrics:
        """Eval-only pass; the stub has no policy, so it re-derives the metrics."""
        return self.train_and_eval(
            candidate, round_index=round_index, config=config
        ).metrics


def _stage2_collision_rate(candidate: RewardCandidate) -> Optional[float]:
    """Stage II holdout CR, or None when metrics are missing."""
    md = candidate.metadata or {}
    metrics = md.get("last_metrics")
    if not isinstance(metrics, dict):
        return None
    holdout = metrics.get("holdout") if isinstance(metrics.get("holdout"), dict) else metrics
    try:
        return float(holdout.get("CR", holdout.get("cr")))
    except (TypeError, ValueError):
        return None


def stage2_is_collapsed(
    candidate: RewardCandidate, *, collapse_cr: float = 0.8
) -> bool:
    """True when Stage II holdout CR is at/above the collapse threshold."""
    cr = _stage2_collision_rate(candidate)
    return cr is not None and cr >= float(collapse_cr)


def _stage3_input(candidate: RewardCandidate) -> RewardCandidate:
    """Detached copy without Stage II results (kept under ``stage2_*`` for reports)."""
    md = dict(candidate.metadata or {})
    preserved = {"stage2_fitness": md.get("fitness")}
    cr = _stage2_collision_rate(candidate)
    if cr is not None:
        preserved["stage2_cr"] = cr
    for key in _STAGE2_PRESERVE_AS:
        if key in md:
            preserved[f"stage2_{key}"] = md[key]
    for key in _STAGE2_RESULT_KEYS:
        md.pop(key, None)
    md.update(preserved)
    return replace(candidate, metadata=md)


def _trained_metrics(candidate: RewardCandidate, bundle: TrainEvalBundle) -> Dict[str, Any]:
    md = candidate.metadata or {}
    if md.get("fitness") is not None and isinstance(md.get("last_metrics"), dict):
        # Domain trainer already stamped the full metrics dict + fitness.
        return dict(md["last_metrics"])
    metrics: Dict[str, Any] = bundle.metrics.as_dict()
    metrics["fitness"] = metrics_fitness(metrics)
    return metrics


class Stage3Runner:
    """Train + evaluate each final candidate once; best by fitness."""

    def __init__(
        self,
        trainer: PolicyTrainer,
        *,
        validator: RewardValidator,
        config: Optional[Stage3Config] = None,
    ) -> None:
        self.trainer = trainer
        self.validator = validator
        self.config = config or Stage3Config()
        self.history: List[Stage3Record] = []
        self.trained: List[RewardCandidate] = []
        self.best_trained: Optional[RewardCandidate] = None

    def _ckpt_dir(self) -> str:
        return stage3_dir_from_train_root(self.config.output_root)

    def _config_snapshot(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "train_env_steps": int(cfg.train_env_steps),
            "eval_episodes": int(cfg.eval_episodes),
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "output_root": str(cfg.output_root),
            "action_mode": str(cfg.action_mode),
            "robot_kinematics": str(cfg.robot_kinematics),
            "action_speed_samples": int(cfg.action_speed_samples),
            "action_rotation_samples": int(cfg.action_rotation_samples),
            "env_config_path": cfg.env_config_path,
            "eval_mode": str(cfg.eval_mode),
            "eval_split": str(cfg.eval_split),
            "warm_start": bool(cfg.warm_start),
            "n_envs": int(cfg.n_envs),
            "ppo_n_steps": int(cfg.ppo_n_steps),
            "eval_deterministic": bool(cfg.eval_deterministic),
            "eval_seed_base": cfg.eval_seed_base,
            "skip_collapsed": bool(cfg.skip_collapsed),
            "collapse_cr": float(cfg.collapse_cr),
            "n_seeds": max(1, int(cfg.n_seeds)),
            "top_k": cfg.top_k,
        }

    def _warn_config_drift(self, saved: Dict[str, Any]) -> None:
        """Resume keeps going, but a changed budget/seed mixes incomparable results."""
        current = self._config_snapshot()
        drift = {
            key: (saved[key], current[key])
            for key in current
            if key in saved and saved[key] != current[key]
        }
        if not drift:
            return
        detail = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in drift.items())
        console.warn(
            f"resuming with settings that differ from the checkpoint ({detail})",
            stage="Stage III",
        )

    def _persist(
        self, *, status: str, population: Sequence[RewardCandidate], next_index: int
    ) -> None:
        if not bool(self.config.resume):
            return
        payload = build_checkpoint(
            status=status,
            next_index=next_index,
            population=population,
            trained=self.trained,
            history=[r.as_dict() for r in self.history],
            best_trained=self.best_trained,
            config=self._config_snapshot(),
        )
        path = save_checkpoint(self._ckpt_dir(), payload)
        logger.info("Stage III checkpoint → %s (%s, next=%d)", path, status, next_index)

    def _record(self, candidate: RewardCandidate, bundle: TrainEvalBundle) -> RewardCandidate:
        metrics = _trained_metrics(candidate, bundle)
        fitness = float(metrics["fitness"])
        trained = replace(
            candidate,
            metadata={
                **(candidate.metadata or {}),
                "last_metrics": metrics,
                "fitness": fitness,
                "selection_scalar": fitness,
                "checkpoint_path": bundle.checkpoint_path,
                "stage3_trained": True,
            },
        )
        self.trained.append(trained)
        self.history.append(
            Stage3Record(
                candidate_id=str(candidate.candidate_id),
                metrics=metrics,
                checkpoint_path=bundle.checkpoint_path,
            )
        )
        prev = (
            candidate_fitness(self.best_trained)
            if self.best_trained is not None
            else float("-inf")
        )
        if fitness > prev:
            self.best_trained = trained
            console.status(
                f"new best {trained.candidate_id} fitness={fitness:.3f}",
                stage="Stage III",
            )
        return trained

    def _record_failure(self, candidate: RewardCandidate, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Stage III train+eval failed for %s", candidate.candidate_id)
        console.warn(
            f"{candidate.candidate_id} failed ({error}); skipping", stage="Stage III"
        )
        self.history.append(
            Stage3Record(
                candidate_id=str(candidate.candidate_id),
                metrics={},
                status="failed",
                error=error,
            )
        )

    def _restore(self, payload: Dict[str, Any]) -> None:
        self.trained = deserialize_population(
            payload.get("trained"), validator=self.validator
        )
        self.history = [
            Stage3Record(
                candidate_id=str(row.get("candidate_id", "")),
                metrics=dict(row.get("metrics") or {}),
                checkpoint_path=row.get("checkpoint_path"),
                status=str(row.get("status") or "ok"),
                error=row.get("error"),
            )
            for row in (payload.get("history") or [])
            if isinstance(row, dict)
        ]
        best = payload.get("best_trained")
        self.best_trained = (
            load_candidate_dict(best, validator=self.validator)
            if isinstance(best, dict)
            else None
        )

    def _apply_collapse_filter(
        self, population: Sequence[RewardCandidate]
    ) -> List[RewardCandidate]:
        """Drop Stage-II collision-collapsed candidates unless that would empty the set."""
        if not bool(self.config.skip_collapsed):
            return [_stage3_input(c) for c in population]
        threshold = float(self.config.collapse_cr)
        kept: List[RewardCandidate] = []
        skipped: List[RewardCandidate] = []
        for cand in population:
            if stage2_is_collapsed(cand, collapse_cr=threshold):
                skipped.append(cand)
            else:
                kept.append(cand)
        if skipped and not kept:
            console.warn(
                f"all {len(skipped)} Stage III candidates are collapsed "
                f"(CR>={threshold:g}); training all of them anyway",
                stage="Stage III",
            )
            return [_stage3_input(c) for c in population]
        for cand in skipped:
            cr = _stage2_collision_rate(cand)
            console.status(
                f"skip {cand.candidate_id} (Stage II CR={cr:.3f} >= {threshold:g})",
                stage="Stage III",
            )
            self.history.append(
                Stage3Record(
                    candidate_id=str(cand.candidate_id),
                    metrics={"CR": cr, "collapse_cr": threshold},
                    status="skipped_collapsed",
                )
            )
        return [_stage3_input(c) for c in kept]

    def _apply_top_k(
        self, population: Sequence[RewardCandidate]
    ) -> List[RewardCandidate]:
        """Keep the top-k by Stage II fitness (after the collapse filter)."""
        k = self.config.top_k
        pop = list(population)
        if k is None or int(k) <= 0 or int(k) >= len(pop):
            return pop
        k = int(k)

        def _s2_fit(c: RewardCandidate) -> float:
            md = c.metadata or {}
            try:
                return float(md.get("stage2_fitness"))
            except (TypeError, ValueError):
                return float("-inf")

        ranked = sorted(pop, key=_s2_fit, reverse=True)
        kept, skipped = ranked[:k], ranked[k:]
        for cand in skipped:
            console.status(
                f"skip {cand.candidate_id} (outside Stage III top_k={k})",
                stage="Stage III",
            )
            self.history.append(
                Stage3Record(
                    candidate_id=str(cand.candidate_id),
                    metrics={"stage2_fitness": _s2_fit(cand), "top_k": k},
                    status="skipped_top_k",
                )
            )
        return kept

    def _prepare_population(
        self, population: Sequence[RewardCandidate]
    ) -> List[RewardCandidate]:
        return self._apply_top_k(self._apply_collapse_filter(population))

    def _train_candidate(self, cand: RewardCandidate) -> None:
        """Train ``cand`` for ``n_seeds`` PPO seeds; select by mean fitness."""
        n_seeds = max(1, int(self.config.n_seeds))
        seed_runs: List[Dict[str, Any]] = []
        best_bundle: Optional[TrainEvalBundle] = None
        best_fit = float("-inf")
        for s in range(n_seeds):
            console.status(
                f"candidate {cand.candidate_id} seed {s + 1}/{n_seeds}",
                stage="Stage III",
            )
            try:
                # ``round_index`` offsets the train seed; ``eval_seed_base`` stays
                # fixed so every seed of a genome sees the same holdout cases.
                bundle = self.trainer.train_and_eval(
                    cand, round_index=s, config=self.config
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Stage III seed %d failed for %s", s, cand.candidate_id
                )
                seed_runs.append(
                    {
                        "seed_index": s,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            metrics = _trained_metrics(cand, bundle)
            fit = float(metrics["fitness"])
            seed_runs.append(
                {
                    "seed_index": s,
                    "status": "ok",
                    "fitness": fit,
                    "fitness_se": metrics.get("fitness_se"),
                    "checkpoint_path": bundle.checkpoint_path,
                    "train_seed": (cand.metadata or {}).get("train_seed"),
                }
            )
            if fit > best_fit:
                best_fit = fit
                best_bundle = bundle

        ok_fits = [float(r["fitness"]) for r in seed_runs if r.get("status") == "ok"]
        if not ok_fits or best_bundle is None:
            first_err = next(
                (str(r.get("error") or "") for r in seed_runs if r.get("status") == "failed"),
                f"all {n_seeds} Stage III seed(s) failed",
            )
            self._record_failure(cand, RuntimeError(first_err or f"all {n_seeds} seed(s) failed"))
            return

        trained = self._record(cand, best_bundle)
        md = dict(trained.metadata or {})
        mean_fit = float(statistics.fmean(ok_fits))
        if n_seeds > 1 and len(ok_fits) > 1:
            # SE of the mean across independent train seeds.
            se = float(statistics.stdev(ok_fits) / math.sqrt(len(ok_fits)))
        else:
            se = float((md.get("last_metrics") or {}).get("fitness_se") or 0.0)
        last = dict(md.get("last_metrics") or {})
        last["fitness"] = mean_fit
        last["fitness_se"] = se
        last["stage3_seed_fits"] = list(ok_fits)
        md["last_metrics"] = last
        md["fitness"] = mean_fit
        md["selection_scalar"] = mean_fit
        md["stage3_n_seeds"] = n_seeds
        md["stage3_seed_runs"] = seed_runs
        # Keep the best-seed checkpoint for blind re-eval / viz.
        md["checkpoint_path"] = best_bundle.checkpoint_path
        trained.metadata = md
        # Refresh history row + best pointer with the aggregated fitness.
        for row in self.history:
            if row.candidate_id == str(cand.candidate_id) and row.status == "ok":
                row.metrics = dict(last)
                row.checkpoint_path = best_bundle.checkpoint_path
                break
        # ``_record`` ranked on the best seed alone; re-select by mean fitness.
        self.best_trained = (
            max(self.trained, key=candidate_fitness) if self.trained else None
        )
        if n_seeds > 1:
            console.status(
                f"{cand.candidate_id} mean_fitness={mean_fit:.3f} "
                f"se={se:.3f} (n_seeds={len(ok_fits)})",
                stage="Stage III",
            )

    def run(self, population: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        n_seeds = max(1, int(self.config.n_seeds))
        console.banner(
            f"Stage III - long PPO (K3={self.config.train_env_steps}, "
            f"E3={self.config.eval_episodes}, seeds={n_seeds})"
        )
        start = 0
        payload = load_checkpoint(self._ckpt_dir()) if self.config.resume else None
        if payload:
            self._warn_config_drift(payload.get("config") or {})
            self._restore(payload)
            saved_pop = deserialize_population(
                payload.get("population"), validator=self.validator
            )
            pop = saved_pop if saved_pop else self._prepare_population(population)
            start = int(payload.get("next_index") or 0)
            if str(payload.get("status")) == "done":
                console.status(
                    "Stage III checkpoint already done — reusing results",
                    stage="Stage III",
                )
                return list(self.trained)
            console.status(
                f"resuming Stage III at candidate {start + 1}/{len(pop)}",
                stage="Stage III",
            )
        else:
            pop = self._prepare_population(population)

        n = len(pop)
        for i in range(start, n):
            cand = pop[i]
            console.status(
                f"candidate {cand.candidate_id} ({i + 1}/{n})", stage="Stage III"
            )
            self._train_candidate(cand)
            self._persist(status="running", population=pop, next_index=i + 1)

        self._persist(status="done", population=pop, next_index=n)
        return list(self.trained)

    def selection_summary(self) -> Dict[str, Any]:
        """Winner plus the candidates statistically tied with it (manifest row)."""
        n_failed = sum(1 for r in self.history if r.status == "failed")
        best = self.best_trained
        if best is None:
            return {
                "status": "all_failed" if self.history else "empty",
                "n_trained": len(self.trained),
                "n_failed": n_failed,
            }
        ties = statistical_ties(self.trained, best)
        if ties:
            console.warn(
                f"best {best.candidate_id} is within 2 SE of {', '.join(ties)}; "
                "the Stage III ranking among them is noise",
                stage="Stage III",
            )
        return {
            "status": "ok",
            "best_id": str(best.candidate_id),
            "fitness": candidate_fitness(best),
            "fitness_se": candidate_fitness_se(best),
            "tied_ids": ties,
            "n_trained": len(self.trained),
            "n_failed": n_failed,
            "n_seeds": max(1, int(self.config.n_seeds)),
            "top_k": self.config.top_k,
            "eval_episodes": int(self.config.eval_episodes),
            "eval_split": str(self.config.eval_split),
            "eval_seed_base": self.config.eval_seed_base,
        }

    def final_eval(
        self, candidate: RewardCandidate, *, episodes: int, seed_base: int
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluate the selected policy alone on unseen scenarios.

        Selection already used the Stage III scores, so only an evaluation on
        fresh seeds gives an unbiased estimate of the winner's fitness.
        Returns ``None`` when disabled or the trainer cannot re-evaluate.
        """
        reevaluate = getattr(self.trainer, "reevaluate", None)
        if int(episodes) <= 0 or reevaluate is None:
            return None
        md = dict(candidate.metadata or {})
        for key in _STAGE2_RESULT_KEYS:
            if key != "checkpoint_path":
                md.pop(key, None)
        probe = replace(candidate, metadata=md)
        cfg = replace(
            self.config,
            eval_episodes=int(episodes),
            eval_seed_base=int(seed_base),
            eval_split=FINAL_EVAL_SPLIT,
        )
        console.banner(
            f"Final blind eval - {candidate.candidate_id} ({int(episodes)} episodes)"
        )
        row: Dict[str, Any] = {
            "candidate_id": str(candidate.candidate_id),
            "episodes": int(episodes),
            "seed_base": int(seed_base),
            "split": FINAL_EVAL_SPLIT,
        }
        try:
            result = reevaluate(probe, round_index=0, config=cfg)
        except Exception as exc:  # noqa: BLE001 - the trained winner is still valid
            logger.exception("final eval failed for %s", candidate.candidate_id)
            console.warn(f"final eval failed ({exc})", stage="Stage III")
            return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        bundle = result if isinstance(result, TrainEvalBundle) else TrainEvalBundle(result)
        metrics = _trained_metrics(probe, bundle)
        row.update(
            status="ok",
            fitness=float(metrics["fitness"]),
            fitness_se=metrics.get("fitness_se"),
            metrics=metrics,
        )
        console.status(
            f"final eval {candidate.candidate_id} fitness={row['fitness']:.3f} "
            f"(Stage III selection score {candidate_fitness(candidate):.3f})",
            stage="Stage III",
        )
        return row


__all__ = [
    "FINAL_EVAL_SEED_OFFSET",
    "FINAL_EVAL_SPLIT",
    "STAGE3_EVAL_SEED_OFFSET",
    "STAGE3_STEPS",
    "PolicyTrainer",
    "Stage3Config",
    "Stage3Record",
    "Stage3Runner",
    "StubPolicyTrainer",
    "TrainEvalBundle",
]
