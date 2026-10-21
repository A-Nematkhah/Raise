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

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

from raise_core import console
from raise_core.candidate import RewardCandidate
from raise_core.candidate import genome_fingerprint
from raise_core.config_defaults import (
    budget_stage3_eval_episodes,
    budget_stage3_train_steps,
    env_action_mode,
    env_kinematics,
    env_rotation_samples,
    env_speed_samples,
    ppo_n_steps,
    run_seed,
    upstream_env_config,
)
from raise_core.population_io import deserialize_population, load_candidate_dict
from raise_core.refine import ProxyMetrics, stub_metrics_from_code
from raise_core.sandbox import RewardValidator
from raise_core.seed_aggregate import (
    aggregate_seed_fits,
    median_seed_checkpoint,
    partial_seeds_flag,
)
from raise_core.selection import (
    candidate_fitness,
    candidate_fitness_se,
    candidate_fitness_se_source,
    metrics_fitness,
    pick_best_trained,
    statistical_ties,
)
from raise_core.seeds import FINAL_EVAL_SEED_OFFSET, STAGE3_EVAL_SEED_OFFSET
from raise_core.stage3_checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    stage3_dir_from_train_root,
)

logger = logging.getLogger(__name__)

# K3 default (environment steps per candidate) lives in ``raise_config.CONFIG``.
STAGE3_STEPS = budget_stage3_train_steps()
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

    train_env_steps: int = field(default_factory=budget_stage3_train_steps)
    eval_episodes: int = field(default_factory=budget_stage3_eval_episodes)
    seed: int = field(default_factory=run_seed)
    output_root: str = "trained_models/stage3"
    device: str = "cpu"
    # Crash-safe resume (``{run}/stage3/checkpoint.json``).
    resume: bool = True
    action_mode: str = field(default_factory=env_action_mode)
    robot_kinematics: str = field(default_factory=env_kinematics)
    action_speed_samples: int = field(default_factory=env_speed_samples)
    action_rotation_samples: int = field(default_factory=env_rotation_samples)
    env_config_path: Optional[str] = None
    env_config_obj: Any = field(default_factory=upstream_env_config, repr=False)
    eval_mode: str = "both"
    # Selection split; ``final_eval`` switches to ``FINAL_EVAL_SPLIT``.
    eval_split: str = "val"
    warm_start: bool = False
    n_envs: int = 1
    # Same PPO rollout / eval policy mode as Stage II so both stages are comparable.
    ppo_n_steps: int = field(default_factory=ppo_n_steps)
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
    # Selection uses mean fitness across seeds (SE across seeds when >1 seed
    # succeeds). The blind final eval re-evaluates ALL successful seed
    # checkpoints on the same final cases and reports their mean fitness;
    # ``checkpoint_path`` on the candidate is the median-fitness seed (used for
    # viz only, never the max). Pay for K>1 via ``top_k`` (top Stage-II
    # non-collapsed).
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
        train_seed_index: Optional[int] = None,
    ) -> TrainEvalBundle:
        """Train + eval one policy.

        ``train_seed_index`` selects the PPO train seed (``config.seed +
        train_seed_index``). ``None`` keeps the legacy behaviour where the train
        seed is derived from ``round_index``.
        """
        raise NotImplementedError


class StubPolicyTrainer(PolicyTrainer):
    """Deterministic no-op trainer for pytest / ``--fast``."""

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        train_seed_index: Optional[int] = None,
        **kwargs: Any,
    ) -> TrainEvalBundle:
        # The stub derives its jitter from ``round_index`` only; ``train_seed_index``
        # is accepted for API parity with real trainers.
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
        train_seed_index: Optional[int] = None,
        **kwargs: Any,
    ) -> ProxyMetrics:
        """Eval-only pass; the stub has no policy, so it re-derives the metrics."""
        return self.train_and_eval(
            candidate,
            round_index=round_index,
            config=config,
            train_seed_index=train_seed_index,
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


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True when callable ``fn`` takes keyword ``name`` (explicitly or via ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_train_and_eval(
    trainer: Any,
    candidate: RewardCandidate,
    *,
    round_index: int,
    config: Stage3Config,
    train_seed_index: Optional[int],
) -> TrainEvalBundle:
    """Call ``trainer.train_and_eval``, passing ``train_seed_index`` when supported.

    Third-party trainers written against the old signature (no ``**kwargs``)
    keep working: they fall back to the legacy ``round_index``-derived seed.
    """
    if _accepts_kwarg(trainer.train_and_eval, "train_seed_index"):
        return trainer.train_and_eval(
            candidate,
            round_index=round_index,
            config=config,
            train_seed_index=train_seed_index,
        )
    return trainer.train_and_eval(candidate, round_index=round_index, config=config)


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

    def _record(
        self,
        candidate: RewardCandidate,
        bundle: TrainEvalBundle,
        *,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> RewardCandidate:
        if metrics is None:
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
        # ``best_trained`` is re-ranked by the caller after seed aggregation
        # (``pick_best_trained``), not here on a single seed's fitness.
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
        filtered = self._apply_top_k(self._apply_collapse_filter(population))
        unique: List[RewardCandidate] = []
        first_by_genome: Dict[str, RewardCandidate] = {}
        for candidate in filtered:
            fingerprint = genome_fingerprint(candidate.code)
            original = first_by_genome.get(fingerprint)
            if original is not None:
                console.status(
                    f"skip {candidate.candidate_id} (duplicate genome of "
                    f"{original.candidate_id})",
                    stage="Stage III",
                )
                self.history.append(
                    Stage3Record(
                        candidate_id=str(candidate.candidate_id),
                        metrics={"duplicate_of": str(original.candidate_id)},
                        status="skipped_duplicate",
                    )
                )
                continue
            first_by_genome[fingerprint] = candidate
            unique.append(candidate)
        return unique

    def _train_candidate(self, cand: RewardCandidate) -> None:
        """Train ``cand`` for ``n_seeds`` PPO seeds; select by mean fitness.

        Fitness / SE come from ``aggregate_seed_fits`` over all successful
        seeds. ``checkpoint_path`` is the *median* seed's checkpoint (never the
        max, which would bias the reported policy upward); every successful
        seed checkpoint is kept in ``stage3_seed_checkpoints`` for the final eval.
        """
        n_seeds = max(1, int(self.config.n_seeds))
        seed_runs: List[Dict[str, Any]] = []
        seed_bundles: Dict[int, TrainEvalBundle] = {}
        seed_metrics: Dict[int, Dict[str, Any]] = {}
        for s in range(n_seeds):
            console.status(
                f"candidate {cand.candidate_id} seed {s + 1}/{n_seeds}",
                stage="Stage III",
            )
            try:
                # ``round_index=s`` keeps bookkeeping / output dirs ordered;
                # ``train_seed_index=s`` selects the PPO train seed explicitly.
                # ``eval_seed_base`` stays fixed so every seed of a genome sees
                # the same holdout cases.
                bundle = _call_train_and_eval(
                    self.trainer,
                    cand,
                    round_index=s,
                    config=self.config,
                    train_seed_index=s,
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
            seed_bundles[s] = bundle
            seed_metrics[s] = dict(metrics)

        ok_runs = [r for r in seed_runs if r.get("status") == "ok"]
        ok_fits = [float(r["fitness"]) for r in ok_runs]
        n_ok = len(ok_runs)
        n_failed = n_seeds - n_ok
        if not ok_fits:
            first_err = next(
                (str(r.get("error") or "") for r in seed_runs if r.get("status") == "failed"),
                f"all {n_seeds} Stage III seed(s) failed",
            )
            self._record_failure(cand, RuntimeError(first_err or f"all {n_seeds} seed(s) failed"))
            return

        # Median seed (not max) supplies the reported checkpoint and metrics.
        median_path, median_idx = median_seed_checkpoint(seed_runs)
        if median_idx is None or median_idx not in seed_bundles:
            median_idx = int(ok_runs[0]["seed_index"])
            median_path = seed_bundles[median_idx].checkpoint_path
        median_bundle = seed_bundles[median_idx]
        median_metrics = seed_metrics[median_idx]

        episode_se = float(median_metrics.get("fitness_se") or 0.0)
        agg = aggregate_seed_fits(ok_fits, episode_se=episode_se)
        mean_fit = float(agg["fitness"])
        se = float(agg["fitness_se"])
        se_source = str(agg["fitness_se_source"])
        partial = partial_seeds_flag(n_ok=n_ok, n_seeds=n_seeds)
        if partial:
            console.warn(
                f"{cand.candidate_id}: only {n_ok}/{n_seeds} seed(s) succeeded; "
                "fitness is a partial-seed mean and ranks below full-seed candidates",
                stage="Stage III",
            )

        trained = self._record(cand, median_bundle, metrics=median_metrics)
        md = dict(trained.metadata or {})
        last = dict(md.get("last_metrics") or {})
        last["fitness"] = mean_fit
        last["fitness_se"] = se
        last["fitness_se_source"] = se_source
        last["stage3_seed_fits"] = list(ok_fits)
        md["last_metrics"] = last
        md["fitness"] = mean_fit
        md["selection_scalar"] = mean_fit
        md["fitness_se_source"] = se_source
        md["stage3_n_seeds"] = n_seeds
        md["stage3_seed_runs"] = seed_runs
        md["stage3_seed_checkpoints"] = [
            str(r["checkpoint_path"])
            for r in ok_runs
            if r.get("checkpoint_path") is not None
        ]
        md["n_ok_seeds"] = n_ok
        md["n_failed_seeds"] = n_failed
        md["partial_seeds"] = bool(partial)
        # Median-fitness seed's checkpoint (viz / single-policy consumers).
        md["checkpoint_path"] = median_path
        md["checkpoint_selection"] = "median_seed"
        md["median_seed_index"] = int(median_idx)
        trained.metadata = md
        # Refresh history row with the aggregated fitness.
        for row in self.history:
            if row.candidate_id == str(cand.candidate_id) and row.status == "ok":
                row.metrics = dict(last)
                row.checkpoint_path = median_path
                break
        # Partial-seed candidates rank below full-seed ones (``pick_best_trained``).
        prev_best = self.best_trained
        self.best_trained = pick_best_trained(self.trained)
        if (
            self.best_trained is not None
            and self.best_trained is not prev_best
            and str(self.best_trained.candidate_id) == str(cand.candidate_id)
        ):
            console.status(
                f"new best {cand.candidate_id} fitness={mean_fit:.3f}",
                stage="Stage III",
            )
        if n_seeds > 1:
            console.status(
                f"{cand.candidate_id} mean_fitness={mean_fit:.3f} "
                f"se={se:.3f} ({se_source}, n_ok_seeds={n_ok}/{n_seeds})",
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
            "fitness_se_source": candidate_fitness_se_source(best),
            "n_ok_seeds": (best.metadata or {}).get("n_ok_seeds"),
            "partial_seeds": bool((best.metadata or {}).get("partial_seeds", False)),
            "top_k": self.config.top_k,
            "eval_episodes": int(self.config.eval_episodes),
            "eval_split": str(self.config.eval_split),
            "eval_seed_base": self.config.eval_seed_base,
        }

    def final_eval(
        self, candidate: RewardCandidate, *, episodes: int, seed_base: int
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluate the selected policy alone on unseen scenarios.

        When the candidate carries multiple successful seed checkpoints
        (``stage3_seed_checkpoints`` / ``stage3_seed_runs``), every seed is
        evaluated on the same final cases and the reported fitness is the mean
        across seeds (with cross-seed SE). ``checkpoint_path`` stays the median
        seed for visualization. Returns ``None`` when disabled or the trainer
        cannot re-evaluate.
        """
        reevaluate = getattr(self.trainer, "reevaluate", None)
        if int(episodes) <= 0 or reevaluate is None:
            return None
        src_md = dict(candidate.metadata or {})
        md = dict(src_md)
        for key in _STAGE2_RESULT_KEYS:
            if key != "checkpoint_path":
                md.pop(key, None)
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
        seed_ckpts = _final_eval_seed_checkpoints(src_md)
        if len(seed_ckpts) > 1:
            return self._final_eval_multi_seed(
                candidate,
                md,
                seed_ckpts,
                reevaluate=reevaluate,
                cfg=cfg,
                row=row,
            )
        probe = replace(candidate, metadata=md)
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
            fitness_se_source="episodes_only",
            metrics=metrics,
            n_ok_seeds=1,
            n_failed_seeds=0,
            partial_seeds=bool(src_md.get("partial_seeds", False)),
        )
        console.status(
            f"final eval {candidate.candidate_id} fitness={row['fitness']:.3f} "
            f"(Stage III selection score {candidate_fitness(candidate):.3f})",
            stage="Stage III",
        )
        return row

    def _final_eval_multi_seed(
        self,
        candidate: RewardCandidate,
        base_md: Dict[str, Any],
        seed_ckpts: List[Dict[str, Any]],
        *,
        reevaluate: Any,
        cfg: Stage3Config,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate every successful seed checkpoint on the same final cases."""
        seed_evals: List[Dict[str, Any]] = []
        ok_fits: List[float] = []
        for entry in seed_ckpts:
            path = entry.get("checkpoint_path")
            seed_index = int(entry.get("seed_index") or 0)
            probe_md = {**base_md, "checkpoint_path": path}
            probe = replace(candidate, metadata=probe_md)
            try:
                result = reevaluate(probe, round_index=0, config=cfg)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "final eval seed %s failed for %s", seed_index, candidate.candidate_id
                )
                seed_evals.append(
                    {
                        "seed_index": seed_index,
                        "checkpoint_path": path,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            bundle = (
                result if isinstance(result, TrainEvalBundle) else TrainEvalBundle(result)
            )
            metrics = _trained_metrics(probe, bundle)
            fit = float(metrics["fitness"])
            ok_fits.append(fit)
            seed_evals.append(
                {
                    "seed_index": seed_index,
                    "checkpoint_path": path,
                    "status": "ok",
                    "fitness": fit,
                    "fitness_se": metrics.get("fitness_se"),
                    "metrics": metrics,
                }
            )
        n_requested = max(
            len(seed_ckpts),
            int((candidate.metadata or {}).get("stage3_n_seeds") or len(seed_ckpts)),
        )
        n_ok = len(ok_fits)
        n_failed = len(seed_ckpts) - n_ok
        if not ok_fits:
            return {
                **row,
                "status": "failed",
                "error": "all seed checkpoints failed final eval",
                "seed_evals": seed_evals,
                "n_ok_seeds": 0,
                "n_failed_seeds": n_failed,
                "partial_seeds": True,
            }
        agg = aggregate_seed_fits(
            ok_fits,
            episode_se=float(seed_evals[0].get("fitness_se") or 0.0),
        )
        partial = partial_seeds_flag(n_ok=n_ok, n_seeds=n_requested) or bool(
            (candidate.metadata or {}).get("partial_seeds")
        )
        if partial:
            console.warn(
                f"final eval {candidate.candidate_id}: partial seeds "
                f"({n_ok}/{n_requested} ok)",
                stage="Stage III",
            )
        median_path = (candidate.metadata or {}).get("checkpoint_path")
        row.update(
            status="ok",
            fitness=float(agg["fitness"]),
            fitness_se=float(agg["fitness_se"]),
            fitness_se_source=str(agg["fitness_se_source"]),
            seed_evals=seed_evals,
            n_ok_seeds=n_ok,
            n_failed_seeds=n_failed,
            partial_seeds=bool(partial),
            checkpoint_path=median_path,
            checkpoint_selection="median_seed",
            metrics={
                "fitness": float(agg["fitness"]),
                "fitness_se": float(agg["fitness_se"]),
                "fitness_se_source": str(agg["fitness_se_source"]),
                "stage3_seed_fits": list(ok_fits),
            },
        )
        console.status(
            f"final eval {candidate.candidate_id} mean_fitness={row['fitness']:.3f} "
            f"se={row['fitness_se']:.3f} ({row['fitness_se_source']}, "
            f"n_ok_seeds={n_ok}/{n_requested})",
            stage="Stage III",
        )
        return row


def _final_eval_seed_checkpoints(md: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Successful Stage III seed checkpoints for multi-seed final eval."""
    runs = md.get("stage3_seed_runs")
    out: List[Dict[str, Any]] = []
    if isinstance(runs, list):
        for row in runs:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            path = row.get("checkpoint_path")
            if not path:
                continue
            out.append(
                {
                    "seed_index": int(row.get("seed_index") or 0),
                    "checkpoint_path": str(path),
                }
            )
        if out:
            return out
    paths = md.get("stage3_seed_checkpoints")
    if isinstance(paths, list):
        for i, path in enumerate(paths):
            if path:
                out.append({"seed_index": i, "checkpoint_path": str(path)})
    return out


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
