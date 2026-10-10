"""
RAISE Stage III — long PPO training + evaluation of the final candidates.

Every candidate of the final closed-loop population is trained once for K3
environment steps and evaluated on E3 episodes. The best is the trained
candidate with the highest fitness. Crash-safe: progress is checkpointed
after each candidate under ``{run}/stage3/checkpoint.json``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

from raise_core import console
from raise_core.candidate import RewardCandidate
from raise_core.population_io import deserialize_population, load_candidate_dict
from raise_core.refine import ProxyMetrics, _stable_code_hash
from raise_core.sandbox import RewardValidator
from raise_core.selection import candidate_nav_scalar, navigation_scalar_from_dict
from raise_core.stage3_checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    stage3_dir_from_train_root,
)

logger = logging.getLogger(__name__)

# K3 default (environment steps per candidate).
STAGE3_STEPS: int = 70_000

# Stage II results that must not leak into Stage III selection.
_STAGE2_RESULT_KEYS = (
    "last_metrics",
    "fitness",
    "selection_scalar",
    "checkpoint_path",
    "proxy_feedback",
)


@dataclass
class Stage3Config:
    """Stage III budget + highway trainer knobs."""

    train_env_steps: int = STAGE3_STEPS
    eval_episodes: int = 30
    seed: int = 425
    output_root: str = "trained_models/stage3"
    device: str = "cpu"
    # Crash-safe resume (``{run}/stage3/checkpoint.json``).
    resume: bool = True
    highway_action_mode: str = "meta_default"
    highway_action_continuous_lateral: bool = True
    highway_meta_fine_low: float = 15.0
    highway_meta_fine_high: float = 35.0
    highway_meta_fine_n: int = 21
    highway_eval_mode: str = "both"
    highway_warm_start: bool = False
    highway_n_envs: int = 1
    # Same PPO rollout / eval policy mode as Stage II so both stages are comparable.
    highway_n_steps: int = 64
    highway_eval_deterministic: bool = False


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
        base = (_stable_code_hash(candidate.code) % 100) / 100.0
        jitter = 0.01 * (round_index % 3)
        sr = min(1.0, max(0.0, 0.5 + 0.4 * base + jitter))
        cr = min(1.0, max(0.0, 0.3 * (1.0 - base)))
        tr = max(0.0, 1.0 - sr - cr)
        metrics = ProxyMetrics(
            sr=sr,
            cr=cr,
            tr=tr,
            nt=12.0 + 4.0 * (1.0 - base),
            pl=14.0 + 6.0 * (1.0 - base),
            itr=4.0 + 15.0 * (1.0 - base),
            sd=0.25 + 0.25 * base,
        )
        return TrainEvalBundle(metrics=metrics, checkpoint_path=None)


def _stage3_input(candidate: RewardCandidate) -> RewardCandidate:
    """Detached copy without Stage II results (Stage II fitness kept for reports)."""
    md = dict(candidate.metadata or {})
    stage2_fitness = md.get("fitness")
    for key in _STAGE2_RESULT_KEYS:
        md.pop(key, None)
    md["stage2_fitness"] = stage2_fitness
    return replace(candidate, metadata=md)


def _trained_metrics(candidate: RewardCandidate, bundle: TrainEvalBundle) -> Dict[str, Any]:
    md = candidate.metadata or {}
    if md.get("fitness") is not None and isinstance(md.get("last_metrics"), dict):
        # Highway trainer already stamped the full metrics dict + fitness.
        return dict(md["last_metrics"])
    metrics: Dict[str, Any] = bundle.metrics.as_dict()
    metrics["fitness"] = navigation_scalar_from_dict(metrics)
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
            candidate_nav_scalar(self.best_trained)
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

    def run(self, population: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        console.banner(
            f"Stage III - long PPO (K3={self.config.train_env_steps}, "
            f"E3={self.config.eval_episodes})"
        )
        pop = [_stage3_input(c) for c in population]
        start = 0
        payload = load_checkpoint(self._ckpt_dir()) if self.config.resume else None
        if payload:
            self._warn_config_drift(payload.get("config") or {})
            saved_pop = deserialize_population(
                payload.get("population"), validator=self.validator
            )
            if saved_pop:
                pop = saved_pop
            self._restore(payload)
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

        n = len(pop)
        for i in range(start, n):
            cand = pop[i]
            console.status(
                f"candidate {cand.candidate_id} ({i + 1}/{n})", stage="Stage III"
            )
            try:
                bundle = self.trainer.train_and_eval(
                    cand, round_index=0, config=self.config
                )
            except Exception as exc:  # noqa: BLE001 - one bad reward must not stop the run
                self._record_failure(cand, exc)
            else:
                self._record(cand, bundle)
            self._persist(status="running", population=pop, next_index=i + 1)

        self._persist(status="done", population=pop, next_index=n)
        return list(self.trained)


__all__ = [
    "STAGE3_STEPS",
    "PolicyTrainer",
    "Stage3Config",
    "Stage3Record",
    "Stage3Runner",
    "StubPolicyTrainer",
    "TrainEvalBundle",
]
