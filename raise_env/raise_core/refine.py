"""
RAISE Stage II — short PPO train + evaluation of one reward candidate.

The closed-loop runner calls ``trainer.train_and_eval(candidate, round_index=,
config=Stage2Config)`` for every new candidate each generation. The highway
trainer (``domains.highway.adapter``) caches ``last_metrics`` / ``fitness`` /
``checkpoint_path`` on the candidate metadata.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from raise_core.candidate import RewardCandidate


@dataclass
class Stage2Config:
    """Stage II budget + highway trainer knobs."""

    # PPO budget per candidate (environment steps).
    train_env_steps: int = 12_000
    eval_episodes: int = 30
    seed: int = 425
    output_root: str = "trained_models/stage2"
    device: str = "cpu"
    highway_n_envs: int = 1
    highway_warm_start: bool = False
    highway_eval_mode: str = "both"
    highway_action_mode: str = "meta_default"
    highway_action_continuous_lateral: bool = True
    highway_meta_fine_low: float = 15.0
    highway_meta_fine_high: float = 35.0
    highway_meta_fine_n: int = 21
    # PPO entropy bonus (SB3 default 0.0) and holdout eval policy mode.
    # Stochastic eval: deterministic argmax labels flip with the training seed.
    highway_ent_coef: float = 0.0
    highway_eval_deterministic: bool = False
    # PPO rollout length cap per env and optimisation epochs per rollout.
    highway_n_steps: int = 64
    highway_n_epochs: int = 10
    # Eval scenario seed shared by every candidate trained in one generation
    # (None → per-candidate ``seed + round_index``).
    highway_eval_seed_base: Optional[int] = None
    # Feedback-card instrumentation: training-rollout outcome windows and
    # per-episode reward-component sums on holdout (no reward / PPO change).
    highway_training_dynamics: bool = False
    highway_reward_components: bool = False


@dataclass
class ProxyMetrics:
    """Evaluation metrics M(r) (highway: PL=progress, ITR=speed, SD=gap)."""

    sr: float = 0.0
    cr: float = 0.0
    tr: float = 0.0
    nt: float = 0.0
    pl: float = 0.0
    itr: float = 0.0
    sd: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "SR": float(self.sr),
            "CR": float(self.cr),
            "TR": float(self.tr),
            "NT": float(self.nt),
            "PL": float(self.pl),
            "ITR": float(self.itr),
            "SD": float(self.sd),
        }

    def feedback_text(self) -> str:
        d = self.as_dict()
        return (
            f"SR={d['SR']:.4f}, CR={d['CR']:.4f}, TR={d['TR']:.4f}, "
            f"NT={d['NT']:.4f}, PL={d['PL']:.4f}, ITR={d['ITR']:.4f}, SD={d['SD']:.4f}"
        )


def _stable_code_hash(code: str) -> int:
    return int(hashlib.md5(code.encode("utf-8")).hexdigest(), 16) % 10_000


class PolicyTrainer(ABC):
    """Train + evaluate a policy under a candidate reward."""

    @abstractmethod
    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        raise NotImplementedError


class StubPolicyTrainer(PolicyTrainer):
    """
    Deterministic no-op trainer for pytest / ``--fast`` (seconds, not GPU-hours).

    Metrics depend only on a stable hash of ``candidate.code`` so ranking
    is reproducible across rounds unless the code changes.
    """

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        base = (_stable_code_hash(candidate.code) % 100) / 100.0
        jitter = 0.01 * (round_index % 5)
        sr = min(1.0, max(0.0, 0.4 + 0.5 * base + jitter))
        cr = min(1.0, max(0.0, 0.4 * (1.0 - base)))
        tr = max(0.0, 1.0 - sr - cr)
        return ProxyMetrics(
            sr=sr,
            cr=cr,
            tr=tr,
            nt=10.0 + 5.0 * (1.0 - base),
            pl=12.0 + 8.0 * (1.0 - base),
            itr=5.0 + 20.0 * (1.0 - base),
            sd=0.2 + 0.3 * base,
        )
