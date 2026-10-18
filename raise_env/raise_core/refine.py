"""
RAISE Stage II — short PPO train + evaluation of one reward candidate.

The closed-loop runner calls ``trainer.train_and_eval(candidate, round_index=,
config=Stage2Config)`` for every new candidate each generation, and
``trainer.reevaluate(...)`` (eval only) for genomes carried in from an earlier
generation, so every fitness it ranks comes from the same scenarios. The domain
trainer (``domains.crowdnav.adapter``) caches ``last_metrics`` / ``fitness`` /
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
    """Stage II budget + domain trainer knobs."""

    # PPO budget per candidate (environment steps).
    train_env_steps: int = 12_000
    eval_episodes: int = 30
    seed: int = 425
    output_root: str = "trained_models/stage2"
    device: str = "cpu"
    n_envs: int = 1
    warm_start: bool = False
    eval_mode: str = "both"
    # Upstream scenario split for selection / held-out evaluation.
    eval_split: str = "val"
    # Robot / action space (domains.crowdnav.settings).
    action_mode: str = "continuous"
    robot_kinematics: str = "unicycle"
    action_speed_samples: int = 5
    action_rotation_samples: int = 16
    env_config_path: Optional[str] = None
    # PPO entropy bonus (SB3 default 0.0) and holdout eval policy mode.
    # Stochastic eval: deterministic argmax labels flip with the training seed.
    ppo_ent_coef: float = 0.0
    eval_deterministic: bool = False
    # PPO rollout length cap per env and optimisation epochs per rollout.
    ppo_n_steps: int = 256
    ppo_n_epochs: int = 10
    # PPO per-step discount; None -> domain default (CrowdNav PPO_GAMMA = 0.99).
    ppo_gamma: Optional[float] = None
    # Eval scenario seed shared by every candidate trained in one generation
    # (None → per-candidate ``seed + round_index``).
    eval_seed_base: Optional[int] = None
    # Feedback-card instrumentation: training-rollout outcome windows and
    # per-episode reward-component sums on holdout (no reward / PPO change).
    record_training_dynamics: bool = False
    record_reward_components: bool = False


@dataclass
class ProxyMetrics:
    """Evaluation metrics M(r) (CrowdNav: NT=nav time, PL=path length, ITR=danger freq, SD=min dist in danger)."""

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


def stub_metrics_from_code(
    code: str,
    round_index: int,
    *,
    sr_offset: float,
    sr_scale: float,
    cr_scale: float,
    jitter_mod: int,
    nt_base: float,
    nt_scale: float,
    pl_base: float,
    pl_scale: float,
    itr_base: float,
    itr_scale: float,
    sd_base: float,
    sd_scale: float,
) -> ProxyMetrics:
    """Shared hash→metrics mapping for Stage II / III stub trainers.

    Parameterised so each stage keeps its historical numeric outputs.
    """
    base = (_stable_code_hash(code) % 100) / 100.0
    jitter = 0.01 * (int(round_index) % int(jitter_mod))
    sr = min(1.0, max(0.0, float(sr_offset) + float(sr_scale) * base + jitter))
    cr = min(1.0, max(0.0, float(cr_scale) * (1.0 - base)))
    tr = max(0.0, 1.0 - sr - cr)
    return ProxyMetrics(
        sr=sr,
        cr=cr,
        tr=tr,
        nt=float(nt_base) + float(nt_scale) * (1.0 - base),
        pl=float(pl_base) + float(pl_scale) * (1.0 - base),
        itr=float(itr_base) + float(itr_scale) * (1.0 - base),
        sd=float(sd_base) + float(sd_scale) * base,
    )


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
        return self._metrics(candidate, round_index)

    def reevaluate(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        """Eval-only pass; the stub has no policy, so it re-derives the metrics."""
        return self._metrics(candidate, round_index)

    @staticmethod
    def _metrics(candidate: RewardCandidate, round_index: int) -> ProxyMetrics:
        return stub_metrics_from_code(
            candidate.code,
            round_index,
            sr_offset=0.4,
            sr_scale=0.5,
            cr_scale=0.4,
            jitter_mod=5,
            nt_base=10.0,
            nt_scale=5.0,
            pl_base=12.0,
            pl_scale=8.0,
            itr_base=5.0,
            itr_scale=20.0,
            sd_base=0.2,
            sd_scale=0.3,
        )
