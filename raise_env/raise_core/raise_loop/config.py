"""Closed-loop reward evolution config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from raise_core.feedback import FeedbackCardConfig


@dataclass
class ClosedLoopConfig:
    """Knobs for ``ClosedLoopRunner``."""

    population_size: int = 6
    generations: int = 4
    # Stage II PPO budget per candidate (environment steps).
    stage2_train_steps: int = 12_000
    stage1_dataset_path: str = "domains/crowdnav/data/stage1_dataset"
    seed: int = 425
    device: str = "cpu"
    use_stub: bool = False
    llm_provider: str = "seed"
    output_dir: str = "results/closed_loop"
    keep_runtime_elite: bool = True
    # Next-generation children per operator; None -> ``operator_counts`` default
    # (N=6 with elitism: 1 elite + 2 crossover + 2 mutation + 1 random).
    n_crossover: Optional[int] = None
    n_mutation: Optional[int] = None
    n_random: Optional[int] = None
    # Mutation parents: lower (bottom half) | rank_weighted | tournament.
    mutation_parents: str = "lower"
    # Drop Stage-II CR>=collapse_cr genomes from the mutation parent pool.
    mutation_exclude_collapsed: bool = False
    collapse_cr: float = 0.8
    # Crash-safe resume from ``checkpoint.json`` written under the run dir.
    resume: bool = True
    domain: str = "crowdnav"
    # Stage II eval episodes (None → 20; 8 with stub trainers).
    eval_episodes: Optional[int] = None
    # Speed knobs (same budget; higher wall throughput).
    n_envs: int = 1
    label_workers: int = 1
    warm_start: bool = False
    # Successive-halving Stage II: train all at K2/3, continue top half from
    # their own checkpoints for the remaining 2K2/3 (total still K2 steps).
    stage2_successive_halving: bool = False
    eval_mode: str = "holdout_only"  # both | holdout_only
    # Stage II holdout pool: val (default) | train (fresh train-split cases).
    # Stage III / final eval stay on val / test regardless of this setting.
    stage2_eval_split: str = "val"
    # Robot / action space (domains.crowdnav.settings).
    action_mode: str = "continuous"
    robot_kinematics: str = "holonomic"
    action_speed_samples: int = 5
    action_rotation_samples: int = 16
    env_config_path: Optional[str] = None
    # Diagnostic feedback card + LLM input-token budget.
    feedback: FeedbackCardConfig = field(default_factory=FeedbackCardConfig)

    def apply_fast_profile(self) -> None:
        self.use_stub = True
        self.population_size = min(self.population_size, 4)
        self.generations = min(self.generations, 2)
        self.stage2_train_steps = min(self.stage2_train_steps, 64)
        self.llm_provider = "seed"
        self.device = "cpu"
        self.n_envs = 1
        self.label_workers = 1
        self.n_crossover = self.n_mutation = self.n_random = None
