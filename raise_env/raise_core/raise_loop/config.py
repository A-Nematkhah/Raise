"""Closed-loop reward evolution config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from raise_core.config_defaults import (
    budget_generations,
    budget_population,
    budget_stage1_dataset_path,
    budget_stage2_paired_seeds,
    budget_stage2_seeds,
    budget_stage2_train_steps,
    env_action_mode,
    env_kinematics,
    env_rotation_samples,
    env_speed_samples,
    ppo_normalize_reward,
    run_seed,
    selection_infra_error_policy,
    selection_max_fallback_fraction,
    selection_mutation_parents,
    selection_strict_fallback,
    upstream_env_config,
)
from raise_core.feedback import FeedbackCardConfig
from raise_core.evo_selection.config import SelectionConfig


@dataclass
class ClosedLoopConfig:
    """Knobs for ``ClosedLoopRunner``."""

    population_size: int = field(default_factory=budget_population)
    generations: int = field(default_factory=budget_generations)
    # Stage II PPO budget per candidate (environment steps).
    stage2_train_steps: int = field(default_factory=budget_stage2_train_steps)
    stage1_dataset_path: str = field(default_factory=budget_stage1_dataset_path)
    seed: int = field(default_factory=run_seed)
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
    mutation_parents: str = field(default_factory=selection_mutation_parents)
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
    # Shared per-generation PPO train seeds (decoupled from candidate index).
    stage2_paired_seeds: bool = field(default_factory=budget_stage2_paired_seeds)
    # Number of shared train seeds per Stage II candidate (K); fitness = mean.
    stage2_seeds: int = field(default_factory=budget_stage2_seeds)
    eval_mode: str = "holdout_only"  # both | holdout_only
    # Stage II selection pool is held-out val; Stage III/final use val/test.
    stage2_eval_split: str = "val"
    # Robot / action space (domains.crowdnav.settings).
    action_mode: str = field(default_factory=env_action_mode)
    robot_kinematics: str = field(default_factory=env_kinematics)
    action_speed_samples: int = field(default_factory=env_speed_samples)
    action_rotation_samples: int = field(default_factory=env_rotation_samples)
    env_config_path: Optional[str] = None
    env_config_obj: Optional[Any] = field(default_factory=upstream_env_config, repr=False)
    # Infrastructure vs reward failure handling in Stage II labeling.
    infra_error_policy: str = field(default_factory=selection_infra_error_policy)
    # Resume with differing result-changing config keys aborts unless True.
    allow_config_drift: bool = False
    # LLM fallback contamination threshold (fraction of population).
    max_fallback_fraction: float = field(default_factory=selection_max_fallback_fraction)
    strict_fallback: bool = field(default_factory=selection_strict_fallback)
    # Optional PPO reward normalization (training VecEnv only).
    ppo_normalize_reward: bool = field(default_factory=ppo_normalize_reward)
    # Diagnostic feedback card + LLM input-token budget.
    feedback: FeedbackCardConfig = field(default_factory=FeedbackCardConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

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
