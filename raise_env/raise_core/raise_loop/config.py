"""Closed-loop reward evolution config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClosedLoopConfig:
    """Knobs for ``ClosedLoopRunner``."""

    population_size: int = 6
    generations: int = 4
    # Stage II PPO budget per candidate (environment steps).
    stage2_train_steps: int = 12_000
    stage1_dataset_path: str = "domains/highway/data/stage1_dataset"
    seed: int = 425
    device: str = "cpu"
    use_stub: bool = False
    llm_provider: str = "seed"
    output_dir: str = "results/closed_loop"
    keep_runtime_elite: bool = True
    n_crossover: int = 2
    n_mutation: int = 4
    n_random: int = 2
    # Crash-safe resume from ``checkpoint.json`` written under the run dir.
    resume: bool = True
    domain: str = "highway"
    # Stage II eval episodes (None → 20; 8 with stub trainers).
    eval_episodes: Optional[int] = None
    # Order genomes for next-generation parents: scalar (fitness) | score1 | hybrid.
    evolve_rank: str = "scalar"
    # Weight on Score1 inside hybrid mode (remainder on fitness).
    evolve_rank_score1_weight: float = 0.4
    # Highway speed knobs (same budget; higher wall throughput).
    highway_n_envs: int = 1
    highway_label_workers: int = 1
    highway_warm_start: bool = True
    highway_eval_mode: str = "holdout_only"  # both | holdout_only
    highway_action_mode: str = "meta_default"
    highway_action_continuous_lateral: bool = True
    highway_meta_fine_low: float = 15.0
    highway_meta_fine_high: float = 35.0
    highway_meta_fine_n: int = 21

    def apply_fast_profile(self) -> None:
        self.use_stub = True
        self.population_size = min(self.population_size, 4)
        self.generations = min(self.generations, 2)
        self.stage2_train_steps = min(self.stage2_train_steps, 64)
        self.llm_provider = "seed"
        self.device = "cpu"
        self.highway_n_envs = 1
        self.highway_label_workers = 1
        # Keep 2/4/2 ratios scaled down for N=4: 1/2/1
        n = int(self.population_size)
        self.n_crossover = min(1, n)
        self.n_mutation = min(2, max(0, n - self.n_crossover))
        self.n_random = max(0, n - self.n_crossover - self.n_mutation)
