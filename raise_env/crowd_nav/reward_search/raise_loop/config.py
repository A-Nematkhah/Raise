"""Closed-loop multi-fidelity config (innovation path). See PLAN.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClosedLoopConfig:
    """Knobs for ``ClosedLoopRunner`` (PLAN §4)."""

    population_size: int = 8
    generations: int = 10
    min_labels_for_gate: int = 24
    refit_every_new_labels: int = 8
    stage2_train_steps: int = 4_000
    k2_unit: str = "gradient_steps"
    drop_fraction: float = 0.25
    max_uncertainty_to_drop: float = 0.15
    min_keep: int = 2
    min_stage2_per_gen: int = 4
    al_enabled: bool = True
    al_max_per_epoch: int = 4
    al_allow_stage1_requests: bool = True
    enable_refine: bool = False  # reserved; v1 does not run D.3 inside loop
    final_stage2_rounds: int = 0
    surrogate_model_dir: str = "artifacts/surrogate"
    surrogate_dataset: str = "data/surrogate_dataset"
    al_root: str = "data/active_learning"
    stage1_dataset_path: str = "data/stage1_dataset"
    label_rejected: bool = True
    seed: int = 425
    device: str = "cpu"
    num_processes: int = 1
    use_stub: bool = False
    llm_provider: str = "seed"
    output_dir: str = "results/closed_loop"
    keep_runtime_elite: bool = False
    n_crossover: int = 2
    n_mutation: int = 4
    n_random: int = 2
    # Stage II env (must match pipeline --easy / --regime / --predict-method).
    human_num: int = 20
    predict_method: str = "inferred"
    randomization_regime: str = "without_random"
    horizon_steps: int = 100
    # Crash-safe resume from ``checkpoint.json`` written under the run dir.
    resume: bool = True
    # Keep surrogate model / dataset / AL queue inside output_dir so a run is
    # self-contained and resumable without guessing sibling paths.
    isolate_run_artifacts: bool = True

    def apply_fast_profile(self) -> None:
        self.use_stub = True
        self.population_size = min(self.population_size, 4)
        self.generations = min(self.generations, 2)
        self.min_labels_for_gate = 4
        self.refit_every_new_labels = 2
        self.stage2_train_steps = min(self.stage2_train_steps, 64)
        self.k2_unit = "env_steps"
        self.al_max_per_epoch = 2
        self.min_stage2_per_gen = 2
        self.llm_provider = "seed"
        self.device = "cpu"
        self.num_processes = 1
        # Keep 2/4/2 ratios scaled down for N=4: 1/2/1
        n = int(self.population_size)
        self.n_crossover = min(1, n)
        self.n_mutation = min(2, max(0, n - self.n_crossover))
        self.n_random = max(0, n - self.n_crossover - self.n_mutation)
