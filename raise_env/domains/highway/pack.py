"""Highway-Env Fast domain pack for RAISE (diagnostic / additive)."""

from __future__ import annotations

import os

from raise_core.domains.base import DomainPack
from domains.highway import prompts as highway_prompts
from domains.highway.adapter import (
    HighwayAdapter,
    make_stage2_trainer,
    make_stage3_trainer,
)
from domains.highway.stage1 import (
    DEFAULT_STAGE1_DATASET,
    make_score_fn as highway_make_score_fn,
)
from domains.highway.sandbox_fields import highway_sandbox_config
from domains.highway.state import fingerprint_smoke_states

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))


def get_pack(*, with_adapter: bool = True) -> DomainPack:
    adapter = HighwayAdapter() if with_adapter else None
    return DomainPack(
        name="highway",
        display_name="Highway-Env Fast (diagnostic)",
        description=(
            "Multi-lane highway driving (highway-fast-v0) for fast RAISE "
            "algorithm validation; SB3 PPO; not a CrowdNav paper substitute."
        ),
        prompts=highway_prompts,
        seed_reward_source=highway_prompts.D5_SEED_FUNCTION,
        spec_path=os.path.join(_PACK_DIR, "spec.md"),
        adapter=adapter,
        stage1_dataset_default=DEFAULT_STAGE1_DATASET,
        make_score_fn=highway_make_score_fn,
        smoke_states_fn=fingerprint_smoke_states,
        sandbox_config=highway_sandbox_config(),
        make_stage2_trainer=make_stage2_trainer if with_adapter else None,
        make_stage3_trainer=make_stage3_trainer if with_adapter else None,
        selection_scalar_name="highway_fitness",
        metadata={
            "env_id": "highway-fast-v0",
            "policy": "sb3_ppo_mlp",
            "pipeline_profile": "highway",
            "stage1": "highway_spearman_rules",
            "diagnostic_only": True,
            "fitness": "highway_fitness",
            "selection_scalar": "highway_fitness",  # alias
            "objective": (
                "Official fitness = highway_fitness(holdout metrics). "
                "Score1 is Stage-I proxy only; PPO optimizes LLM compute_reward."
            ),
            "metrics_note": (
                "SR/CR/TR = survival rates; PL=progress_m; ITR=mean_speed_m/s; "
                "also soft_success, lane_change_rate, high_speed_frac; "
                "fitness is the sole selection objective"
            ),
        },
    )
