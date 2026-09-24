"""Highway-Env Fast domain pack for RAISE (diagnostic / additive)."""

from __future__ import annotations

import os

from crowd_nav.domains.base import DomainPack
from crowd_nav.domains.highway import prompts as highway_prompts
from crowd_nav.domains.highway.adapter import (
    HighwayAdapter,
    make_stage2_trainer,
    make_stage3_trainer,
)
from crowd_nav.domains.highway.stage1 import (
    DEFAULT_STAGE1_DATASET,
    make_score_fn as highway_make_score_fn,
)
from crowd_nav.domains.highway.state import default_smoke_states

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
        smoke_states_fn=default_smoke_states,
        make_stage2_trainer=make_stage2_trainer if with_adapter else None,
        make_stage3_trainer=make_stage3_trainer if with_adapter else None,
        selection_scalar_name="navigation_scalar",
        metadata={
            "env_id": "highway-fast-v0",
            "policy": "sb3_ppo_mlp",
            "pipeline_profile": "highway",
            "stage1": "highway_spearman_rules",
            "diagnostic_only": True,
        },
    )
