"""CrowdNav++ / RAISE domain pack (default baseline)."""

from __future__ import annotations

import os

from crowd_nav.domains.base import DomainPack
from crowd_nav.domains.crowdnav import prompts as crowdnav_prompts
from crowd_nav.domains.crowdnav.adapter import (
    CrowdNavAdapter,
    make_stage2_trainer,
    make_stage3_trainer,
)
from crowd_nav.domains.crowdnav.explore_score import (
    DEFAULT_STAGE1_DATASET,
    make_score_fn as crowdnav_make_score_fn,
)

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))


def get_pack(*, with_adapter: bool = True) -> DomainPack:
    """
    Build the CrowdNav DomainPack.

    Stage I scoring is exposed via ``make_score_fn`` (Score1 / smoke).
    Stage II/III trainers via ``make_stage*_trainer`` on the adapter module.
    """
    adapter = CrowdNavAdapter() if with_adapter else None

    return DomainPack(
        name="crowdnav",
        display_name="CrowdNav++ (RAISE)",
        description=(
            "Robot crowd navigation in continuous 2D; LLM-proposed rewards "
            "scored with Score1 then trained via A2C (Stage II) / PPO (Stage III)."
        ),
        prompts=crowdnav_prompts,
        seed_reward_source=crowdnav_prompts.D5_SEED_FUNCTION,
        spec_path=os.path.join(_PACK_DIR, "spec.md"),
        adapter=adapter,
        stage1_dataset_default=DEFAULT_STAGE1_DATASET,
        make_score_fn=crowdnav_make_score_fn,
        make_stage2_trainer=make_stage2_trainer if with_adapter else None,
        make_stage3_trainer=make_stage3_trainer if with_adapter else None,
        selection_scalar_name="navigation_scalar",
        metadata={
            "env_inferred": "CrowdSimPredRealGST-v0",
            "env_none": "CrowdSimVarNum-v0",
            "policy": "selfAttn_merge_srnn",
            "paper": "arXiv:2605.11859",
            "stage1": "score1_spearman_rules",
            "pipeline_profile": "crowdnav",
        },
    )
