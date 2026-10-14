"""CrowdNav domain pack for RAISE (vita-epfl/CrowdNav simulator)."""

from __future__ import annotations

import os

from raise_core.domains.base import DomainPack
from domains.crowdnav import feedback_card as crowdnav_feedback_card
from domains.crowdnav import prompts as crowdnav_prompts
from domains.crowdnav.adapter import (
    CrowdNavAdapter,
    make_stage2_trainer,
    make_stage3_trainer,
)
from domains.crowdnav.metrics import evidence_line
from domains.crowdnav.post_run import post_run_summary, write_run_artifacts
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.sandbox_fields import crowdnav_sandbox_config
from domains.crowdnav.score1_worker import make_score_fn as crowdnav_make_score_fn
from domains.crowdnav.settings import configure_from_config
from domains.crowdnav.stage1 import DEFAULT_STAGE1_DATASET
from domains.crowdnav.state import fingerprint_smoke_states

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))

UPSTREAM_REPO = "https://github.com/vita-epfl/CrowdNav"
UPSTREAM_COMMIT = "20d678085c06831e658a65b9e20c8bb6f6ecdc10"


def get_pack(*, with_adapter: bool = True) -> DomainPack:
    adapter = CrowdNavAdapter() if with_adapter else None
    return DomainPack(
        name="crowdnav",
        display_name="CrowdNav (vita-epfl)",
        description=(
            "Crowd-aware robot navigation in the vita-epfl/CrowdNav simulator "
            "(ORCA humans, circle crossing); SB3 PPO."
        ),
        prompts=crowdnav_prompts,
        feedback_card=crowdnav_feedback_card,
        seed_reward_source=crowdnav_prompts.D5_SEED_FUNCTION,
        spec_path=os.path.join(_PACK_DIR, "spec.md"),
        adapter=adapter,
        stage1_dataset_default=DEFAULT_STAGE1_DATASET,
        make_score_fn=crowdnav_make_score_fn,
        smoke_states_fn=fingerprint_smoke_states,
        sandbox_config=crowdnav_sandbox_config(),
        make_validator=make_crowdnav_validator,
        configure=configure_from_config,
        evidence_line=evidence_line,
        post_run=write_run_artifacts,
        post_run_summary=post_run_summary,
        make_stage2_trainer=make_stage2_trainer if with_adapter else None,
        make_stage3_trainer=make_stage3_trainer if with_adapter else None,
        selection_scalar_name="crowdnav_task_return",
        metadata={
            "simulator": "vita-epfl/CrowdNav crowd_sim (vendored, unmodified)",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": UPSTREAM_COMMIT,
            "env_config": "third_party/crowdnav/configs/env.config",
            "policy": "sb3_ppo_mlp",
            "pipeline_profile": "crowdnav",
            "stage1": "crowdnav_outcome_preference",
            "fitness": "crowdnav_task_return",
            "selection_scalar": "crowdnav_task_return",
            "objective": (
                "Official fitness = mean discounted upstream CrowdNav task return "
                "(Explorer.run_k_episodes, gamma 0.9 per second) on holdout val "
                "scenarios. Score1 is a Stage-I proxy only; PPO optimizes the LLM "
                "compute_reward."
            ),
            "metrics_note": (
                "SR/CR/TR = success/collision/timeout rates; NT = nav time of "
                "successes (s); PL = path length of successes (m); ITR = danger "
                "frequency; SD = mean min distance in danger (m)"
            ),
        },
    )
