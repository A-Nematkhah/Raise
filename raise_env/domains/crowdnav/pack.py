"""CrowdNav++ domain pack for RAISE."""

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

UPSTREAM_REPO = "https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph"
UPSTREAM_COMMIT = "390773137be04ed14e27620dc5fd7c5e1a5b1f62"


def get_pack(*, with_adapter: bool = True) -> DomainPack:
    adapter = CrowdNavAdapter() if with_adapter else None
    return DomainPack(
        name="crowdnav",
        display_name="CrowdNav++ (variable-human)",
        description=(
            "Crowd-aware robot navigation in CrowdNav++ CrowdSimVarNum "
            "(variable human population, ORCA humans); SB3 PPO."
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
            "simulator": "Shuijing725/CrowdNav_Prediction_AttnGraph CrowdSimVarNum (vendored, unmodified)",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": UPSTREAM_COMMIT,
            "env_config": "third_party/crowdnavpp/crowd_nav/configs/config.py",
            "policy": "sb3_ppo_multiinput_masked_human_pool",
            "pipeline_profile": "crowdnav",
            "stage1": "crowdnav_outcome_preference",
            "fitness": "crowdnav_task_return",
            "selection_scalar": "crowdnav_task_return",
            "objective": (
                "Official fitness = mean undiscounted sum of native upstream "
                "CrowdNav++ rewards per episode (rl/evaluation.py) on holdout val "
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
