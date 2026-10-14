"""CrowdNav reward sandbox: field allowlist, runtime limits, math, rejection buckets, prompts."""

from __future__ import annotations

import math

import pytest

from domains.crowdnav import prompts
from domains.crowdnav.adapter import clone_reward_fn
from domains.crowdnav.prompts import D5_SEED_FUNCTION, format_d1_initial
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.sandbox_fields import (
    CROWDNAV_FORBIDDEN_HALLUCINATIONS,
    CROWDNAV_STATE_ATTRIBUTES,
    crowdnav_sandbox_config,
)
from domains.crowdnav.settings import CrowdNavSettings, action_space_prompt_block
from domains.crowdnav.state import CrowdNavRewardState, default_smoke_states, fingerprint_smoke_states
from raise_core.domains import load_domain, make_validator_for_domain
from raise_core.llm import SeedVariantLLMClient, extract_python_code
from raise_core.rejection_log import categorize_validation_error
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import SandboxedReward, compile_compute_reward

_FN = "def compute_reward(state, memory):\n"
_FAST = _FN + "    return float(state.progress)\n"


def _validator():
    return make_validator_for_domain(load_domain("crowdnav"))


# ------------------------------------------------------------------ allowlist


def test_allowlist_matches_the_reward_state_dataclasses():
    fields = set(CrowdNavRewardState.__dataclass_fields__)
    robot = default_smoke_states()[0].robot
    human = default_smoke_states()[0].humans[0]
    expected = fields | set(robot.__dataclass_fields__) | set(human.__dataclass_fields__)
    assert set(CROWDNAV_STATE_ATTRIBUTES) == expected
    assert not set(CROWDNAV_FORBIDDEN_HALLUCINATIONS) & expected
    cfg = crowdnav_sandbox_config()
    assert cfg.restrict_top_level and "humans" in cfg.allowed_attributes


def test_seed_and_human_clearance_rewards_validate():
    v = _validator()
    assert v.validate_code(D5_SEED_FUNCTION) is not None
    ok = _FN + (
        "    nearest = 10.0\n"
        "    for h in state.humans:\n"
        "        d = ((h.px - state.robot.px) ** 2 + (h.py - state.robot.py) ** 2) ** 0.5\n"
        "        nearest = min(nearest, d - h.radius - state.robot.radius)\n"
        "    return float(state.progress + 0.1 * nearest)\n"
    )
    assert v.validate_code(ok) is not None


@pytest.mark.parametrize("field", ["self_state", "dmin", "position", "ego", "others", "speed"])
def test_hallucinated_fields_are_rejected(field):
    with pytest.raises(RewardSandboxError) as ei:
        _validator().validate_code(_FN + f"    return float(state.{field})\n")
    assert field in str(ei.value)


def test_seed_variants_are_valid_and_diverse():
    v = _validator()
    client = SeedVariantLLMClient(base_code=D5_SEED_FUNCTION, variant_fn=prompts.seed_variant)
    codes = [extract_python_code(client.complete("gen")) for _ in range(10)]
    for code in codes:
        v.validate_code(code)
    assert len(set(codes)) >= 8
    joined = "\n".join(codes)
    assert "state.humans" in joined and "state.goal_distance" in joined


# ------------------------------------------------------------------ runtime limits


def _reward(code: str, **cfg_overrides) -> SandboxedReward:
    cfg = crowdnav_sandbox_config(**cfg_overrides)
    return SandboxedReward(compile_compute_reward(code, cfg), cfg, source_code=code, allow_components=True)


def test_smoke_rejects_state_dependent_slow_loop():
    code = _FN + (
        "    total = 0.0\n"
        "    for i in range(int(state.goal_distance * 2000000)):\n"
        "        total = total + 0.0\n"
        "    return total\n"
    )
    reward_fn, err = make_crowdnav_validator(smoke_states=fingerprint_smoke_states()).try_validate(code)
    assert reward_fn is None and "too slow" in (err or "")


def test_compute_rejects_a_single_call_over_budget():
    with pytest.raises(RewardSandboxError, match="too slow"):
        _reward(_FAST, max_call_seconds=0.0).compute(default_smoke_states()[0])


def test_huge_reward_scale_rejected_but_bounded_terminal_penalty_accepted():
    _, err = make_crowdnav_validator().try_validate(_FN + "    return 1e15 * float(state.progress)\n")
    assert "magnitude" in (err or "")
    code = _FN + (
        "    if state.collision:\n"
        "        return -1000.0, {'crash': -1000.0}\n"
        "    return float(state.progress), {'progress': float(state.progress)}\n"
    )
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is not None, err


def test_compute_rejects_out_of_bound_reward_at_train_time():
    reward = _reward(_FN + "    return float(state.goal_distance * 10.0)\n", max_abs_reward=50.0)
    with pytest.raises(RewardSandboxError, match="magnitude"):
        for state in default_smoke_states():
            reward.compute(state)


def test_memory_is_per_episode_and_clones_are_independent():
    reward = _validator().validate_code(
        _FN + "    memory['n'] = memory.get('n', 0) + 1\n    return float(memory['n'])\n"
    )
    state = default_smoke_states()[0]
    clone = clone_reward_fn(reward)
    assert clone is not reward
    assert (reward.compute(state), reward.compute(state), clone.compute(state)) == (1.0, 2.0, 1.0)
    reward.reset()
    assert reward.compute(state) == 1.0


# ------------------------------------------------------------------ math / buckets


@pytest.mark.parametrize("expr", ["math.sqrt(1.0 + s)", "math.exp(-s)", "math.tanh(s)",
                                  "math.hypot(state.robot.vx, state.robot.vy)"])
def test_safe_math_functions_are_accepted(expr):
    code = _FN + f"    s = float(state.goal_distance)\n    return float({expr})\n"
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is not None, err
    assert math.isfinite(reward_fn.compute(default_smoke_states()[0]))


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def compute_reward(state, memory)\n    return 0.0\n", "syntax"),
        ("import os\n" + _FN + "    return 0.0\n", "forbidden_import"),
        (_FN + "    return float(state.dmin)\n", "not_on_allowlist"),
        (_FN + "    xs = [h.px for h in state.humans]\n    return 0.0\n", "forbidden_construct"),
        ("HIST = []\n" + _FN + "    return 0.0\n", "mutable_global"),
        (_FN + "    return 1e15\n", "reward_too_large"),
        (_FN + "    return float('nan')\n", "non_finite_output"),
        ("def compute_reward(state, memory, extra):\n    return 0.0\n", "wrong_signature"),
        (_FN + "    return float(state.humans[99].px)\n", "runtime_error"),
    ],
)
def test_validator_messages_map_to_buckets(code, expected):
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is None
    assert categorize_validation_error(err) == expected, err


# ------------------------------------------------------------------ prompts


def test_prompts_describe_the_crowdnav_state_and_objective():
    user = format_d1_initial()
    for needle in ("CrowdNavRewardState", "state.humans", "state.min_separation", "state.reached_goal",
                   "math.tanh", "never write `import math`"):
        assert needle in user, needle
    assert "CrowdNav" in prompts.D1_SYSTEM_PROMPT
    assert "def compute_reward(state, memory)" in D5_SEED_FUNCTION
    for name in ("format_d1_initial_batch", "format_d2_crossover", "format_d2_mutation"):
        assert callable(getattr(prompts, name))
    low = (user + prompts.D1_SYSTEM_PROMPT + prompts.D4_EXTERNAL_KNOWLEDGE).lower()
    for word in ("highway", "lane", "vehicle", "overtake"):
        assert word not in low, word


def test_action_space_prompt_block_follows_kinematics():
    holo = action_space_prompt_block(CrowdNavSettings())
    uni = action_space_prompt_block(CrowdNavSettings(kinematics="unicycle", action_mode="continuous"))
    assert "(vx, vy)" in holo and "81 discrete CADRL actions" in holo
    assert "(v, r)" in uni and "continuous" in uni
