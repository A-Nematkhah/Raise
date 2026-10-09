"""Highway reward-component returns ``(float, dict[str, float])`` in the sandbox."""

from __future__ import annotations

import pytest

from domains.highway.prompts import (
    D5_SEED_FUNCTION,
    _DIAGNOSIS_BEFORE_CODE,
    format_d2_mutation,
)
from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states
from raise_core.raise_loop.proxy_feedback import evidence_block
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import (
    MAX_REWARD_COMPONENT_KEYS,
    SandboxedReward,
    unpack_reward_return,
)
from raise_core.sandbox.validator import RewardValidator


def test_highway_accepts_tuple_dict_and_shim_float():
    v = make_highway_validator(smoke_states=default_smoke_states())
    seed = v.validate_code(D5_SEED_FUNCTION)
    assert isinstance(seed, SandboxedReward)
    assert seed._allow_components is True
    st = default_smoke_states()[0]
    total = seed.compute(st)
    assert isinstance(total, float)
    comps = seed.last_reward_components()
    assert isinstance(comps, dict)
    assert "progress_term" in comps or "collision_penalty" in comps

    bare = v.validate_code(
        "def compute_reward(state, memory):\n    return 1.5\n"
    )
    assert bare.compute(st) == pytest.approx(1.5)
    assert bare.last_reward_components() == {}


@pytest.mark.parametrize(
    "code,needle",
    [
        (
            "def compute_reward(state, memory):\n"
            '    return 1.0, {"a": {"nested": 1.0}}\n',
            "nested",
        ),
        (
            "def compute_reward(state, memory):\n"
            "    return 1.0, {1: 2.0}\n",
            "str",
        ),
        (
            "def compute_reward(state, memory):\n"
            "    return 1.0, {"
            + ", ".join(f'"k{i}": 1.0' for i in range(MAX_REWARD_COMPONENT_KEYS + 1))
            + "}\n",
            "max",
        ),
        (
            "def compute_reward(state, memory):\n"
            "    return 1.0, [1.0, 2.0]\n",
            "dict",
        ),
    ],
)
def test_highway_rejects_bad_component_dicts(code: str, needle: str):
    v = make_highway_validator(smoke_states=default_smoke_states())
    with pytest.raises(RewardSandboxError) as ei:
        v.validate_code(code)
    assert needle in str(ei.value).lower()


def test_highway_validator_normalizes_unicode_punctuation():
    # LLMs emit U+2011 / "25\u202fm"-style tokens.
    code = (
        "def compute_reward(state, memory):\n"
        "    # off\u2011road guard, safe speed ~25\u202fm/s\n"
        "    safe = 25.0\u202f\n"
        "    excess = state.speed \u2212 safe\n"
        "    return float(excess), {\u201cexcess\u201d: float(excess)}\n"
    )
    v = make_highway_validator(smoke_states=default_smoke_states())
    rw = v.validate_code(code)
    assert rw.compute(default_smoke_states()[0]) == pytest.approx(0.0)
    assert "excess" in rw.last_reward_components()


def test_unpack_shim_and_strict_scalar_mode():
    total, comps = unpack_reward_return(3.0, allow_components=True)
    assert total == 3.0 and comps == {}
    with pytest.raises(RewardSandboxError):
        unpack_reward_return((1.0, {"a": 1.0}), allow_components=False)


def test_reward_keeps_last_step_components_only():
    v = make_highway_validator(smoke_states=default_smoke_states())
    rw = v.validate_code(D5_SEED_FUNCTION)
    st = default_smoke_states()[0]
    for _ in range(5):
        rw.compute(st)
    assert not hasattr(rw, "component_history")
    assert isinstance(rw.last_reward_components(), dict)


def test_evidence_block_and_mutation_prompt():
    metrics = {
        "SR": 0.4,
        "CR": 0.3,
        "TR": 0.1,
        "mean_speed": 22.0,
        "mean_progress": 80.0,
        "lane_change_rate": 0.05,
        "overtakes_per_km": 0.2,
        "overtake_episode_frac": 0.1,
        "n_eval_episodes": 20,
        "speed_p10": 18.0,
    }
    block = evidence_block(metrics)
    assert "Holdout eval (20 episodes)" in block
    assert "Score1" not in block

    assert "survival/success rate is always near zero" in _DIAGNOSIS_BEFORE_CODE

    mut = format_d2_mutation(
        "def compute_reward(state, memory):\n    return 0.0, {}\n",
        "Parent x\n" + block + "\nGlobal reflection: test",
    )
    assert "Holdout eval" in mut
    assert "dict[str, float]" in mut
    assert "survival/success rate is always near zero" in mut


def test_plain_validator_is_scalar_only():
    v = RewardValidator(smoke_states=default_smoke_states())
    assert v.allow_components is False
    ok, err = v.try_validate(
        "def compute_reward(state, memory):\n    return 1.0\n"
    )
    assert ok is not None and err is None
    bad, err2 = v.try_validate(
        "def compute_reward(state, memory):\n"
        '    return 1.0, {"a": 1.0}\n'
    )
    assert bad is None and err2
