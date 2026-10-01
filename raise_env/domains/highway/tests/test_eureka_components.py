"""Highway EUREKA-style reward-component returns + reflection."""

from __future__ import annotations

import pytest

from domains.highway.diagnostics_callback import RolloutDiagnosticsCallback
from domains.highway.prompts import (
    D5_SEED_FUNCTION,
    _DIAGNOSIS_BEFORE_CODE,
    format_d2_mutation,
)
from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states
from raise_core.domains import load_domain, make_validator_for_domain
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


def test_crowdnav_rejects_tuple_return():
    pack = load_domain("crowdnav")
    v = make_validator_for_domain(pack)
    assert getattr(v, "allow_components", False) is False
    with pytest.raises(RewardSandboxError) as ei:
        v.validate_code(
            "def compute_reward(state, memory):\n"
            '    return 1.0, {"a": 1.0}\n'
        )
    assert "float" in str(ei.value).lower() or "dict" in str(ei.value).lower()


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
    # Stage III refine in highway_4h_20260930 died on U+2011 / "25\u202fm"-style tokens.
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


def test_unpack_shim_and_strict_crowdnav():
    total, comps = unpack_reward_return(3.0, allow_components=True)
    assert total == 3.0 and comps == {}
    with pytest.raises(RewardSandboxError):
        unpack_reward_return((1.0, {"a": 1.0}), allow_components=False)


def test_components_not_aggregated_outside_rollout_callback():
    """Env / SandboxedReward keep last-step only; history grows only on flush."""
    v = make_highway_validator(smoke_states=default_smoke_states())
    rw = v.validate_code(D5_SEED_FUNCTION)
    st = default_smoke_states()[0]
    for _ in range(5):
        rw.compute(st)
    # Only last step retained on the reward object — no history list.
    assert not hasattr(rw, "component_history")
    last = rw.last_reward_components()
    assert isinstance(last, dict)

    cb = RolloutDiagnosticsCallback(candidate_id="c0", log_path="/tmp/x.jsonl")
    for i in range(4):
        cb._ingest_step_components({"const_term": 1.0, "vary_term": float(i)})
        # Mid-rollout: history must still be empty (no aggregation yet).
        assert cb.component_history == {}
    means = cb._flush_rollout_component_means()
    assert means["const_term"] == pytest.approx(1.0)
    assert means["vary_term"] == pytest.approx(1.5)
    assert len(cb.component_history["const_term"]) == 1
    # Second rollout
    for _ in range(2):
        cb._ingest_step_components({"const_term": 1.0})
    cb._flush_rollout_component_means()
    assert len(cb.component_history["const_term"]) == 2
    assert abs(max(cb.component_history["const_term"]) - min(cb.component_history["const_term"])) < 1e-9


def test_constant_component_flat_in_trend_summary():
    cb = RolloutDiagnosticsCallback(candidate_id="c0", log_path="/tmp/x.jsonl")
    for rollout in range(5):
        for _step in range(8):
            cb._ingest_step_components(
                {
                    "fixed_term": 0.42,
                    # Vary *across* rollouts so Max/Min of checkpoint means differ.
                    "progress_term": float(rollout) + 0.1 * float(_step),
                }
            )
        cb._flush_rollout_component_means()
    summary = cb.component_trend_summary()
    fixed = summary["fixed_term"]
    assert fixed["max"] == pytest.approx(fixed["mean"])
    assert fixed["min"] == pytest.approx(fixed["mean"])
    prog = summary["progress_term"]
    assert prog["max"] > prog["min"]


def test_evidence_block_and_mutation_include_component_trends():
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
    md = {
        "pareto_rank": 1,
        "pareto_n": 8,
        "pareto_feasible": True,
        "pareto_front": 1,
        "pareto_require_survival": True,
        "pareto_calibration_source": "no_speed_floor",
        "pareto_objectives": [
            "-CR",
            "-TR",
            "progress",
            "mean_speed",
            "lane_change_rate",
            "overtakes_per_km",
        ],
        "reward_component_trends": {
            "speed_term": {
                "values": [0.12, 0.31, 0.30, 0.34],
                "max": 0.34,
                "mean": 0.2675,
                "min": 0.12,
            },
            "collision_penalty": {
                "values": [0.0, 0.0, 0.0, 0.0],
                "max": 0.0,
                "mean": 0.0,
                "min": 0.0,
            },
        },
    }
    block = evidence_block(metrics, score1=0.5, metadata=md)
    assert "Holdout eval" in block
    assert "Reward component trends" in block
    assert "speed_term:" in block
    assert "collision_penalty:" in block
    assert "Max:" in block and "Mean:" in block and "Min:" in block

    assert "survival/success rate is always near zero" in _DIAGNOSIS_BEFORE_CODE
    assert "nearly identical across all checkpoints" in _DIAGNOSIS_BEFORE_CODE
    assert "significantly larger than the others" in _DIAGNOSIS_BEFORE_CODE

    mut = format_d2_mutation(
        "def compute_reward(state, memory):\n    return 0.0, {}\n",
        "Parent x\n" + block + "\nGlobal reflection: test",
    )
    assert "Reward component trends" in mut
    assert "returning (float, dict[str, float])" in mut or "dict[str, float]" in mut
    assert "survival/success rate is always near zero" in mut


def test_default_validator_still_scalar_only():
    v = RewardValidator()
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
