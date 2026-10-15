"""P1-5: behavioural duplicate detection."""

from __future__ import annotations

from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import behavior_fingerprint, genome_fingerprint
from raise_core.explore import StageIConfig, StageIEvolver
from raise_core.feedback import FeedbackCardConfig
from raise_core.llm import ScriptedLLMClient
from raise_core.rejection_log import categorize_validation_error
import domains.crowdnav.prompts as cn_prompts


def test_behavior_fingerprint_collides_on_comment_only_diff():
    validator = make_crowdnav_validator()
    a = D5_SEED_FUNCTION
    b = D5_SEED_FUNCTION.replace(
        '"""CrowdNav seed: the upstream CrowdSim reward (crowd_nav/configs/env.config)."""',
        '"""Different comment only."""',
    )
    # float() cast changes the AST fingerprint but not the totals.
    b2 = a.replace("discomfort_penalty_factor = 0.5", "discomfort_penalty_factor = float(0.5)")
    fa, ea = validator.try_validate(a)
    fb, eb = validator.try_validate(b)
    fc, ec = validator.try_validate(b2)
    assert fa and fb and fc, (ea, eb, ec)
    states = validator.smoke_states
    assert behavior_fingerprint(fa, states) == behavior_fingerprint(fb, states)
    assert behavior_fingerprint(fa, states) == behavior_fingerprint(fc, states)
    assert genome_fingerprint(a) == genome_fingerprint(b)
    assert genome_fingerprint(a) != genome_fingerprint(b2)


def test_behavior_fingerprint_differs_for_constant_change():
    validator = make_crowdnav_validator()
    a = D5_SEED_FUNCTION
    b = D5_SEED_FUNCTION.replace("success_reward = 1.0", "success_reward = 2.0")
    fa, _ = validator.try_validate(a)
    fb, _ = validator.try_validate(b)
    assert fa and fb
    assert behavior_fingerprint(fa, validator.smoke_states) != behavior_fingerprint(
        fb, validator.smoke_states
    )


def test_categorize_duplicate_behavior():
    assert (
        categorize_validation_error("duplicate behaviour of cro_0026")
        == "duplicate_behavior"
    )


def test_evolver_rejects_behavioral_duplicate(tmp_path):
    validator = make_crowdnav_validator()
    twin = D5_SEED_FUNCTION.replace(
        "discomfort_penalty_factor = 0.5", "discomfort_penalty_factor = float(0.5)"
    )
    assert genome_fingerprint(D5_SEED_FUNCTION) != genome_fingerprint(twin)
    evo = StageIEvolver(
        ScriptedLLMClient(["unused"]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.1, "rejected": False})(),
        validator=validator,
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(enabled=False),
        config=StageIConfig(population_size=2, behavioral_dedup=True),
        rejection_log_path=str(tmp_path / "rej.jsonl"),
    )
    first = evo._attempt_candidate(
        D5_SEED_FUNCTION, origin="initial", phase="test", attempt=1
    )
    assert first.valid
    second = evo._attempt_candidate(
        twin, origin="initial", phase="test", attempt=2
    )
    assert not second.valid
    assert "duplicate behaviour" in (second.validation_error or "")
    assert second.metadata.get("behavior_fp") == first.metadata.get("behavior_fp")
