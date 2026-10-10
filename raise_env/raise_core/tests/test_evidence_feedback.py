"""Evidence-driven feedback (highway) — numbers only, no hand-coded verdicts."""

from __future__ import annotations

import domains.highway.prompts as hwy_prompts
from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.reward_checks import make_highway_validator
from raise_core.explore import (
    RewardCandidate,
    StageIEvolver,
    extract_llm_diagnosis,
)
from raise_core.feedback import FeedbackCardConfig
from raise_core.llm import ScriptedLLMClient
from raise_core.raise_loop.proxy_feedback import attach_proxy_feedback, evidence_block

_VERDICT_WORDS = ("lags", "hack", "degenerate", "improve")


def _hwy_metrics(**extra):
    m = {
        "domain": "highway",
        "SR": 0.8,
        "CR": 0.1,
        "TR": 0.0,
        "mean_speed": 20.0,
        "PL": 700.0,
        "mean_progress": 700.0,
        "soft_success": 0.2,
        "lane_change_rate": 0.01,
        "speed_p10": 19.0,
        "speed_p90": 21.0,
        "n_eval_episodes": 20,
        "fitness": 0.4,
    }
    m.update(extra)
    return m


def _cand(cid: str = "c0", *, score: float = 0.5, metrics=None) -> RewardCandidate:
    fn = make_highway_validator().validate_code(D5_SEED_FUNCTION)
    md = {}
    if metrics is not None:
        md["last_metrics"] = dict(metrics)
    return RewardCandidate(
        candidate_id=cid,
        code=D5_SEED_FUNCTION,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=score,
        metadata=md,
    )


def _evolver(*, card: bool = True) -> StageIEvolver:
    return StageIEvolver(
        ScriptedLLMClient([D5_SEED_FUNCTION] * 2),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.0, "rejected": False})(),
        validator=make_highway_validator(),
        prompts=hwy_prompts,
        feedback=FeedbackCardConfig(enabled=card),
    )


def test_evidence_block_is_numbers_only():
    block = evidence_block(_hwy_metrics())
    low = block.lower()
    for word in _VERDICT_WORDS:
        assert word not in low, f"verdict word {word!r} leaked into evidence"
    assert "SR=0.80" in block
    assert "fitness=0.400" in block
    assert "score1" not in low


def test_attach_always_writes_metrics_and_evidence():
    c = _cand(metrics=_hwy_metrics())
    attach_proxy_feedback(c, _hwy_metrics(mean_speed=24.5))
    md = c.metadata or {}
    assert md["last_metrics"]["mean_speed"] == 24.5
    fb = md["proxy_feedback"]
    assert "SR=0.80" in fb
    assert "mean_speed=24.5" in fb
    assert "Score1" not in fb


def test_extract_llm_diagnosis():
    raw = (
        "Speed term is saturated while CR rises.\n"
        "Need clearer safety vs throughput trade-off.\n"
        "```python\n"
        "def compute_reward(state, memory):\n"
        "    return 0.0\n"
        "```\n"
    )
    diag = extract_llm_diagnosis(raw)
    assert diag is not None
    assert "saturated" in diag
    assert "def compute_reward" not in diag
    assert extract_llm_diagnosis("```python\ndef compute_reward(state, memory):\n return 0.0\n```") is None


def test_build_reflection_is_fitness_table_with_diagnoses():
    evo = _evolver()
    pop = [
        _cand("b", score=0.1, metrics=_hwy_metrics(mean_speed=24.0, fitness=0.6)),
        _cand("a", score=0.2, metrics=_hwy_metrics(mean_speed=20.0, fitness=0.3)),
    ]
    pop[0].metadata["fitness"] = 0.6
    pop[1].metadata["fitness"] = 0.3
    pop[0].metadata["llm_diagnosis"] = "CR rising with speed shaping."
    note = evo._build_reflection(pop, generation=1)
    assert "strengthen goal progress" not in note
    assert "Population(fitness desc):" in note
    assert "b: SR=0.80 CR=0.10 mean_speed=24.0 fitness=0.600" in note
    assert "best_fitness_trend=[0.600]" in note
    assert "PriorLLMDiagnoses:" in note
    assert "CR rising" in note


def test_mutation_weakness_uses_evidence_block():
    """With the feedback card disabled, mutation keeps the plain evidence text."""
    parent = _cand("p0", score=0.4)
    attach_proxy_feedback(parent, _hwy_metrics(TR=0.7))
    evo = _evolver(card=False)
    captured = {}

    def _fake_format(code, weakness, **kwargs):
        captured["weakness"] = weakness
        return f"MUTATE:\n{code}"

    orig = evo._format_d2_mutation
    evo._format_d2_mutation = _fake_format
    try:
        evo._try_mutate(parent, "global note", phase="mutation", attempt=0)
    finally:
        evo._format_d2_mutation = orig
    assert "Holdout eval" in captured["weakness"]
    assert "TR=0.70" in captured["weakness"]
    assert "Global reflection: global note" in captured["weakness"]
    assert "Score1" not in captured["weakness"]

    bare = _cand("p1", score=0.9)
    evo._format_d2_mutation = _fake_format
    try:
        evo._try_mutate(bare, "global note", phase="mutation", attempt=0)
    finally:
        evo._format_d2_mutation = orig
    assert "no holdout evaluation yet" in captured["weakness"]
    assert "Score1" not in captured["weakness"]


def test_highway_prompts_render():
    text = hwy_prompts.format_d2_mutation(
        "def compute_reward(state, memory):\n    return 0.0\n",
        "evidence here",
    )
    assert "Before writing code" in text
    assert "evidence here" in text
    assert "sigmoid" not in hwy_prompts.D1_SYSTEM_PROMPT.lower()
    assert "Do NOT reward lagging" not in hwy_prompts.D1_SYSTEM_PROMPT
    assert "Do NOT reward lagging" not in hwy_prompts.D4_EXTERNAL_KNOWLEDGE
