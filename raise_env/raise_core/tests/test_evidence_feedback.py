"""Evidence-driven feedback (CrowdNav) — numbers only, no hand-coded verdicts."""

from __future__ import annotations

import domains.crowdnav.prompts as cn_prompts
from domains.crowdnav.metrics import evidence_line
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import (
    RewardCandidate,
    StageIEvolver,
    extract_llm_diagnosis,
)
from raise_core.feedback import FeedbackCardConfig
from raise_core.llm import ScriptedLLMClient
from raise_core.raise_loop.proxy_feedback import attach_proxy_feedback, evidence_block

_VERDICT_WORDS = ("lags", "hack", "degenerate", "improve")


def _cn_metrics(**extra):
    m = {
        "domain": "crowdnav",
        "SR": 0.8,
        "CR": 0.1,
        "TR": 0.0,
        "NT": 10.5,
        "PL": 8.4,
        "ITR": 0.05,
        "SD": 0.12,
        "n_eval_episodes": 20,
        "fitness": 0.4,
    }
    m.update(extra)
    return m


def _cand(cid: str = "c0", *, score: float = 0.5, metrics=None) -> RewardCandidate:
    fn = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
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
        validator=make_crowdnav_validator(),
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(enabled=card),
    )


def test_evidence_block_is_numbers_only():
    block = evidence_block(_cn_metrics())
    low = block.lower()
    for word in _VERDICT_WORDS:
        assert word not in low, f"verdict word {word!r} leaked into evidence"
    assert "SR=0.80" in block
    assert "fitness=0.400" in block
    assert "score1" not in low


def test_attach_always_writes_metrics_and_evidence():
    c = _cand(metrics=_cn_metrics())
    attach_proxy_feedback(c, _cn_metrics(NT=12.25), evidence_line=evidence_line)
    md = c.metadata or {}
    assert md["last_metrics"]["NT"] == 12.25
    fb = md["proxy_feedback"]
    assert "SR=0.80" in fb
    assert "nav_time=12.25s" in fb
    assert "danger_freq=0.050" in fb
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
        _cand("b", score=0.1, metrics=_cn_metrics(TR=0.1, fitness=0.6)),
        _cand("a", score=0.2, metrics=_cn_metrics(fitness=0.3)),
    ]
    pop[0].metadata["fitness"] = 0.6
    pop[1].metadata["fitness"] = 0.3
    pop[0].metadata["llm_diagnosis"] = "CR rising with speed shaping."
    pop[0].metadata["llm_diagnosis_subjects"] = ["parent_x"]
    note = evo._build_reflection(pop, generation=1)
    assert "strengthen goal progress" not in note
    assert "Population(fitness desc):" in note
    assert "b: SR=0.80 CR=0.10 TR=0.10 fitness=0.600" in note
    assert "best_fitness_trend=[0.600]" in note
    assert "PriorLLMDiagnoses:" in note
    assert "CR rising" in note
    assert "written when creating b from parent_x" in note


def test_batch_candidates_do_not_store_shared_preamble_as_diagnosis(tmp_path):
    """Gen0 batch preamble must not become every child's llm_diagnosis (P0-5)."""
    from pathlib import Path

    from raise_core.explore import StageIConfig

    batch_body = (
        "Below are six independent reward functions for CrowdNav.\n\n"
        + "\n\n".join(
            f"```python\n{D5_SEED_FUNCTION.replace('0.5', str(0.4 + 0.1 * i))}\n```"
            for i in range(2)
        )
    )
    evo = StageIEvolver(
        ScriptedLLMClient([batch_body]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.1, "rejected": False})(),
        validator=make_crowdnav_validator(),
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(enabled=False),
        config=StageIConfig(population_size=2),
        rejection_log_path=str(Path(tmp_path) / "rej.jsonl"),
    )
    pop = evo.initialize_population()
    assert len(pop) >= 2
    assert all(not (c.metadata or {}).get("llm_diagnosis") for c in pop)
    note = evo._build_reflection(pop, generation=0)
    assert "Below are six independent" not in note
    assert "PriorLLMDiagnoses: (none)" in note


def test_mutation_weakness_uses_evidence_block():
    """With the feedback card disabled, mutation keeps the plain evidence text."""
    parent = _cand("p0", score=0.4)
    attach_proxy_feedback(parent, _cn_metrics(TR=0.7))
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


def test_crowdnav_prompts_render():
    text = cn_prompts.format_d2_mutation(
        "def compute_reward(state, memory):\n    return 0.0\n",
        "evidence here",
    )
    assert "Before writing code" in text
    assert "evidence here" in text
    assert "sigmoid" not in cn_prompts.D1_SYSTEM_PROMPT.lower()
    assert "Do NOT reward lagging" not in cn_prompts.D1_SYSTEM_PROMPT
    assert "Do NOT reward lagging" not in cn_prompts.D4_EXTERNAL_KNOWLEDGE
