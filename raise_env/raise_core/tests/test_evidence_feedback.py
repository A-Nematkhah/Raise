"""Evidence-driven proxy feedback (highway) — no hand-coded verdicts."""

from __future__ import annotations

import domains.highway.prompts as hwy_prompts
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from raise_core.explore import (
    RewardCandidate,
    StageIEvolver,
    extract_llm_diagnosis,
)
from raise_core.llm import ScriptedLLMClient
from raise_core.raise_loop.proxy_feedback import (
    attach_proxy_feedback,
    evidence_block,
    focus_note_from_metrics,
    format_proxy_feedback_block,
    should_attach_proxy_feedback,
)
from raise_core.sandbox.validator import RewardValidator

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
    fn = RewardValidator().validate_code(D5_SEED_FUNCTION)
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


def test_evidence_block_is_numbers_only():
    block = evidence_block(_hwy_metrics(), score1=0.42)
    low = block.lower()
    for word in _VERDICT_WORDS:
        assert word not in low, f"verdict word {word!r} leaked into evidence"
    assert "SR=0.80" in block
    assert "fitness" not in low
    assert "gate" not in low
    assert "Score1=0.420" in block


def test_highway_attach_always_when_enabled():
    c = _cand(metrics=_hwy_metrics())
    # Good-looking metrics still attach (no threshold gate).
    good = _hwy_metrics(mean_speed=24.5, soft_success=0.7, speed_p10=23.0)
    assert attach_proxy_feedback(
        c,
        good,
        enabled=True,
        n_labeled_dataset=0,
        min_labels=99,
        epoch=0,
    )
    fb = (c.metadata or {})["proxy_feedback"].lower()
    assert "sr=" in fb
    assert "fitness" not in fb
    assert "proxy_feedback_focus" not in (c.metadata or {})
    assert "proxy_hack_mode" not in (c.metadata or {})


def test_crowdnav_attach_gate_unchanged():
    c = _cand()
    m = {"SR": 0.4, "CR": 0.1, "TR": 0.1}
    assert not should_attach_proxy_feedback(m)
    assert not attach_proxy_feedback(
        c, m, enabled=True, n_labeled_dataset=100, min_labels=16, epoch=1
    )
    bad = {"SR": 0.0, "CR": 0.0, "TR": 0.8}
    assert should_attach_proxy_feedback(bad)
    assert attach_proxy_feedback(
        c, bad, enabled=True, n_labeled_dataset=100, min_labels=16, epoch=1
    )
    assert "Focus:" in (c.metadata or {})["proxy_feedback"]


def test_crowdnav_focus_notes_still_work():
    assert "timeout" in focus_note_from_metrics({"SR": 0.0, "CR": 0.2, "TR": 0.6})
    assert "collision" in focus_note_from_metrics({"SR": 0.0, "CR": 0.6, "TR": 0.2})


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


def test_highway_build_reflection_no_hardcoded_advice():
    evo = StageIEvolver(
        ScriptedLLMClient([D5_SEED_FUNCTION]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.0, "rejected": False})(),
        validator=RewardValidator(),
        prompts=hwy_prompts,
    )
    pop = [
        _cand("a", score=0.2, metrics=_hwy_metrics(mean_speed=20.0, fitness=0.3)),
        _cand("b", score=0.1, metrics=_hwy_metrics(mean_speed=24.0, fitness=0.6)),
    ]
    pop[0].metadata["pareto_rank"] = 1
    pop[0].metadata["pareto_n"] = 2
    pop[0].metadata["pareto_feasible"] = True
    pop[1].metadata["pareto_rank"] = 0
    pop[1].metadata["pareto_n"] = 2
    pop[1].metadata["pareto_feasible"] = True
    pop[1].metadata["llm_diagnosis"] = "CR rising with speed shaping."
    pop[1].metadata["fitness"] = 0.6
    pop[1].metadata["last_metrics"]["fitness"] = 0.6
    note = evo._build_reflection(pop, generation=1)
    assert "strengthen goal progress" not in note
    assert "best_speed_among_feasible=" in note
    assert "Population(pareto_rank asc):" in note
    assert "PriorLLMDiagnoses:" in note
    assert "CR rising" in note
    assert "fitness" not in note.lower()


def test_highway_prompts_render_with_pareto_objective():
    text = hwy_prompts.format_d2_mutation(
        "def compute_reward(state, memory):\n    return 0.0\n",
        "evidence here",
    )
    assert "Before writing code" in text
    assert "Pareto dominance" in hwy_prompts.D1_SYSTEM_PROMPT
    assert "sigmoid" not in hwy_prompts.D1_SYSTEM_PROMPT.lower()
    assert "Do NOT reward lagging" not in hwy_prompts.D1_SYSTEM_PROMPT
    assert "Do NOT reward lagging" not in hwy_prompts.D4_EXTERNAL_KNOWLEDGE
    d3 = hwy_prompts.format_d3_refinement(
        "def compute_reward(state, memory):\n    return 0.0\n",
        last_score=0.5,
        feedback=evidence_block(_hwy_metrics()),
    )
    assert "Before writing code" in d3
    assert "fitness" not in evidence_block(_hwy_metrics()).lower()
    assert "Fitness=" not in d3


def test_format_proxy_highway_is_evidence():
    block = format_proxy_feedback_block(_hwy_metrics(), score1=0.3)
    assert "Focus:" not in block
    assert "fitness" not in block.lower()
    assert "SR=" in block
