"""Unit tests for selective Stage-II proxy feedback → LLM prompts."""

from __future__ import annotations

from raise_core.explore import RewardCandidate
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from raise_core.raise_loop.proxy_feedback import (
    attach_proxy_feedback,
    focus_note_from_metrics,
    format_proxy_feedback_block,
    nav_scalar,
    proxy_summary_for_reflection,
    select_for_in_loop_d3,
    should_attach_proxy_feedback,
)
from raise_core.sandbox.validator import RewardValidator


def _cand(cid: str, *, score: float = 0.5, metrics=None) -> RewardCandidate:
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


def test_nav_scalar_and_format():
    m = {"SR": 0.2, "CR": 0.1, "TR": 0.4}
    assert abs(nav_scalar(m) - (0.2 - 0.1 - 0.2)) < 1e-9
    block = format_proxy_feedback_block(m, score1=0.7)
    assert "SR=0.20" in block
    assert "Score1=0.700" in block
    assert "Focus:" in block


def test_focus_prioritizes_timeout_then_collision():
    assert "timeout" in focus_note_from_metrics({"SR": 0.0, "CR": 0.2, "TR": 0.6})
    assert "collision" in focus_note_from_metrics({"SR": 0.0, "CR": 0.6, "TR": 0.2})
    assert "near-zero success" in focus_note_from_metrics(
        {"SR": 0.05, "CR": 0.1, "TR": 0.1}
    )


def test_should_attach_on_bad_proxy():
    assert should_attach_proxy_feedback({"SR": 0.05, "CR": 0.1, "TR": 0.1})
    assert should_attach_proxy_feedback({"SR": 0.5, "CR": 0.55, "TR": 0.1})
    assert should_attach_proxy_feedback({"SR": 0.5, "CR": 0.1, "TR": 0.55})
    assert not should_attach_proxy_feedback({"SR": 0.4, "CR": 0.1, "TR": 0.1})


def test_should_attach_score1_proxy_mismatch():
    # High Score1 + weak scalar vs population → attach
    pop_s1 = [0.1, 0.2, 0.3, 0.8, 0.9]
    pop_sc = [0.5, 0.4, 0.3, 0.2, -0.1]
    assert should_attach_proxy_feedback(
        {"SR": 0.15, "CR": 0.1, "TR": 0.2},  # scalar ~ -0.05
        score1=0.9,
        population_score1=pop_s1,
        population_scalars=pop_sc,
    )
    # Mid Score1, mid scalar → no mismatch attach
    assert not should_attach_proxy_feedback(
        {"SR": 0.4, "CR": 0.1, "TR": 0.1},
        score1=0.3,
        population_score1=pop_s1,
        population_scalars=pop_sc,
    )


def test_attach_gates_on_epoch_and_labels():
    c = _cand("x", score=0.5)
    m = {"SR": 0.0, "CR": 0.0, "TR": 0.8}
    assert not attach_proxy_feedback(
        c, m, enabled=True, n_labeled_dataset=100, min_labels=16, epoch=0
    )
    assert "proxy_feedback" not in (c.metadata or {})
    assert (c.metadata or {}).get("last_metrics", {}).get("TR") == 0.8

    assert attach_proxy_feedback(
        c, m, enabled=True, n_labeled_dataset=100, min_labels=16, epoch=1
    )
    assert "ProxyRefine" in (c.metadata or {})["proxy_feedback"]

    # Good metrics clear stale block
    good = {"SR": 0.6, "CR": 0.05, "TR": 0.1}
    assert not attach_proxy_feedback(
        c, good, enabled=True, n_labeled_dataset=100, min_labels=16, epoch=2
    )
    assert "proxy_feedback" not in (c.metadata or {})


def test_attach_disabled():
    c = _cand("y")
    assert not attach_proxy_feedback(
        c,
        {"SR": 0.0, "CR": 0.9, "TR": 0.0},
        enabled=False,
        n_labeled_dataset=100,
        min_labels=16,
        epoch=2,
    )


def test_proxy_summary_and_d3_select():
    pop = [
        _cand("a", metrics={"SR": 0.5, "CR": 0.0, "TR": 0.0}),
        _cand("b", metrics={"SR": 0.0, "CR": 0.5, "TR": 0.5}),
        _cand("c", score=0.1),  # no metrics
    ]
    summary = proxy_summary_for_reflection(pop)
    assert summary is not None
    assert "best=a" in summary
    assert "mean_SR=" in summary

    worst = select_for_in_loop_d3(pop, max_n=1)
    assert len(worst) == 1
    assert worst[0].candidate_id == "b"
    assert select_for_in_loop_d3(pop, max_n=0) == []


def test_mutation_weakness_prefers_proxy_feedback():
    from raise_core.explore import StageIEvolver
    from raise_core.llm import ScriptedLLMClient

    parent = _cand("p0", score=0.4)
    parent.metadata["proxy_feedback"] = (
        "ProxyRefine: SR=0.00 CR=0.10 TR=0.70 scalar=-0.45\n"
        "Focus: high timeout — strengthen dense goal progress"
    )
    evo = StageIEvolver(
        ScriptedLLMClient([D5_SEED_FUNCTION]),
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.0, "rejected": False})(),
        validator=RewardValidator(),
    )
    captured = {}

    def _fake_format(code, weakness, **kwargs):
        captured["weakness"] = weakness
        return f"MUTATE:\n{code}"

    import raise_core.explore as explore_mod

    orig = explore_mod.format_d2_mutation
    explore_mod.format_d2_mutation = _fake_format
    try:
        evo._try_mutate(parent, "global note", phase="mutation", attempt=0)
    finally:
        explore_mod.format_d2_mutation = orig
    assert "ProxyRefine" in captured["weakness"]
    assert "higher priority" in captured["weakness"]
    assert "timeout" in captured["weakness"]
