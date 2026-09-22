"""Active learning v1 unit tests (PLAN.md §9)."""

from __future__ import annotations

import math
import os

import pytest

from crowd_nav.reward_search.active_learning.acquire import execute_query
from crowd_nav.reward_search.active_learning.loop import run_active_learning_step
from crowd_nav.reward_search.active_learning.query import QueryItem, score_queries
from crowd_nav.reward_search.active_learning.queue import (
    dequeue_batch,
    enqueue,
    mark_done,
    pending_items,
)
from crowd_nav.reward_search.explore import RewardCandidate
from crowd_nav.reward_search.sandbox.validator import RewardValidator
from crowd_nav.reward_search.surrogate.model import SurrogatePrediction


def _cand(cid: str, pot: float = 2.0) -> RewardCandidate:
    src = (
        "def compute_reward(state, memory):\n"
        "    success_reward = 10.0\n"
        "    collision_penalty = -20.0\n"
        f"    pot_factor = {pot:.4f}\n"
        "    if state.reaching_goal:\n"
        "        return float(success_reward)\n"
        "    if state.collision:\n"
        "        return float(collision_penalty)\n"
        "    dist = ((state.robot.px - state.robot.gx) ** 2 + "
        "(state.robot.py - state.robot.gy) ** 2) ** 0.5\n"
        "    prev = memory.get('prev_dist')\n"
        "    if prev is None:\n"
        "        memory['prev_dist'] = dist\n"
        "        return float(0.0)\n"
        "    progress = float(prev) - dist\n"
        "    memory['prev_dist'] = dist\n"
        "    return float(pot_factor * progress)\n"
    )
    reward_fn, err = RewardValidator().try_validate(src)
    return RewardCandidate(
        candidate_id=cid,
        code=src,
        reward_fn=reward_fn,
        valid=reward_fn is not None,
        validation_error=err,
        score=0.5,
        origin="test",
    )


def test_score_queries_orders_by_uncertainty():
    cands = [_cand("a", 2.0), _cand("b", 2.1), _cand("c", 2.2)]
    assert all(c.valid for c in cands)
    preds = [
        SurrogatePrediction(y_hat={"SR": 0.5, "CR": 0.1, "TR": 0.1}, uncertainty=0.1),
        SurrogatePrediction(y_hat={"SR": 0.5, "CR": 0.1, "TR": 0.1}, uncertainty=0.9),
        SurrogatePrediction(y_hat={"SR": 0.5, "CR": 0.1, "TR": 0.1}, uncertainty=0.4),
    ]
    items = score_queries(
        cands,
        surrogate_preds=preds,
        score1_results=[0.5, 0.5, 0.5],
        top_k=3,
        weights={"u": 1.0, "disagree": 0.0, "borderline": 0.0, "diversity": 0.0},
        max_stage1_scenarios=0,
    )
    assert len(items) == 3
    assert items[0].payload["candidate_id"] == "b"
    assert items[0].priority >= items[1].priority >= items[2].priority


def test_queue_roundtrip(tmp_path):
    root = str(tmp_path / "al")
    items = [
        QueryItem(kind="stage2_label", priority=0.2, payload={"candidate_id": "x"}),
        QueryItem(kind="stage2_label", priority=0.9, payload={"candidate_id": "y"}),
        QueryItem(kind="stage1_scenario", priority=0.5, payload={"scenario_ids": ["s0"]}),
    ]
    enqueue(root, items)
    assert len(pending_items(root)) == 3
    taken = dequeue_batch(root, limit=2)
    assert len(taken) == 2
    assert taken[0].payload["candidate_id"] == "y"
    assert len(pending_items(root)) == 1
    mark_done(root, taken[0], result={"status": "ok"})
    assert os.path.isfile(os.path.join(root, "done.jsonl"))


def test_loop_noop_without_model(tmp_path):
    summary = run_active_learning_step(
        surrogate_model_dir=str(tmp_path / "missing"),
        queue_root=str(tmp_path / "al"),
        max_queries=2,
    )
    assert summary["status"] == "error"
    assert "Surrogate model required" in summary["message"]


def test_execute_stage2_label_stub(tmp_path):
    pytest.importorskip("sklearn")
    from crowd_nav.reward_search.surrogate.bootstrap import run_bootstrap

    out_dir = str(tmp_path / "surr_data")
    model_dir = str(tmp_path / "surr_model")
    queue_root = str(tmp_path / "al")
    boot = run_bootstrap(
        out_dir=out_dir,
        model_dir=model_dir,
        n_candidates=3,
        use_stub=True,
        force=True,
        seed=7,
        llm_provider="seed",
    )
    assert boot["status"] == "ok"

    cand = _cand("al_new", 3.5)
    assert cand.valid
    # Fresh code so example_id is new
    item = QueryItem(
        kind="stage2_label",
        priority=1.0,
        payload={
            "candidate_id": cand.candidate_id,
            "code": cand.code,
            "behavior_fingerprint": [0.1, 0.2],
        },
    )
    result = execute_query(
        item,
        config={
            "use_stub": True,
            "surrogate_dataset": out_dir,
            "surrogate_model_dir": model_dir,
            "queue_root": queue_root,
            "seed": 7,
        },
    )
    assert result["status"] in ("ok", "failed")
    assert result.get("example_id")
    assert os.path.isfile(os.path.join(out_dir, "features.jsonl"))

    # Full loop with force_refit
    summary = run_active_learning_step(
        surrogate_model_dir=model_dir,
        queue_root=queue_root,
        max_queries=2,
        refit_every=1,
        force_refit=True,
        candidates=[cand, _cand("al_new2", 3.7)],
        surrogate_dataset=out_dir,
        use_stub=True,
        seed=7,
    )
    assert summary["status"] == "ok"
    assert summary["n_executed"] >= 1
    assert summary["refit"] is True
