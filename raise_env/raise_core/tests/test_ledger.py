"""Closed-loop ledger rows (highway debug-first logging)."""

from __future__ import annotations

import json
import os

import pytest

from raise_core.explore import RewardCandidate
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.ledger import (
    GENERATIONS_FILE,
    LEDGER_FILE,
    candidate_row,
    label_reason,
    write_generation_ledger,
)
from raise_core.raise_loop.runner import ClosedLoopRunner


def _cand(cid: str, *, origin: str = "mutation", **md) -> RewardCandidate:
    return RewardCandidate(
        candidate_id=cid,
        code=f"def compute_reward(state, memory):\n    return 0.0  # {cid}\n",
        score=0.5,
        valid=True,
        origin=origin,
        parent_ids=("ini_0000",),
        metadata=dict(md),
    )


def test_label_reason_soft_hard_al_and_dropped():
    soft = {"soft": True, "reason": "gen0"}
    hard = {"soft": False, "kept_ids": ["a", "b"], "dropped_ids": ["c"]}
    al = {"enabled": True, "pick_ids": ["b", "d"]}
    assert label_reason("a", selected_ids=["a"], gate_report=soft, al_report={}) == (
        "soft_gate:gen0"
    )
    assert label_reason("a", selected_ids=["a", "b", "d"], gate_report=hard, al_report=al) == (
        "gate_kept"
    )
    assert label_reason("b", selected_ids=["a", "b", "d"], gate_report=hard, al_report=al) == (
        "gate_kept+al_pick"
    )
    assert label_reason("d", selected_ids=["a", "b", "d"], gate_report=hard, al_report=al) == (
        "al_pick"
    )
    assert label_reason("c", selected_ids=["a", "b"], gate_report=hard, al_report=al) == (
        "gate_dropped"
    )
    assert label_reason("e", selected_ids=["a", "b", "f"], gate_report=hard, al_report=al) == (
        "not_selected"
    )
    assert label_reason("f", selected_ids=["a", "b", "f"], gate_report=hard, al_report=al) == (
        "top_up"
    )


def test_candidate_row_reads_metrics_seeds_and_fallback_flag():
    cand = _cand(
        "mut_0001",
        origin="mutation_fallback",
        llm_fallback_clone=True,
        last_metrics={"SR": 0.9, "CR": 0.1, "mean_speed": 21.5, "mean_progress": 800.0,
                      "warm_start": 0.0},
        train_seed=1425,
        eval_seed_base=51428,
        eval_episodes=20,
        train_steps=12000,
        score1_scenario_scores={"spearman": 0.5},
    )
    row = candidate_row(
        cand, generation=1, evolve_index=0, elite_id="mut_0001",
        reason="soft_gate:gen0", label_status="ok",
    )
    assert row["is_fallback_clone"] is True
    assert row["prompt_type"] == "mutation_fallback_clone"
    assert row["SR"] == pytest.approx(0.9)
    assert row["progress"] == pytest.approx(800.0)
    assert row["warm_start"] is False
    assert row["eval_seed_base"] == 51428
    assert row["is_elite"] is True
    assert row["selected_for_stage2"] is True
    assert row["score1_components"] == {"spearman": 0.5}


def test_write_generation_ledger_appends_rows_and_summary(tmp_path):
    a = _cand("a", last_metrics={"SR": 1.0, "CR": 0.0, "mean_progress": 900.0},
              eval_seed_base=100)
    b = _cand("b", origin="seed_fallback", last_metrics={"SR": 0.0, "CR": 1.0},
              eval_seed_base=200)
    summary = write_generation_ledger(
        str(tmp_path),
        [a, b],
        generation=0,
        selected_ids=["a", "b"],
        label_statuses={"a": "ok", "b": "failed"},
        gate_report={"soft": True, "reason": "gen0"},
        al_report={"enabled": False},
        d3_report={"n_attempts": 0, "n_accepted": 0, "attempts": []},
        buckets={"crossover": 2, "mutation": 3, "random": 0, "elite": 1},
        elite_id="a",
    )
    assert summary["n_population"] == 2
    assert summary["n_fallback_clones"] == 1
    assert summary["n_trained_ok"] == 1
    assert summary["n_trained_failed"] == 1
    assert summary["eval_seed_bases"] == [100, 200]
    assert summary["best_SR"] == pytest.approx(1.0)
    root = os.path.join(str(tmp_path), "closed_loop")
    with open(os.path.join(root, LEDGER_FILE), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert [r["candidate_id"] for r in rows] == ["a", "b"]
    with open(os.path.join(root, GENERATIONS_FILE), encoding="utf-8") as fh:
        gens = [json.loads(line) for line in fh if line.strip()]
    assert len(gens) == 1 and gens[0]["elite_id"] == "a"


def test_highway_stub_closed_loop_writes_ledger(tmp_path):
    pytest.importorskip("sklearn")
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(
        domain="highway",
        population_size=4,
        generations=2,
        stage2_train_steps=32,
        k2_unit="env_steps",
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        surrogate_dataset=str(tmp_path / "surr_data"),
        surrogate_model_dir=str(tmp_path / "surr_model"),
        al_root=str(tmp_path / "al"),
        device="cpu",
        num_processes=1,
    )
    cfg.apply_fast_profile()
    cfg.output_dir = out
    cfg.generations = 2
    ClosedLoopRunner(cfg).run()
    root = os.path.join(out, "closed_loop")
    with open(os.path.join(root, LEDGER_FILE), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    with open(os.path.join(root, GENERATIONS_FILE), encoding="utf-8") as fh:
        gens = [json.loads(line) for line in fh if line.strip()]
    assert len(gens) == 2
    assert {r["generation"] for r in rows} == {0, 1}
    for g in gens:
        assert g["n_population"] == len([r for r in rows if r["generation"] == g["generation"]])
        assert "buckets" in g and "gate" in g and "d3" in g
    assert all(r["label_reason"] for r in rows)
