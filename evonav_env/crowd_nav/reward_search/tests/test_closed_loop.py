"""Closed-loop multi-fidelity unit tests (PLAN §8)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from crowd_nav.reward_search.closed_loop.config import ClosedLoopConfig
from crowd_nav.reward_search.closed_loop.epoch import select_to_label, unique_by_code
from crowd_nav.reward_search.closed_loop.logging_io import read_epochs
from crowd_nav.reward_search.closed_loop.runner import ClosedLoopRunner
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
from crowd_nav.reward_search.sandbox.validator import RewardValidator


def _cand(cid: str, code: str | None = None) -> RewardCandidate:
    src = code or (
        D5_SEED_FUNCTION.replace("pot_factor = 2.0", f"pot_factor = {1.0 + hash(cid) % 50 * 0.01}")
    )
    fn = RewardValidator().validate_code(src)
    return RewardCandidate(
        candidate_id=cid,
        code=src,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=0.5,
    )


def test_select_gen0_never_gates():
    pop = [_cand("a"), _cand("b"), _cand("c")]
    preds = [
        {"candidate_id": "a", "quality": 0.9, "uncertainty": 0.01},
        {"candidate_id": "b", "quality": -1.0, "uncertainty": 0.01},
        {"candidate_id": "c", "quality": 0.1, "uncertainty": 0.01},
    ]
    to_label, gate, al = select_to_label(
        pop,
        epoch=0,
        n_labeled=100,
        min_labels_for_gate=24,
        predictions=preds,
        model_ready=True,
        drop_fraction=0.5,
        max_uncertainty_to_drop=0.2,
        min_keep=1,
        min_stage2_per_gen=1,
        al_enabled=True,
        al_max_per_epoch=4,
        al_allow_stage1_requests=False,
        al_root="data/active_learning",
    )
    assert gate.get("soft") is True
    assert len(to_label) == 3
    assert al.get("enabled") is False


def test_soft_gate_until_min_labels():
    pop = [_cand("a"), _cand("b"), _cand("c"), _cand("d")]
    preds = [
        {"candidate_id": c.candidate_id, "quality": -1.0, "uncertainty": 0.01}
        for c in pop
    ]
    to_label, gate, _al = select_to_label(
        pop,
        epoch=1,
        n_labeled=10,
        min_labels_for_gate=24,
        predictions=preds,
        model_ready=True,
        drop_fraction=0.5,
        max_uncertainty_to_drop=0.2,
        min_keep=1,
        min_stage2_per_gen=1,
        al_enabled=False,
        al_max_per_epoch=2,
        al_allow_stage1_requests=False,
        al_root="data/active_learning",
    )
    assert gate.get("soft") is True
    assert len(to_label) == 4


def test_al_inside_epoch_hard_gate(monkeypatch):
    pop = [_cand(f"c{i}") for i in range(6)]
    # Make distinct codes
    pop = unique_by_code(pop)
    preds = []
    for i, c in enumerate(pop):
        preds.append(
            {
                "candidate_id": c.candidate_id,
                "quality": 0.8 if i < 3 else -0.5,
                "uncertainty": 0.05 if i >= 3 else 0.4,
                "y_hat": {"SR": 0.5, "CR": 0.1, "TR": 0.1},
            }
        )

    called = {"n": 0}

    def _fake_score_queries(*_a, **_k):
        called["n"] += 1
        from crowd_nav.reward_search.active_learning.query import QueryItem

        # Pick a dropped-like candidate
        return [
            QueryItem(
                kind="stage2_label",
                priority=1.0,
                payload={"candidate_id": pop[-1].candidate_id, "code": pop[-1].code},
            )
        ]

    monkeypatch.setattr(
        "crowd_nav.reward_search.closed_loop.epoch.score_queries",
        _fake_score_queries,
    )
    to_label, gate, al = select_to_label(
        pop,
        epoch=1,
        n_labeled=30,
        min_labels_for_gate=24,
        predictions=preds,
        model_ready=True,
        drop_fraction=0.5,
        max_uncertainty_to_drop=0.2,
        min_keep=2,
        min_stage2_per_gen=2,
        al_enabled=True,
        al_max_per_epoch=2,
        al_allow_stage1_requests=False,
        al_root="data/active_learning",
    )
    assert gate.get("enabled") is True
    assert gate.get("soft") is False
    assert called["n"] == 1
    assert al.get("enabled") is True
    assert pop[-1].candidate_id in {c.candidate_id for c in to_label}


def test_closed_loop_fast_two_epochs(tmp_path):
    pytest.importorskip("sklearn")
    out = str(tmp_path / "run")
    data = str(tmp_path / "surr_data")
    model = str(tmp_path / "surr_model")
    cfg = ClosedLoopConfig(
        population_size=4,
        generations=2,
        min_labels_for_gate=4,
        refit_every_new_labels=2,
        stage2_train_steps=32,
        k2_unit="env_steps",
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        surrogate_dataset=data,
        surrogate_model_dir=model,
        al_root=str(tmp_path / "al"),
        al_enabled=True,
        al_max_per_epoch=2,
        min_stage2_per_gen=2,
        device="cpu",
        num_processes=1,
    )
    cfg.apply_fast_profile()
    cfg.output_dir = out
    cfg.surrogate_dataset = data
    cfg.surrogate_model_dir = model
    cfg.generations = 2
    cfg.min_labels_for_gate = 4

    result = ClosedLoopRunner(cfg).run()
    assert result.n_labeled_total >= 1
    assert os.path.isfile(os.path.join(model, "model.joblib"))
    epochs = read_epochs(out)
    assert len(epochs) == 2
    assert epochs[0]["gate"].get("soft") is True
    # After Gen0 refit, Gen1 may hard-gate if enough labels
    assert epochs[1]["epoch"] == 1
    assert result.manifest.get("mode") == "closed_loop_v1"
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    assert os.path.isfile(report)
    text = open(report, encoding="utf-8").read()
    assert "epoch 0:" in text
    assert "Closed-loop run report" in text


def test_format_epoch_summary_hard_gate():
    from crowd_nav.reward_search.closed_loop.report import format_epoch_summary

    line = format_epoch_summary(
        {
            "epoch": 1,
            "n_population": 4,
            "best_id": "mut_0006",
            "best_score1": 0.606,
            "labeled_ids": ["mut_0006", "cro_0004"],
            "gate": {
                "soft": False,
                "n_kept": 3,
                "n_dropped": 1,
                "dropped_ids": ["cro_0004"],
            },
            "al": {"enabled": True, "n_al_stage2": 2, "n_stage1_requests": 0},
            "n_labeled_dataset": 8,
            "refit": True,
            "n_label_ok": 4,
            "n_label_failed": 0,
        }
    )
    assert "gate=hard" in line
    assert "al=on" in line
    assert "refit=yes" in line
    assert "mut_0006" in line
