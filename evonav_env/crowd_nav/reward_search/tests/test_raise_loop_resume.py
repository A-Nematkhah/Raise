"""Closed-loop checkpoint / resume tests."""

from __future__ import annotations

import os

import pytest

from crowd_nav.reward_search.closed_loop.checkpoint import (
    build_checkpoint,
    deserialize_population,
    load_checkpoint,
    save_checkpoint,
)
from crowd_nav.reward_search.closed_loop.config import ClosedLoopConfig
from crowd_nav.reward_search.closed_loop.runner import ClosedLoopRunner
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
from crowd_nav.reward_search.sandbox.validator import RewardValidator


def _cand(cid: str, pot: float) -> RewardCandidate:
    code = D5_SEED_FUNCTION.replace("pot_factor = 2.0", f"pot_factor = {pot}")
    fn = RewardValidator().validate_code(code)
    return RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=0.5,
    )


def test_checkpoint_roundtrip(tmp_path):
    out = str(tmp_path / "run")
    pop = [_cand("a", 1.1), _cand("b", 1.2)]
    payload = build_checkpoint(
        status="running",
        phase="labeling",
        epoch=1,
        next_epoch=1,
        population=pop,
        ranked=pop,
        to_label=pop[:1],
        labeled_ids_this_epoch=["a"],
        global_best=pop[0],
        reflection="hi",
        labels_since_refit=2,
        n_labeled=5,
        history=[{"epoch": 0}],
        config={"generations": 2},
    )
    path = save_checkpoint(out, payload)
    assert os.path.isfile(path)
    assert os.path.isfile(os.path.join(out, "RESUME.json"))
    loaded = load_checkpoint(out)
    assert loaded is not None
    assert loaded["phase"] == "labeling"
    assert loaded["labeled_ids_this_epoch"] == ["a"]
    restored = deserialize_population(loaded["population"])
    assert len(restored) == 2
    assert restored[0].candidate_id == "a"
    assert "pot_factor" in restored[0].code


def test_closed_loop_resume_skips_finished_epochs(tmp_path):
    pytest.importorskip("sklearn")
    out = str(tmp_path / "run")
    data = str(tmp_path / "surr_data")
    model = str(tmp_path / "surr_model")
    os.makedirs(data, exist_ok=True)
    os.makedirs(model, exist_ok=True)

    cfg = ClosedLoopConfig(
        population_size=2,
        generations=2,
        min_labels_for_gate=4,
        refit_every_new_labels=2,
        stage2_train_steps=8,
        k2_unit="env_steps",
        al_enabled=False,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        surrogate_dataset=data,
        surrogate_model_dir=model,
        isolate_run_artifacts=False,
        resume=True,
        device="cpu",
        num_processes=1,
        n_crossover=1,
        n_mutation=0,
        n_random=1,
    )

    # First run fully (generations=2 → Gen0 + 2 evolutions = 3 epochs).
    r1 = ClosedLoopRunner(cfg).run()
    assert r1.n_labeled_total >= 1
    assert len(r1.history) == 3
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"

    # Second run should short-circuit on completed checkpoint.
    r2 = ClosedLoopRunner(cfg).run()
    assert r2.resumed is True
    assert len(r2.population) == len(r1.population)


def test_resume_mid_labeling_continues(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    out = str(tmp_path / "run")
    data = str(tmp_path / "surr_data")
    model = str(tmp_path / "surr_model")
    os.makedirs(data, exist_ok=True)
    os.makedirs(model, exist_ok=True)

    pop = [_cand("c0", 1.0), _cand("c1", 1.5)]
    save_checkpoint(
        out,
        build_checkpoint(
            status="running",
            phase="labeling",
            epoch=0,
            next_epoch=0,
            population=pop,
            ranked=pop,
            to_label=pop,
            labeled_ids_this_epoch=[pop[0].candidate_id],
            global_best=pop[0],
            reflection="",
            labels_since_refit=0,
            n_labeled=0,
            history=[],
            config={},
            extra={
                "surrogate_model_dir": model,
                "surrogate_dataset": data,
                "al_root": str(tmp_path / "al"),
            },
        ),
    )

    labeled: list[str] = []

    def _fake_label(cand, **kwargs):
        labeled.append(cand.candidate_id)
        # Pretend success without touching heavy trainers.
        eid = f"ex_{cand.candidate_id}"
        known = kwargs.get("known_ids")
        if known is not None:
            known.add(eid)
        # Write minimal jsonl so dataset count moves.
        from crowd_nav.reward_search.surrogate.dataset_io import append_example

        append_example(
            kwargs["out_dir"],
            features={"example_id": eid, "code_len": 1, "score1": 0.1},
            labels={
                "example_id": eid,
                "SR": 0.5,
                "CR": 0.1,
                "TR": 0.1,
                "ok": True,
                "k2_unit": "env_steps",
            },
            example_id=eid,
        )
        return {"status": "ok", "example_id": eid}

    monkeypatch.setattr(
        "crowd_nav.reward_search.closed_loop.runner.label_and_append_candidate",
        _fake_label,
    )

    cfg = ClosedLoopConfig(
        population_size=2,
        generations=1,
        min_labels_for_gate=99,
        al_enabled=False,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        surrogate_dataset=data,
        surrogate_model_dir=model,
        isolate_run_artifacts=False,
        resume=True,
        device="cpu",
        num_processes=1,
        n_crossover=1,
        n_mutation=0,
        n_random=1,
        stage2_train_steps=8,
        k2_unit="env_steps",
    )
    # Avoid Gen0 re-init / Score1: resume labeling path uses saved ranked.
    result = ClosedLoopRunner(cfg).run()

    # c0 already labeled in checkpoint → only c1 should be newly labeled.
    assert pop[0].candidate_id not in labeled
    assert pop[1].candidate_id in labeled
    assert result.resumed is True
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"
