"""Closed-loop checkpoint / resume tests."""

from __future__ import annotations

import os

import pytest

from raise_core.raise_loop.checkpoint import (
    build_checkpoint,
    deserialize_population,
    load_checkpoint,
    save_checkpoint,
)
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core.explore import RewardCandidate
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from raise_core.sandbox.validator import RewardValidator


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

    r1 = ClosedLoopRunner(cfg).run()
    assert r1.n_labeled_total >= 1
    # generations=G → G scored epochs (Gen0 is epoch 0), same as test_raise_loop.py.
    assert len(r1.history) == 2
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"

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
        eid = f"ex_{cand.candidate_id}"
        known = kwargs.get("known_ids")
        if known is not None:
            known.add(eid)
        from raise_core.surrogate.dataset_io import append_example

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
        "raise_core.raise_loop.runner.label_and_append_candidate",
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
    result = ClosedLoopRunner(cfg).run()

    assert pop[0].candidate_id not in labeled
    assert pop[1].candidate_id in labeled
    assert result.resumed is True
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"

def test_highway_deserialize_uses_domain_validator():
    """Resume must not re-validate highway code against CrowdNav RewardState."""
    from domains.highway.prompts import D5_SEED_FUNCTION
    from raise_core.domains import load_domain, make_validator_for_domain
    from raise_core.raise_loop.checkpoint import deserialize_population
    from domains.crowdnav.reporting import load_candidate_dict

    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    rows = [
        {
            "candidate_id": "hw0",
            "code": D5_SEED_FUNCTION,
            "score": 0.5,
            "valid": True,
            "origin": "test",
            "parent_ids": [],
            "metadata": {},
        }
    ]
    bad = load_candidate_dict(rows[0])
    assert bad.reward_fn is None

    good = deserialize_population(rows, validator=v)
    assert len(good) == 1
    assert good[0].reward_fn is not None
