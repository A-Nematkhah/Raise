"""Crash-safe checkpoints, resume id state and label ordering."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.reward_checks import make_highway_validator
from domains.highway.stage1 import make_score_fn
from raise_core.explore import RewardCandidate, StageIEvolver
from raise_core.json_io import atomic_write_json
from raise_core.raise_loop.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.parallel_label import label_candidates_parallel
from raise_core.raise_loop.runner import (
    ClosedLoopRunner,
    _load_ckpt,
    _resume_known_ids,
)


def _cand(cid: str, **md) -> RewardCandidate:
    fn = make_highway_validator().validate_code(D5_SEED_FUNCTION)
    return RewardCandidate(
        candidate_id=cid,
        code=D5_SEED_FUNCTION,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=0.5,
        metadata=dict(md),
    )


def test_atomic_write_keeps_previous_file_on_failure(tmp_path):
    path = str(tmp_path / "state.json")
    atomic_write_json(path, {"v": 1})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"v": object()})
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == {"v": 1}
    assert os.listdir(tmp_path) == ["state.json"]


def _ckpt_payload(epoch: int) -> dict:
    return build_checkpoint(status="running", phase="init", epoch=epoch, next_epoch=epoch)


def test_load_ckpt_falls_back_to_root_copy(tmp_path):
    out = str(tmp_path / "run")
    save_checkpoint(out, _ckpt_payload(3))
    nested = os.path.join(out, "closed_loop")
    os.makedirs(nested)
    with open(os.path.join(nested, "checkpoint.json"), "w", encoding="utf-8") as fh:
        fh.write('{"status": "runn')
    data = _load_ckpt(out)
    assert data is not None and data["epoch"] == 3


def test_load_ckpt_refuses_when_every_copy_is_corrupt(tmp_path):
    out = str(tmp_path / "run")
    for d in (os.path.join(out, "closed_loop"), out):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "checkpoint.json"), "w", encoding="utf-8") as fh:
            fh.write("{")
    with pytest.raises(RuntimeError, match="unreadable"):
        _load_ckpt(out)


def test_restore_id_counter_skips_issued_ids():
    score_fn, _ds = make_score_fn(mode="smoke")
    evolver = StageIEvolver(None, score_fn=score_fn, validator=make_highway_validator())
    assert evolver.restore_id_counter(None, existing_ids=["ini_0003", "mut_0011_v2", "x"]) == 12
    assert evolver.restore_id_counter(20) == 20
    assert evolver._next_id("cro") == "cro_0020"


def test_resume_known_ids_reads_checkpoint_and_train_dirs(tmp_path):
    stage2 = tmp_path / "stage2_train"
    (stage2 / "r00_mut_0042").mkdir(parents=True)
    (stage2 / "notes").mkdir()
    ckpt = {
        "population": [{"candidate_id": "ini_0001"}],
        "global_best": {"candidate_id": "cro_0007"},
        "history": [{"ranking": ["ran_0009"], "labeled_ids": ["ini_0001"]}],
    }
    ids = set(_resume_known_ids(ckpt, str(stage2)))
    assert {"ini_0001", "cro_0007", "ran_0009", "mut_0042"} <= ids
    assert "notes" not in ids


def test_parallel_label_emits_in_order_and_survives_crash():
    cands = [_cand(f"c{i}") for i in range(4)]
    delays = {"c0": 0.15, "c1": 0.0, "c2": 0.05, "c3": 0.0}

    def _label(cand, _round):
        time.sleep(delays[cand.candidate_id])
        if cand.candidate_id == "c2":
            raise RuntimeError("boom")
        return {"status": "ok"}

    seen = []
    main = threading.current_thread()

    def _on_done(cand, result, i):
        assert threading.current_thread() is main
        seen.append((i, result["status"]))

    n_ok, n_fail = label_candidates_parallel(
        cands,
        already_done={"c3"},
        workers=3,
        label_one=_label,
        on_done=_on_done,
        round_index_fn=lambda i: i,
    )
    assert seen == [(0, "ok"), (1, "ok"), (2, "failed")]
    assert (n_ok, n_fail) == (2, 1)


def test_sequential_label_records_crash_as_failed():
    def _label(cand, _round):
        raise ValueError("bad")

    seen = []
    n_ok, n_fail = label_candidates_parallel(
        [_cand("c0")],
        already_done=set(),
        workers=1,
        label_one=_label,
        on_done=lambda c, r, i: seen.append(r),
        round_index_fn=lambda i: i,
    )
    assert (n_ok, n_fail) == (0, 1)
    assert "ValueError" in seen[0]["reason"]


def test_closed_loop_resume_does_not_reissue_ids(tmp_path):
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(
        population_size=4,
        generations=2,
        stage2_train_steps=32,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        device="cpu",
        resume=True,
    )
    ClosedLoopRunner(cfg).run()
    ckpt1 = load_checkpoint(out)
    issued = ckpt1["extra"]["id_counter"]
    # Gen0 issues 4 ids, Gen1 breeds 3 (the elite is carried over).
    assert issued >= 7

    # Pretend the run crashed before epoch 0 bred, from a checkpoint written
    # before id_counter was persisted (forces the derive-from-ids fallback).
    ckpt1.update(status="running", phase="init", epoch=0, next_epoch=0, history=[])
    ckpt1["extra"].pop("id_counter")
    for d in (os.path.join(out, "closed_loop"), out):
        save_checkpoint(d, ckpt1)
    resumed_ids = {r["candidate_id"] for r in ckpt1["population"]}

    second = ClosedLoopRunner(cfg).run()
    assert second.resumed is True
    bred = [c for c in second.population if c.candidate_id not in resumed_ids]
    assert bred
    for cand in bred:
        serial = int(cand.candidate_id.split("_")[1])
        assert serial >= issued, cand.candidate_id
