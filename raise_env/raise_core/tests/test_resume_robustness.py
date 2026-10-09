"""Crash-safe checkpoints, resume id/elite state, dataset I/O and label ordering."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from domains.crowdnav.prompts import D5_SEED_FUNCTION
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
from raise_core.refine import ProxyMetrics, _untested_child_metadata
from raise_core.sandbox.validator import RewardValidator
from raise_core.scoring import make_smoke_score_fn
from raise_core.surrogate.dataset_io import (
    append_example,
    existing_example_ids,
    load_table,
)


def _cand(cid: str, **md) -> RewardCandidate:
    fn = RewardValidator().validate_code(D5_SEED_FUNCTION)
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
    evolver = StageIEvolver(None, score_fn=make_smoke_score_fn())
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


def test_dataset_nan_becomes_none(tmp_path):
    root = str(tmp_path)
    append_example(
        root,
        features={"score1": float("nan"), "behavior_fingerprint": [1.0, float("inf")]},
        labels={"SR": 1.0, "SD": float("nan")},
        example_id="e1",
    )
    feats, labs = load_table(root)
    assert feats[0]["score1"] is None
    assert feats[0]["behavior_fingerprint"] == [1.0, None]
    assert labs[0]["SD"] is None


def test_dataset_orphan_feature_row_is_relabeled(tmp_path):
    root = str(tmp_path)
    append_example(root, features={"a": 1}, labels={"SR": 1.0}, example_id="e1")
    with open(os.path.join(root, "features.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"example_id": "e2", "a": 2}) + "\n")
    assert existing_example_ids(root) == {"e1"}
    append_example(root, features={"a": 2}, labels={"SR": 0.5}, example_id="e2")
    feats, labs = load_table(root)
    assert [f["example_id"] for f in feats] == ["e1", "e2"]
    assert [l["SR"] for l in labs] == [1.0, 0.5]


def test_dataset_truncated_row_does_not_swallow_next_append(tmp_path):
    root = str(tmp_path)
    append_example(root, features={"a": 1}, labels={"SR": 1.0}, example_id="e1")
    for name in ("features.jsonl", "labels.jsonl"):
        with open(os.path.join(root, name), "a", encoding="utf-8") as fh:
            fh.write('{"example_id": "e2", "a"')
    assert existing_example_ids(root) == {"e1"}
    append_example(root, features={"a": 3}, labels={"SR": 0.2}, example_id="e3")
    feats, _labs = load_table(root)
    assert [f["example_id"] for f in feats] == ["e1", "e3"]


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


def test_highway_closed_loop_resume_does_not_reissue_ids(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("gymnasium")
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
        resume=True,
    )
    ClosedLoopRunner(cfg).run()
    ckpt1 = load_checkpoint(out)
    issued = ckpt1["extra"]["id_counter"]
    assert issued >= 8

    # Pretend the run crashed before epoch 0 bred, from a checkpoint written
    # before id_counter was persisted (forces the derive-from-ids fallback).
    ckpt1.update(status="running", phase="init", epoch=0, next_epoch=0, history=[])
    ckpt1["extra"].pop("id_counter")
    ckpt1["extra"].pop("highway_elite", None)
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


def test_untested_child_drops_parent_training_results():
    parent = _cand(
        "p", last_metrics={"SR": 1.0}, checkpoint_path="p.zip", lineage="keep"
    )
    md = _untested_child_metadata(parent, ProxyMetrics(sr=0.9))
    assert "last_metrics" not in md and "checkpoint_path" not in md
    assert md["lineage"] == "keep"
    assert md["parent_metrics"]["SR"] == pytest.approx(0.9)
    assert md["refine_kept_previous"] is False
