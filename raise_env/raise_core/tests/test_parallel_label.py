"""Tests for parallel Stage II labeling helpers and warm-start stamp."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from raise_core.raise_loop.parallel_label import (
    attach_warm_start_checkpoints,
    classify_label_failure,
    label_candidates_parallel,
)
from raise_core.sandbox.errors import RewardSandboxError


def test_attach_warm_start_from_parent():
    parent = SimpleNamespace(
        candidate_id="p1",
        parent_ids=(),
        metadata={"checkpoint_path": __file__},  # existing file
    )
    child = SimpleNamespace(
        candidate_id="c1",
        parent_ids=("p1",),
        metadata={},
    )
    attach_warm_start_checkpoints([child], [parent, child])
    assert child.metadata.get("warm_start_checkpoint") == Path(__file__).as_posix()



def test_label_candidates_parallel_serial_fallback():
    done = set()
    calls = []

    def label_one(cand, round_index):
        calls.append((cand.candidate_id, round_index))
        return {"status": "ok", "example_id": cand.candidate_id}

    def on_done(cand, result, i):
        done.add(cand.candidate_id)

    cands = [
        SimpleNamespace(candidate_id="a", reward_fn=object()),
        SimpleNamespace(candidate_id="b", reward_fn=object()),
    ]
    stats = label_candidates_parallel(
        cands,
        already_done=set(),
        workers=1,
        label_one=label_one,
        on_done=on_done,
        round_index_fn=lambda i: 100 + i,
    )
    assert stats.n_ok == 2 and stats.n_fail == 0
    assert done == {"a", "b"}
    assert [c[0] for c in calls] == ["a", "b"]


def test_label_candidates_parallel_two_workers():
    done = set()

    def label_one(cand, round_index):
        return {"status": "ok", "example_id": cand.candidate_id}

    def on_done(cand, result, i):
        done.add(cand.candidate_id)

    cands = [
        SimpleNamespace(candidate_id=f"c{i}", reward_fn=object()) for i in range(4)
    ]
    stats = label_candidates_parallel(
        cands,
        already_done=set(),
        workers=2,
        label_one=label_one,
        on_done=on_done,
        round_index_fn=lambda i: i,
    )
    assert stats.n_ok == 4 and stats.n_fail == 0
    assert done == {f"c{i}" for i in range(4)}


def test_classify_reward_vs_infra():
    assert classify_label_failure(RewardSandboxError("bad")) == "reward"
    assert classify_label_failure(FileNotFoundError("missing.zip")) == "infra"
    assert classify_label_failure(MemoryError("oom")) == "infra"
    assert classify_label_failure(RuntimeError("CUDA out of memory")) == "infra"
    assert classify_label_failure(RuntimeError("something else")) == "unknown"


def test_infra_penalize_returns_failed():
    def label_one(cand, round_index):
        raise OSError("disk full")

    results = []

    def on_done(cand, result, i):
        results.append(result)

    cands = [SimpleNamespace(candidate_id="a", reward_fn=object())]
    stats = label_candidates_parallel(
        cands,
        already_done=set(),
        workers=1,
        label_one=label_one,
        on_done=on_done,
        round_index_fn=lambda i: i,
        infra_error_policy="penalize",
    )
    assert stats.n_fail == 1 and stats.n_infra_fail == 1
    assert results[0]["failure_kind"] == "infra"


def test_infra_abort_raises_after_on_done():
    from raise_core.raise_loop.parallel_label import LabelAbortError

    def label_one(cand, round_index):
        raise OSError("disk full")

    seen = []
    cands = [SimpleNamespace(candidate_id="a", reward_fn=object())]
    with pytest.raises(LabelAbortError):
        label_candidates_parallel(
            cands,
            already_done=set(),
            workers=1,
            label_one=label_one,
            on_done=lambda c, r, i: seen.append(r["status"]),
            round_index_fn=lambda i: i,
            infra_error_policy="abort",
        )
    assert seen == ["aborted_infra"]
