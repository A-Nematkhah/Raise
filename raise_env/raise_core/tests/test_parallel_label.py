"""Tests for highway parallel labeling helpers and warm-start stamp."""

from __future__ import annotations

from types import SimpleNamespace

from raise_core.raise_loop.parallel_label import (
    attach_warm_start_checkpoints,
    label_candidates_parallel,
)


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
    assert child.metadata.get("warm_start_checkpoint") == __file__


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
    n_ok, n_fail = label_candidates_parallel(
        cands,
        already_done=set(),
        workers=1,
        label_one=label_one,
        on_done=on_done,
        round_index_fn=lambda i: 100 + i,
    )
    assert n_ok == 2 and n_fail == 0
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
    n_ok, n_fail = label_candidates_parallel(
        cands,
        already_done=set(),
        workers=2,
        label_one=label_one,
        on_done=on_done,
        round_index_fn=lambda i: i,
    )
    assert n_ok == 4 and n_fail == 0
    assert done == {f"c{i}" for i in range(4)}
