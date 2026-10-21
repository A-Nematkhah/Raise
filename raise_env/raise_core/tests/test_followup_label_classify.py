"""classify_label_failure kinds + abort checkpoint-before-raise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raise_core.raise_loop.parallel_label import (
    LabelAbortError,
    classify_label_failure,
    label_candidates_parallel,
)
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.runtime import SandboxedReward, compile_compute_reward


def test_classify_unknown_vs_typed():
    assert classify_label_failure(RewardSandboxError("bad")) == "reward"
    assert classify_label_failure(FileNotFoundError("x")) == "infra"
    assert classify_label_failure(MemoryError("oom")) == "infra"
    assert classify_label_failure(RuntimeError("CUDA out of memory")) == "infra"
    # Unknown: not matched by message alone when type is plain Exception.
    assert classify_label_failure(ValueError("cuda out of memory")) == "unknown"
    assert classify_label_failure(RuntimeError("something weird")) == "unknown"


def test_memory_error_inside_sandboxed_compute_is_reward():
    code = (
        "def compute_reward(state, memory):\n"
        "    raise MemoryError('reward blew up')\n"
    )
    cfg = SandboxConfig(allowed_attributes=("progress",))
    # Bypass smoke; compile directly for the unit test.
    fn = compile_compute_reward(code, cfg)
    reward = SandboxedReward(fn, cfg, source_code=code)
    with pytest.raises(RewardSandboxError, match="MemoryError"):
        reward.compute(SimpleNamespace(progress=0.0))
    # Classification of the wrapped error is reward.
    try:
        reward.compute(SimpleNamespace(progress=0.0))
    except RewardSandboxError as exc:
        assert classify_label_failure(exc) == "reward"


def test_unknown_retry_then_abort_writes_checkpoint_before_raise():
    calls = {"n": 0}
    ckpts: list = []

    def label_one(cand, round_index):
        calls["n"] += 1
        raise ValueError("totally unexpected")

    def on_done(cand, result, i):
        ckpts.append(dict(result))

    cands = [SimpleNamespace(candidate_id="c0", reward_fn=object())]
    with pytest.raises(LabelAbortError, match="unknown error"):
        label_candidates_parallel(
            cands,
            already_done=set(),
            workers=1,
            label_one=label_one,
            on_done=on_done,
            round_index_fn=lambda i: i,
            infra_error_policy="retry_then_abort",
        )
    assert calls["n"] == 2  # initial + retry
    assert len(ckpts) == 1
    assert ckpts[0]["status"] == "aborted_infra"
    assert ckpts[0]["failure_kind"] == "unknown"
    assert "Traceback" in (ckpts[0].get("traceback") or "")


def test_infra_penalize_continues():
    def label_one(cand, round_index):
        raise OSError("disk full")

    results = []
    stats = label_candidates_parallel(
        [SimpleNamespace(candidate_id="a", reward_fn=object())],
        already_done=set(),
        workers=1,
        label_one=label_one,
        on_done=lambda c, r, i: results.append(r),
        round_index_fn=lambda i: i,
        infra_error_policy="penalize",
    )
    assert stats.n_fail == 1 and stats.n_infra_fail == 1
    assert results[0]["status"] == "failed"
    assert results[0]["failure_kind"] == "infra"
