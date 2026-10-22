"""Closed-loop resume consistency: epoch log, failed labels, Score1 rejects, config drift."""

from __future__ import annotations

import json
import os

import pytest

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import RewardCandidate
from raise_core.raise_loop import runner as runner_mod
from raise_core.raise_loop.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.logging_io import append_epoch_record, read_epochs
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core.refine import StubPolicyTrainer
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.scoring import Score1Result


def _cand(cid: str, k: float, **md) -> RewardCandidate:
    code = f"def compute_reward(state, memory):\n    return float(state.progress) * {k}\n"
    return RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=make_crowdnav_validator().validate_code(code),
        valid=True,
        origin="test",
        score=0.5,
        metadata=dict(md),
    )


def _cfg(out: str, **kw) -> ClosedLoopConfig:
    base = dict(
        population_size=2,
        generations=1,
        stage2_train_steps=8,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        resume=True,
        device="cpu",
    )
    base.update(kw)
    return ClosedLoopConfig(**base)


def _labeling_ckpt(out: str, pop, done=()) -> None:
    save_checkpoint(
        out,
        build_checkpoint(
            status="running",
            phase="labeling",
            epoch=0,
            next_epoch=0,
            population=pop,
            to_label=pop,
            labeled_ids_this_epoch=list(done),
            history=[],
            config={},
        ),
    )


class _ScriptedTrainer(StubPolicyTrainer):
    """Stub trainer that raises the exception mapped to a candidate id."""

    def __init__(self, raises=None) -> None:
        self.raises = dict(raises or {})
        self.trained: list[str] = []

    def train_and_eval(self, candidate, *, round_index, config):
        cid = str(candidate.candidate_id)
        self.trained.append(cid)
        if cid in self.raises:
            raise self.raises[cid]
        return super().train_and_eval(candidate, round_index=round_index, config=config)


def test_append_epoch_record_replaces_rows_from_a_rerun_epoch(tmp_path):
    out = str(tmp_path / "run")
    for g in (0, 1):
        append_epoch_record(out, {"epoch": g, "tag": "first"})
    append_epoch_record(out, {"epoch": 1, "tag": "rerun"})
    assert [(r["epoch"], r["tag"]) for r in read_epochs(out)] == [(0, "first"), (1, "rerun")]
    # Re-running an earlier epoch drops everything logged after it.
    append_epoch_record(out, {"epoch": 0, "tag": "again"})
    assert [(r["epoch"], r["tag"]) for r in read_epochs(out)] == [(0, "again")]
    assert not [f for f in os.listdir(os.path.join(out, "closed_loop")) if ".tmp." in f]


def test_read_epochs_skips_torn_line(tmp_path):
    out = str(tmp_path / "run")
    append_epoch_record(out, {"epoch": 0})
    path = os.path.join(out, "closed_loop", "epochs.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"epoch": 1, "rank')
    assert [r["epoch"] for r in read_epochs(out)] == [0]
    append_epoch_record(out, {"epoch": 1})
    with open(path, encoding="utf-8") as fh:
        assert [json.loads(line)["epoch"] for line in fh] == [0, 1]


def test_rerun_epoch_after_crash_does_not_duplicate_epoch_row(tmp_path):
    out = str(tmp_path / "run")
    ClosedLoopRunner(_cfg(out, generations=2)).run()
    assert [r["epoch"] for r in read_epochs(out)] == [0, 1]

    # Crash after epoch 1 was logged but before its epoch_done checkpoint.
    ckpt = load_checkpoint(out)
    ckpt.update(status="running", phase="init", epoch=1, next_epoch=1, history=ckpt["history"][:1])
    for d in (os.path.join(out, "closed_loop"), out):
        save_checkpoint(d, ckpt)
    ClosedLoopRunner(_cfg(out, generations=2)).run()
    assert [r["epoch"] for r in read_epochs(out)] == [0, 1]


def test_failed_label_is_retried_on_resume(tmp_path):
    out = str(tmp_path / "run")
    pop = [_cand("c0", 1.0), _cand("c1", 1.5), _cand("c2", 2.0)]
    _labeling_ckpt(out, pop)

    # Reward-side failure (not infra/unknown): marked failed and retried on resume.
    # Unknown RuntimeError / infra abort under the default retry_then_abort policy.
    crashing = _ScriptedTrainer(
        raises={"c1": RewardSandboxError("diverged"), "c2": KeyboardInterrupt()}
    )
    with pytest.raises(KeyboardInterrupt):
        ClosedLoopRunner(_cfg(out), trainer=crashing).run()
    assert load_checkpoint(out)["labeled_ids_this_epoch"] == ["c0"]

    trainer = _ScriptedTrainer()
    ClosedLoopRunner(_cfg(out), trainer=trainer).run()
    assert sorted(trainer.trained) == ["c1", "c2"]
    (rec,) = read_epochs(out)
    assert sorted(rec["labeled_ids"]) == ["c0", "c1", "c2"]
    assert rec["failed_ids"] == []
    assert rec["resumed"] is True


def test_failed_label_is_reported_separately_and_has_no_inf(tmp_path):
    out = str(tmp_path / "run")
    _labeling_ckpt(out, [_cand("c0", 1.0), _cand("c1", 1.5)])
    trainer = _ScriptedTrainer(raises={"c1": RewardSandboxError("diverged")})
    ClosedLoopRunner(_cfg(out), trainer=trainer).run()

    (rec,) = read_epochs(out)
    assert rec["labeled_ids"] == ["c0"]
    assert rec["failed_ids"] == ["c1"]
    assert rec["n_label_failed"] == 1
    assert rec["label_metrics"]["c1"]["fitness"] is None
    with open(os.path.join(out, "closed_loop", "epochs.jsonl"), encoding="utf-8") as fh:
        assert "Infinity" not in fh.read()


def test_score1_rejected_candidate_is_not_trained(tmp_path):
    out = str(tmp_path / "run")
    save_checkpoint(
        out,
        build_checkpoint(
            status="running",
            phase="init",
            epoch=0,
            next_epoch=0,
            population=[_cand("good", 1.0), _cand("bad", 0.0)],
            history=[],
            config={},
        ),
    )

    def score_fn(_fn, *, candidate_id=None):
        bad = candidate_id == "bad"
        return Score1Result(
            score=0.0 if bad else 0.4,
            n_nan_frames=10 if bad else 0,
            n_frames=10,
            rejected=bad,
            reject_reason="score1_error" if bad else None,
        )

    trainer = _ScriptedTrainer()
    ClosedLoopRunner(_cfg(out), trainer=trainer, score_fn=score_fn).run()
    assert trainer.trained == ["good"]
    (rec,) = read_epochs(out)
    assert rec["skipped_score1_ids"] == ["bad"]
    assert rec["labeled_ids"] == ["good"]


def test_resume_aborts_on_config_drift_unless_allowed(tmp_path, monkeypatch):
    out = str(tmp_path / "run")
    ClosedLoopRunner(_cfg(out)).run()
    with pytest.raises(RuntimeError, match="config drift"):
        ClosedLoopRunner(_cfg(out, stage2_train_steps=16, seed=7)).run()

    warnings: list[str] = []
    monkeypatch.setattr(
        runner_mod.console, "warn", lambda msg, **_kw: warnings.append(str(msg))
    )
    ClosedLoopRunner(
        _cfg(out, stage2_train_steps=16, seed=7, allow_config_drift=True)
    ).run()
    drift = [w for w in warnings if "differ from the checkpoint" in w]
    assert len(drift) == 1
    assert "stage2_train_steps: 8 -> 16" in drift[0]
    assert "seed: 425 -> 7" in drift[0]

    # The resumed run saved its own settings: same settings again, no warning.
    warnings.clear()
    ClosedLoopRunner(
        _cfg(out, stage2_train_steps=16, seed=7, allow_config_drift=True)
    ).run()
    assert not [w for w in warnings if "differ from the checkpoint" in w]
