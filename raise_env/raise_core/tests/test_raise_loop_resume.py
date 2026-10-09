"""Closed-loop checkpoint / resume tests."""

from __future__ import annotations

import os

from domains.highway.reward_checks import make_highway_validator
from raise_core.explore import RewardCandidate
from raise_core.raise_loop.checkpoint import (
    build_checkpoint,
    deserialize_population,
    load_checkpoint,
    save_checkpoint,
)
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core.refine import StubPolicyTrainer


def _cand(cid: str, k: float, **md) -> RewardCandidate:
    code = f"def compute_reward(state, memory):\n    return float(state.progress) * {k}\n"
    fn = make_highway_validator().validate_code(code)
    return RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=fn,
        valid=True,
        origin="test",
        score=0.5,
        metadata=dict(md),
    )


class _RecordingTrainer(StubPolicyTrainer):
    def __init__(self) -> None:
        self.trained: list[str] = []

    def train_and_eval(self, candidate, *, round_index, config):
        self.trained.append(str(candidate.candidate_id))
        return super().train_and_eval(candidate, round_index=round_index, config=config)


def _cfg(out: str, **kw) -> ClosedLoopConfig:
    base = dict(
        population_size=2,
        generations=2,
        stage2_train_steps=8,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        resume=True,
        device="cpu",
    )
    base.update(kw)
    return ClosedLoopConfig(**base)


def test_checkpoint_roundtrip(tmp_path):
    out = str(tmp_path / "run")
    pop = [_cand("a", 1.1), _cand("b", 1.2)]
    payload = build_checkpoint(
        status="running",
        phase="labeling",
        epoch=1,
        next_epoch=1,
        population=pop,
        to_label=pop[:1],
        labeled_ids_this_epoch=["a"],
        global_best=pop[0],
        reflection="hi",
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
    restored = deserialize_population(loaded["population"], validator=make_highway_validator())
    assert len(restored) == 2
    assert restored[0].candidate_id == "a"
    assert restored[0].reward_fn is not None
    assert "* 1.1" in restored[0].code


def test_closed_loop_resume_skips_finished_epochs(tmp_path):
    out = str(tmp_path / "run")
    r1 = ClosedLoopRunner(_cfg(out)).run()
    assert r1.n_labeled_total >= 1
    assert len(r1.history) == 2
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"

    trainer = _RecordingTrainer()
    r2 = ClosedLoopRunner(_cfg(out), trainer=trainer).run()
    assert r2.resumed is True
    assert trainer.trained == []
    assert len(r2.population) == len(r1.population)


def test_resume_mid_labeling_continues(tmp_path):
    out = str(tmp_path / "run")
    done = _cand("c0", 1.0, fitness=0.3, last_metrics={"SR": 0.5, "CR": 0.1, "TR": 0.1})
    todo = _cand("c1", 1.5)
    pop = [done, todo]
    save_checkpoint(
        out,
        build_checkpoint(
            status="running",
            phase="labeling",
            epoch=0,
            next_epoch=0,
            population=pop,
            to_label=pop,
            labeled_ids_this_epoch=["c0"],
            global_best=done,
            history=[],
            config={},
        ),
    )

    trainer = _RecordingTrainer()
    result = ClosedLoopRunner(_cfg(out, generations=1), trainer=trainer).run()

    assert trainer.trained == ["c1"]
    assert result.resumed is True
    ckpt = load_checkpoint(out)
    assert ckpt is not None
    assert ckpt["status"] == "completed"
