"""Stage III crash-safe resume (stub trainer — no real PPO)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import RewardCandidate
from raise_core.stage3_checkpoint import load_checkpoint, stage3_dir_from_train_root
from raise_core.validate import Stage3Config, Stage3Runner, StubPolicyTrainer


def _make_population(n: int = 3) -> list:
    validator = make_crowdnav_validator()
    pop = []
    for i in range(n):
        code = f"def compute_reward(state, memory):\n    return float({i}.0)\n"
        reward_fn, err = validator.try_validate(code)
        assert reward_fn is not None, err
        pop.append(
            RewardCandidate(
                candidate_id=f"c{i}",
                code=code,
                reward_fn=reward_fn,
                valid=True,
                origin="initial",
            )
        )
    return pop


class _Killed(BaseException):
    """Process-level interruption (like Ctrl+C); Stage III must not swallow it."""


class _CrashAfter(StubPolicyTrainer):
    def __init__(self, n_ok: int) -> None:
        self.n_ok = n_ok
        self.calls = 0

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        if self.calls >= self.n_ok:
            raise _Killed("simulated crash")
        self.calls += 1
        return super().train_and_eval(
            candidate, round_index=round_index, config=config
        )


def test_stage3_resume_skips_finished_candidates(tmp_path: Path):
    pop = _make_population(3)
    out_root = tmp_path / "stage3_train"
    cfg = Stage3Config(
        train_env_steps=8, eval_episodes=2, output_root=str(out_root), resume=True
    )
    with pytest.raises(_Killed, match="simulated crash"):
        Stage3Runner(_CrashAfter(2), validator=make_crowdnav_validator(), config=cfg).run(pop)

    ckpt_dir = stage3_dir_from_train_root(str(out_root))
    mid = load_checkpoint(ckpt_dir)
    assert mid is not None
    assert mid.get("status") == "running"
    assert int(mid.get("next_index")) == 2

    trainer = _CrashAfter(99)
    resumed = Stage3Runner(trainer, validator=make_crowdnav_validator(), config=cfg)
    final = resumed.run(pop)
    assert trainer.calls == 1
    assert [c.candidate_id for c in final] == ["c0", "c1", "c2"]
    assert [r.candidate_id for r in resumed.history] == ["c0", "c1", "c2"]
    done = load_checkpoint(ckpt_dir)
    assert done is not None
    assert done.get("status") == "done"
    assert os.path.isfile(os.path.join(ckpt_dir, "RESUME.json"))

    again = _CrashAfter(0)
    reused = Stage3Runner(again, validator=make_crowdnav_validator(), config=cfg).run(pop)
    assert again.calls == 0
    assert len(reused) == 3


def test_stage3_resume_disabled_starts_fresh(tmp_path: Path):
    pop = _make_population(2)
    out_root = tmp_path / "stage3_train"
    cfg = Stage3Config(output_root=str(out_root), resume=True)
    Stage3Runner(StubPolicyTrainer(), validator=make_crowdnav_validator(), config=cfg).run(pop)

    trainer = _CrashAfter(99)
    runner = Stage3Runner(
        trainer,
        validator=make_crowdnav_validator(),
        config=Stage3Config(output_root=str(out_root), resume=False),
    )
    out = runner.run(pop)
    assert trainer.calls == 2
    assert len(out) == 2
    assert len(runner.history) == 2


class _FailOn(StubPolicyTrainer):
    def __init__(self, bad_id: str) -> None:
        self.bad_id = bad_id

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        if candidate.candidate_id == self.bad_id:
            raise RuntimeError("reward blew up mid-training")
        return super().train_and_eval(
            candidate, round_index=round_index, config=config
        )


def test_stage3_failed_candidate_is_skipped_and_run_finishes(tmp_path: Path):
    pop = _make_population(3)
    out_root = tmp_path / "stage3_train"
    cfg = Stage3Config(output_root=str(out_root), resume=True)
    runner = Stage3Runner(_FailOn("c1"), validator=make_crowdnav_validator(), config=cfg)
    out = runner.run(pop)

    assert [c.candidate_id for c in out] == ["c0", "c2"]
    assert runner.best_trained is not None
    assert runner.best_trained.candidate_id != "c1"
    statuses = {r.candidate_id: r.status for r in runner.history}
    assert statuses == {"c0": "ok", "c1": "failed", "c2": "ok"}
    failed = next(r for r in runner.history if r.status == "failed")
    assert "reward blew up" in (failed.error or "")

    done = load_checkpoint(stage3_dir_from_train_root(str(out_root)))
    assert done is not None and done.get("status") == "done"
    restored = Stage3Runner(
        _FailOn("none"), validator=make_crowdnav_validator(), config=cfg
    )
    restored.run(pop)
    assert [r.status for r in restored.history] == ["ok", "failed", "ok"]
