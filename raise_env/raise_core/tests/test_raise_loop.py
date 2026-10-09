"""Closed-loop runner: every new candidate is trained + evaluated each generation."""

from __future__ import annotations

import os

from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.logging_io import read_epochs
from raise_core.raise_loop.report import format_epoch_summary
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core.refine import StubPolicyTrainer


class _RecordingTrainer(StubPolicyTrainer):
    def __init__(self) -> None:
        self.trained: list[str] = []

    def train_and_eval(self, candidate, *, round_index, config):
        self.trained.append(str(candidate.candidate_id))
        return super().train_and_eval(candidate, round_index=round_index, config=config)


def _cfg(out: str, **kw) -> ClosedLoopConfig:
    base = dict(
        population_size=4,
        generations=2,
        stage2_train_steps=32,
        use_stub=True,
        llm_provider="seed",
        output_dir=out,
        device="cpu",
    )
    base.update(kw)
    return ClosedLoopConfig(**base)


def test_closed_loop_fast_two_epochs(tmp_path):
    out = str(tmp_path / "run")
    trainer = _RecordingTrainer()
    result = ClosedLoopRunner(_cfg(out), trainer=trainer).run()

    epochs = read_epochs(out)
    assert len(epochs) == 2
    assert [e["epoch"] for e in epochs] == [0, 1]
    # Gen0: all 4 trained; Gen1: the carried-over elite keeps its evaluation.
    assert epochs[0]["n_to_label"] == 4
    assert epochs[1]["n_to_label"] == 3
    assert epochs[0]["evolve_best_id"] not in epochs[1]["labeled_ids"]
    assert len(trainer.trained) == len(set(trainer.trained)) == 7
    assert result.n_labeled_total == 7

    # Evolution order is by fitness (scalar mode); Gen0 trained everyone.
    gen0 = epochs[0]
    fits = {cid: gen0["label_metrics"][cid]["fitness"] for cid in gen0["labeled_ids"]}
    assert gen0["evolve_rank"] == "scalar"
    assert gen0["evolve_ranking"] == sorted(fits, key=fits.get, reverse=True)
    assert gen0["evolve_best_fitness"] == max(fits.values())

    # Every final candidate carries metrics + evidence for the LLM.
    for cand in result.population:
        md = cand.metadata or {}
        assert isinstance(md.get("last_metrics"), dict)
        assert "Holdout eval" in md.get("proxy_feedback", "")

    assert result.manifest.get("mode") == "closed_loop"
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    assert os.path.isfile(report)
    text = open(report, encoding="utf-8").read()
    assert "epoch 0:" in text and "epoch 1:" in text
    assert "Closed-loop run report" in text


def test_format_epoch_summary():
    line = format_epoch_summary(
        {
            "epoch": 1,
            "n_population": 4,
            "evolve_best_id": "mut_0006",
            "evolve_best_fitness": 0.5,
            "evolve_best_score1": 0.606,
            "evolve_rank": "scalar",
            "labeled_ids": ["mut_0006", "cro_0004"],
            "n_label_ok": 2,
            "n_label_failed": 0,
        }
    )
    assert line.startswith("epoch 1: best=mut_0006 fitness=0.500 Score1=0.6060")
    assert "[rank=scalar]" in line
    assert "trained 2/4 ok=2 fail=0" in line
