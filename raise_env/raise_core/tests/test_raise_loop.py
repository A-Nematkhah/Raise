"""Closed-loop runner: every new candidate is trained + evaluated each generation."""

from __future__ import annotations

import json
import os

from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.logging_io import read_epochs
from raise_core.raise_loop.report import format_epoch_summary
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core.evo_selection.config import SelectionConfig
from raise_core.evo_selection.llm_ranker import RankResult, LLMRanker
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
    # Gen0: all 4 trained; Gen1: 3 new genomes trained, the carried elite is
    # evaluated again (no training) on Gen1's scenarios.
    elite = epochs[0]["evolve_best_id"]
    assert epochs[0]["n_to_label"] == 4
    assert epochs[1]["n_to_label"] == 3
    assert elite not in epochs[1]["labeled_ids"]
    assert epochs[1]["reevaluated_ids"] == [elite]
    assert epochs[1]["stale_fitness_ids"] == []
    assert elite in epochs[1]["label_metrics"]
    assert len(trainer.trained) == len(set(trainer.trained)) == 7
    assert result.n_labeled_total == 7

    # Evolution order is by fitness; Gen0 trained everyone.
    gen0 = epochs[0]
    fits = {cid: gen0["label_metrics"][cid]["fitness"] for cid in gen0["labeled_ids"]}
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


def test_llm_rank_mode_ranks_cards_and_uses_elitist_evolution(monkeypatch, tmp_path):
    calls = []

    def rank(_self, cards, *, fallback_order, **kwargs):
        calls.append((cards, list(fallback_order), kwargs))
        ranking = list(reversed(fallback_order))
        return RankResult(
            ranking=ranking,
            scores={candidate_id: 0.5 for candidate_id in ranking},
            weaknesses={candidate_id: ["improve safe progress"] for candidate_id in ranking},
            rationale={candidate_id: "Balance safety and completion." for candidate_id in ranking},
        )

    monkeypatch.setattr(LLMRanker, "rank", rank)
    out = str(tmp_path / "llm-rank-run")
    cfg = _cfg(
        out,
        selection=SelectionConfig(
            mode="llm_rank",
            top_k=2,
            p_restart=0.0,
            n_votes=1,
            seed=61,
        ),
    )
    result = ClosedLoopRunner(cfg, trainer=StubPolicyTrainer()).run()
    epochs = read_epochs(out)

    assert len(calls) == 2
    assert all(len(cards) == 4 for cards, _, _ in calls)
    assert all("DIAGNOSTIC FEEDBACK CARD" in card for cards, _, _ in calls for _, card in cards)
    assert epochs[0]["selection"]["ranking"] == list(reversed(calls[0][1]))
    assert epochs[0]["evolve_ranking"] == epochs[0]["selection"]["ranking"]
    assert len(result.population) == 4
    assert len(result.manifest["selection_notes"]) == 2


class _ReevalRecordingTrainer(_RecordingTrainer):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.reevaluated: list[tuple[str, int]] = []

    def reevaluate(self, candidate, *, round_index, config):
        self.reevaluated.append((str(candidate.candidate_id), int(config.eval_seed_base)))
        if self.fail:
            raise RuntimeError("saved policy unreadable")
        return super().reevaluate(candidate, round_index=round_index, config=config)


def _pop_epoch(out: str, g: int) -> dict:
    with open(os.path.join(out, "closed_loop", f"pop_epoch_{g:03d}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_every_ranked_fitness_comes_from_its_own_generation(tmp_path):
    out = str(tmp_path / "run")
    trainer = _ReevalRecordingTrainer()
    ClosedLoopRunner(_cfg(out), trainer=trainer).run()
    epochs = read_epochs(out)
    last = len(epochs) - 1
    assert last >= 1

    for g in range(last + 1):
        rows = _pop_epoch(out, g)["candidates"]
        assert all(r["eval_generation"] == g for r in rows)
    # Each carried elite was scored on its new generation's eval seeds.
    for elite_id, seed_base in trainer.reevaluated:
        g = next(e["epoch"] for e in epochs if elite_id in e["reevaluated_ids"])
        assert seed_base == 425 + g * 1000
    assert {cid for cid, _ in trainer.reevaluated} == {
        cid for e in epochs for cid in e["reevaluated_ids"]
    }
    assert len(trainer.trained) == len(set(trainer.trained))  # never retrained

    final = _pop_epoch(out, last)["candidates"]
    carried = [r for r in final if r.get("fitness_history")]
    assert carried, "elitism carries at least one genome into the last generation"
    for row in carried:
        gens = [h["generation"] for h in row["fitness_history"]]
        assert gens == sorted(gens) and gens[-1] < last


def test_failed_reevaluation_keeps_stale_fitness_and_flags_it(tmp_path):
    out = str(tmp_path / "run")
    trainer = _ReevalRecordingTrainer(fail=True)
    result = ClosedLoopRunner(_cfg(out), trainer=trainer).run()
    gen1 = read_epochs(out)[1]
    elite = read_epochs(out)[0]["evolve_best_id"]
    assert gen1["stale_fitness_ids"] == [elite]
    assert gen1["reevaluated_ids"] == []
    assert gen1["n_label_failed"] == 0  # trainings, not eval-only passes
    md = next(c for c in result.population if c.candidate_id == elite).metadata
    assert md["fitness_stale"] is True
    assert md["eval_generation"] == 0
    assert md["fitness"] is not None


class _TrainOnlyTrainer:
    def __init__(self) -> None:
        self.stub = StubPolicyTrainer()
        self.trained: list[str] = []

    def train_and_eval(self, candidate, *, round_index, config):
        self.trained.append(str(candidate.candidate_id))
        return self.stub.train_and_eval(candidate, round_index=round_index, config=config)


def test_trainer_without_reevaluate_keeps_carried_fitness(tmp_path):
    out = str(tmp_path / "run")
    trainer = _TrainOnlyTrainer()
    ClosedLoopRunner(_cfg(out), trainer=trainer).run()
    epochs = read_epochs(out)
    assert epochs[1]["reevaluated_ids"] == []
    assert epochs[0]["evolve_best_id"] not in trainer.trained[4:]


def test_format_epoch_summary():
    line = format_epoch_summary(
        {
            "epoch": 1,
            "n_population": 4,
            "evolve_best_id": "mut_0006",
            "evolve_best_fitness": 0.5,
            "evolve_best_score1": 0.606,
            "labeled_ids": ["mut_0006", "cro_0004"],
            "n_label_ok": 2,
            "n_label_failed": 0,
        }
    )
    assert line == "epoch 1: best=mut_0006 fitness=0.500 Score1=0.6060 | trained 2/4 ok=2 fail=0"
