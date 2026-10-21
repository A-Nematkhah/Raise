"""Identical genomes are never trained twice (fingerprints, LLM duplicates, fallback clones)."""

from __future__ import annotations

import json

from domains.crowdnav import feedback_card as crowdnav_feedback_card
from domains.crowdnav import prompts as cn_prompts
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate, genome_fingerprint
from raise_core.explore import StageIConfig, StageIEvolver
from raise_core.feedback import ArchiveRecord, CandidateArchive, FeedbackCardConfig
from raise_core.llm import EmptyCompletionError, LLMClient, ScriptedLLMClient
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.logging_io import read_epochs
from raise_core.raise_loop.runner import ClosedLoopRunner, _same_genome_copies
from raise_core.refine import StubPolicyTrainer


def _code(bonus: float, *, comment: str = "", doc: bool = False) -> str:
    docstring = '    """Progress plus a bonus."""\n' if doc else ""
    return (
        "def compute_reward(state, memory):\n"
        f"{docstring}"
        f"    total = float(state.progress) + {bonus}{comment}\n"
        "    return total, {'progress': total}\n"
    )


def _fenced(code: str) -> str:
    return "```python\n" + code + "```\n"


def test_fingerprint_ignores_layout_comments_and_docstrings():
    base = genome_fingerprint(_code(1.0))
    assert genome_fingerprint(_code(1.0, comment="  # tuned")) == base
    assert genome_fingerprint(_code(1.0, doc=True)) == base
    assert genome_fingerprint("\n\n" + _code(1.0).replace("    ", "        ")) == base
    assert genome_fingerprint(_code(2.0)) != base
    assert genome_fingerprint("def broken(:") == genome_fingerprint("def broken(:  \n")


def _evolver(completions, *, archive=None, log_path=None, **cfg) -> StageIEvolver:
    base = dict(population_size=4, max_invalid_replacements=3)
    base.update(cfg)
    evo = StageIEvolver(
        ScriptedLLMClient(completions),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_crowdnav_validator(),
        config=StageIConfig(**base),
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(enabled=False),
        feedback_card=crowdnav_feedback_card,
        rejection_log_path=log_path,
    )
    if archive is not None:
        evo.archive = archive
    return evo


def _evaluated(cid: str, code: str, fitness: float, generation: int = 0) -> RewardCandidate:
    md = {
        "fitness": fitness,
        "last_metrics": {"SR": 0.5, "fitness": fitness},
        "checkpoint_path": f"/runs/{cid}/model.zip",
        "eval_generation": generation,
    }
    return RewardCandidate(cid, code, valid=True, origin="initial", metadata=md)


def test_llm_child_identical_to_archived_genome_is_rejected_and_regenerated(tmp_path):
    archive = CandidateArchive()
    archive.add(ArchiveRecord.from_candidate(_evaluated("ini_0000", _code(1.0), 0.4), generation=0))
    log = tmp_path / "rejections.jsonl"
    evo = _evolver(
        [_fenced(_code(1.0, comment="  # same genome")), _fenced(_code(3.0))],
        archive=archive,
        log_path=str(log),
    )
    child = evo._fill_valid(1, lambda n: evo._try_random_restart(phase="random", attempt=n),
                            max_extra=3, phase="random")[0]
    assert child.valid and "+ 3.0" in child.code
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["rejection_category"] == "duplicate"
    assert "ini_0000" in rows[0]["validation_error"]


def test_children_of_one_generation_must_differ():
    evo = _evolver([_fenced(_code(5.0)), _fenced(_code(5.0)), _fenced(_code(6.0))])
    first = evo._try_random_restart(phase="random", attempt=1)
    second = evo._try_random_restart(phase="random", attempt=2)
    third = evo._try_random_restart(phase="random", attempt=3)
    assert first.valid and not second.valid and third.valid
    assert second.validation_error == f"duplicate genome of {first.candidate_id}"


def test_fallback_clone_takes_over_the_source_evaluation():
    top = _evaluated("ini_0000", _code(1.0), 0.9)
    second = _evaluated("ini_0001", _code(2.0), 0.5)
    lower = [_evaluated(f"ini_000{i}", _code(float(i)), 0.1 * i) for i in (2, 3)]
    # Every LLM reply repeats the elite, so every slot ends in its fallback.
    evo = _evolver([_fenced(_code(1.0))] * 40, keep_runtime_elite=True)
    gen1 = evo._next_generation([top, second, *lower])
    clones = [c for c in gen1 if c.origin in ("crossover_fallback", "mutation_fallback")]
    assert {c.origin for c in clones} == {"crossover_fallback", "mutation_fallback"}
    for clone in clones:
        source = next(c for c in [top, second, *lower] if c.candidate_id == clone.parent_ids[0])
        assert clone.metadata["duplicate_of"] == source.candidate_id
        assert clone.metadata["fitness"] == source.metadata["fitness"]
        assert clone.metadata["checkpoint_path"] == source.metadata["checkpoint_path"]


def test_crossover_partner_is_a_different_genome():
    top = _evaluated("ini_0000", _code(1.0), 0.9)
    copy = _evaluated("cro_0004", _code(1.0, comment="  # copy"), 0.8)
    other = _evaluated("ini_0001", _code(2.0), 0.5)
    low = _evaluated("ini_0002", _code(3.0), 0.1)
    evo = _evolver([_fenced(_code(9.0)), _fenced(_code(8.0)), _fenced(_code(7.0)),
                    _fenced(_code(6.0))], keep_runtime_elite=False)
    gen1 = evo._next_generation([top, copy, other, low])
    cross = next(c for c in gen1 if c.origin == "crossover")
    assert cross.parent_ids == ("ini_0000", "ini_0001")


def test_same_genome_copies_maps_later_duplicates_to_first():
    a = RewardCandidate("a", _code(1.0))
    b = RewardCandidate("b", _code(1.0, comment="  # again"))
    c = RewardCandidate("c", _code(2.0))
    copies = _same_genome_copies([a, b, c])
    assert set(copies) == {"b"} and copies["b"] is a


class _CountingTrainer(StubPolicyTrainer):
    def __init__(self) -> None:
        self.trained: list[str] = []

    def train_and_eval(self, candidate, *, round_index, config):
        self.trained.append(candidate.candidate_id)
        return super().train_and_eval(candidate, round_index=round_index, config=config)


class _OutageLLM(LLMClient):
    provider = "scripted"

    def complete(self, prompt, *, max_tokens=None):
        raise EmptyCompletionError("Groq returned an empty completion.")


def test_llm_outage_trains_the_seed_fallback_once(tmp_path):
    """Every slot falls back; no genome is trained twice across the run."""
    out = str(tmp_path / "run")
    trainer = _CountingTrainer()
    cfg = ClosedLoopConfig(population_size=4, generations=2, stage2_train_steps=32,
                           use_stub=True, output_dir=out, device="cpu",
                           feedback=FeedbackCardConfig(enabled=False))
    ClosedLoopRunner(cfg, llm=_OutageLLM(), trainer=trainer).run()
    gen0, gen1 = read_epochs(out)

    # Gen0: four seed copies -> one training, three take over its evaluation.
    assert gen0["n_unique_genomes"] == 1
    assert len(gen0["labeled_ids"]) == 1 and len(gen0["duplicate_ids"]) == 3
    # Gen1: every fallback copies an evaluated genome -> eval-only passes.
    assert gen1["labeled_ids"] == [] and gen1["duplicate_ids"] == []
    assert len(gen1["reevaluated_ids"]) == 4
    assert trainer.trained == gen0["labeled_ids"]
