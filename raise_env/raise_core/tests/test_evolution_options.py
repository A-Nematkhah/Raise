"""Opt-in evolution knobs: mutation parent selection, reference card on random calls."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from domains.highway import prompts as hwy_prompts
from domains.highway.reward_checks import make_highway_validator
from raise_core.candidate import RewardCandidate
from raise_core.explore import StageIConfig, StageIEvolver, select_mutation_parents
from raise_core.feedback import ArchiveRecord, CandidateArchive, FeedbackCardConfig
from raise_core.llm import ScriptedLLMClient


def _ranked(n: int = 6):
    return [
        RewardCandidate(f"c{i}", f"def compute_reward(state, memory):\n    return {i}.0\n",
                        valid=True, origin="initial",
                        metadata={"fitness": 1.0 - 0.1 * i, "last_metrics": {"SR": 0.5}})
        for i in range(n)
    ]


def _ids(parents):
    return [p.candidate_id for p in parents]


def test_lower_is_the_bottom_half_cycled():
    ranked = _ranked()
    assert _ids(select_mutation_parents(ranked, 2, mode="lower", rng=random.Random(0))) == ["c3", "c4"]
    assert _ids(select_mutation_parents(ranked, 4, mode="lower", rng=random.Random(0))) == [
        "c3", "c4", "c5", "c3",
    ]


def test_rank_weighted_prefers_better_ranks():
    picks = Counter(_ids(select_mutation_parents(_ranked(), 6000, mode="rank_weighted",
                                                 rng=random.Random(1))))
    # Weights 6:5:4:3:2:1 -> best about 6x the worst.
    assert picks["c0"] > picks["c2"] > picks["c5"]
    assert 4.5 < picks["c0"] / picks["c5"] < 7.5


def test_tournament_never_picks_the_worst():
    picks = set(_ids(select_mutation_parents(_ranked(), 500, mode="tournament",
                                             rng=random.Random(2))))
    assert "c5" not in picks and "c0" in picks


_INVALID = "```python\nimport os\ndef compute_reward(state, memory):\n    return 0.0\n```\n"


def _evolver(**cfg) -> StageIEvolver:
    return StageIEvolver(
        ScriptedLLMClient([_INVALID] * 20),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_highway_validator(),
        config=StageIConfig(population_size=6, **cfg),
        prompts=hwy_prompts,
        feedback=FeedbackCardConfig(enabled=False),
    )


def test_unknown_mutation_parents_is_rejected():
    with pytest.raises(ValueError, match="mutation_parents"):
        _evolver(mutation_parents="roulette")


def test_mutation_parent_draws_repeat_for_the_same_ranking():
    """Seeded by run seed + ranked ids: a resumed run picks the same parents."""
    ranked = _ranked()

    def _parents(seed):
        evo = _evolver(mutation_parents="rank_weighted", seed=seed, max_invalid_replacements=0)
        gen = evo._next_generation(ranked)
        return [c.parent_ids for c in gen if c.origin == "mutation_fallback"]

    assert _parents(7) == _parents(7)
    assert len(_parents(7)) == 2


def _card_evolver(random_calls: bool) -> StageIEvolver:
    evo = StageIEvolver(
        ScriptedLLMClient([_INVALID]),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_highway_validator(),
        prompts=hwy_prompts,
        feedback=FeedbackCardConfig(random_calls=random_calls),
    )
    archive = CandidateArchive()
    archive.add(ArchiveRecord.from_candidate(_ranked(1)[0], generation=0))
    evo.archive = archive
    return evo


def test_random_calls_switch_controls_the_reference_card():
    assert _card_evolver(True)._make_card("random", ()) is not None
    assert _card_evolver(False)._make_card("random", ()) is None
    # Mutation keeps its card either way.
    assert _card_evolver(False)._make_card("mutation", _ranked(1)) is not None
