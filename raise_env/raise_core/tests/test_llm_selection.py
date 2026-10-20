from __future__ import annotations

import json
import pytest

from raise_core.candidate import RewardCandidate
from raise_core.evo_selection.config import SelectionConfig
from raise_core.evo_selection.consistency import compare_rankings
from raise_core.evo_selection.evolution_ops import make_evolution_plan
from raise_core.evo_selection.feedback import build_refinement_feedback
from raise_core.evo_selection.guards import candidate_guard_flags
from raise_core.evo_selection.llm_ranker import (
    LLMRanker,
    _validate_response,
)


def _response(
    ranking=("A", "B", "C"),
    *,
    scores=None,
    weaknesses=None,
    rationale=None,
):
    labels = ("A", "B", "C")
    payload = {
        "ranking": list(ranking),
        "scores": scores or {label: 0.5 for label in labels},
        "weaknesses": weaknesses or {label: ["needs safer progress"] for label in labels},
        "rationale": rationale or {label: f"{label} is a balanced candidate." for label in labels},
    }
    return json.dumps(payload)


class RecordingClient:
    temperature = 0.7

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt, *, max_tokens=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _cards():
    return [
        ("reward-1", "card reward-1\nA metrics line."),
        ("reward-2", "card reward-2\nB metrics line."),
        ("reward-3", "card reward-3\nC metrics line."),
    ]


def test_json_validator_requires_exact_labels_and_schema():
    valid = json.loads(_response())
    assert _validate_response(json.dumps(valid), ["A", "B", "C"]) == valid
    valid["ranking"] = ["A", "A", "C"]
    with pytest.raises(ValueError, match="exactly once"):
        _validate_response(json.dumps(valid), ["A", "B", "C"])


def test_ranker_retries_invalid_json_then_falls_back_to_existing_order(tmp_path):
    client = RecordingClient(["not json"] * 3)
    result = LLMRanker(
        client,
        SelectionConfig(mode="llm_rank", n_votes=1),
        log_path=str(tmp_path / "ranking.jsonl"),
    ).rank(
        _cards(),
        fallback_order=["reward-2", "reward-1", "reward-3"],
        seed=9,
    )
    assert result.fallback
    assert result.ranking == ["reward-2", "reward-1", "reward-3"]
    assert len(client.prompts) == 3
    rows = [json.loads(line) for line in (tmp_path / "ranking.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "ranking_vote",
        "ranking_vote",
        "ranking_vote",
        "ranking_fallback",
    ]
    assert rows[0]["raw_response"] == "not json"
    assert rows[0]["parsed_result"] is None


def test_borda_aggregation_and_label_to_id_mapping():
    client = RecordingClient(
        [
            _response(("A", "B", "C")),
            _response(("B", "C", "A")),
            _response(("B", "A", "C")),
        ]
    )
    result = LLMRanker(
        client, SelectionConfig(mode="llm_rank", n_votes=3)
    ).rank(_cards(), fallback_order=["reward-1", "reward-2", "reward-3"], seed=2)
    assert result.ranking == ["reward-2", "reward-1", "reward-3"]
    assert result.label_to_id == {
        "A": "reward-1",
        "B": "reward-2",
        "C": "reward-3",
    }
    assert result.weaknesses["reward-1"] == ["needs safer progress"]
    assert result.scores["reward-2"] == pytest.approx(0.5)
    assert client.temperature == 0.7


def test_seeded_shuffle_is_reproducible_and_card_text_is_unchanged_except_id():
    def run():
        client = RecordingClient([_response()] * 3)
        result = LLMRanker(
            client,
            SelectionConfig(mode="llm_rank", n_votes=3, temperature=0.05),
        ).rank(_cards(), fallback_order=["reward-1", "reward-2", "reward-3"], seed=77)
        return client, result

    first_client, first_result = run()
    second_client, second_result = run()
    assert first_result.ranking == second_result.ranking
    assert first_client.prompts == second_client.prompts
    prompt = first_client.prompts[0]
    assert "card Candidate A\nA metrics line." in prompt
    assert "card Candidate B\nB metrics line." in prompt
    assert "card Candidate C\nC metrics line." in prompt
    assert "card reward-1" not in prompt
    assert first_client.temperature == 0.7


def test_anonymization_only_replaces_identifier_tokens():
    from raise_core.evo_selection.llm_ranker import anonymize_card_text

    assert anonymize_card_text("a is a metric; id=a.", "a", "A") == (
        "a is a metric; id=Candidate A."
    )


def test_guard_flags_are_optional_and_use_existing_metrics():
    candidate = RewardCandidate(
        candidate_id="bad",
        code="",
        valid=False,
        score=float("nan"),
        metadata={
            "fitness": float("nan"),
            "last_metrics": {"SR": 0.95, "path_length": 0.01},
        },
    )
    disabled = SelectionConfig(mode="llm_rank", guards=False)
    assert candidate_guard_flags(candidate, disabled) == []
    enabled = SelectionConfig(mode="llm_rank", guards=True)
    assert candidate_guard_flags(candidate, enabled) == [
        "invalid_reward_code",
        "non_finite_score",
        "non_finite_fitness",
        "high_success_near_zero_progress",
    ]


def test_refinement_feedback_keeps_card_text_byte_for_byte_as_prefix():
    card = "=== CARD ===\r\ncandidate A: metrics"
    prompt = build_refinement_feedback(
        card,
        weaknesses=["high timeout rate"],
        reflection_notes=["Prefer safe progress."],
    )
    assert prompt.startswith(card)
    assert prompt[len(card) :].startswith("\n\n")
    assert "high timeout rate" in prompt
    assert "Prefer safe progress." in prompt


def test_evolution_plan_is_seeded_elitist_and_uses_top_parents():
    population = [
        RewardCandidate(candidate_id=f"c{i}", code=f"def reward_{i}(): pass")
        for i in range(4)
    ]
    first = make_evolution_plan(
        population,
        population_size=4,
        top_k=2,
        p_restart=0.0,
        seed=4,
        generation=3,
    )
    again = make_evolution_plan(
        population,
        population_size=4,
        top_k=2,
        p_restart=0.0,
        seed=4,
        generation=3,
    )
    assert [candidate.candidate_id for candidate in first.elites] == ["c0", "c1"]
    assert first.offspring == again.offspring
    assert {child.operator for child in first.offspring} == {"mutation", "crossover"}
    assert first.offspring[1].parent_ids == ("c0", "c1")


def test_consistency_reports_spearman_and_top_k_preservation():
    same = compare_rankings(["a", "b", "c"], ["a", "b", "c"], top_k=2)
    assert same["spearman"] == pytest.approx(1.0)
    assert same["top_k_preservation"] == 1.0
    assert same["intersection_n"] == 3
    reverse = compare_rankings(["a", "b", "c"], ["c", "b", "a"], top_k=1)
    assert reverse["spearman"] == pytest.approx(-1.0)
    assert reverse["top_k_preservation"] == 0.0
    assert compare_rankings(["a"], ["a"], top_k=2)["spearman"] is None
