"""
Tests for Stage I evolver + LLM client (mocked / scripted completions).
"""

from __future__ import annotations

import json
import os

import pytest

from domains.crowdnav.prompts import (
    D1_SYSTEM_PROMPT,
    D4_EXTERNAL_KNOWLEDGE,
    D5_SEED_FUNCTION,
    format_d1_initial,
    format_d2_crossover,
    format_d2_mutation,
)
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.state import default_smoke_states
from raise_core.explore import StageIConfig, StageIEvolver
from raise_core.llm import (
    CompletionTruncatedError,
    EmptyCompletionError,
    LLMClient,
    ScriptedLLMClient,
    extract_python_code,
    is_recoverable_llm_error,
    make_llm_client,
    normalize_to_compute_reward,
)
from raise_core.sandbox.errors import RewardSandboxError


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float(state.progress) + {value}\n"
        "```\n"
    )


def _invalid_import_code() -> str:
    return (
        "```python\n"
        "import os\n"
        "def compute_reward(state, memory):\n"
        "    return float(0.0)\n"
        "```\n"
    )


def _score_by_smoke(reward_fn, *, candidate_id: str = "") -> float:
    return float(reward_fn.compute(default_smoke_states()[0]))


def _evolver(client, **cfg) -> StageIEvolver:
    return StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        validator=make_crowdnav_validator(),
        config=StageIConfig(**cfg),
    )


def test_extract_and_normalize_code():
    raw = "Here you go:\n```python\ndef cal_reward(state):\n    return float(1.0)\n```\n"
    code = extract_python_code(raw)
    assert "def cal_reward" in code
    norm = normalize_to_compute_reward(code)
    assert "def compute_reward(state, memory)" in norm
    assert "cal_reward" not in norm


def test_prompt_templates_contain_appendix_anchors():
    assert "expert in reinforcement learning" in D1_SYSTEM_PROMPT
    user = format_d1_initial(reflection="keep exploring")
    assert "compute_reward" in user
    assert "CrowdNavRewardState" in user
    assert "keep exploring" in user
    assert "memory" in user
    assert "CrowdNav" in D4_EXTERNAL_KNOWLEDGE
    assert "def compute_reward(state, memory)" in D5_SEED_FUNCTION
    cross = format_d2_crossover("codeA", "codeB", "note")
    assert "codeA" in cross and "codeB" in cross
    mut = format_d2_mutation("elitist", "weakness")
    assert "elitist" in mut and "weakness" in mut
    assert "state, memory" in mut


def test_evolver_requires_validator():
    with pytest.raises(ValueError):
        StageIEvolver(ScriptedLLMClient([]), score_fn=_score_by_smoke)


def test_scripted_generate_n():
    client = ScriptedLLMClient(["a", "b", "c"])
    assert client.generate("p", 2) == ["a", "b"]
    assert client.complete("p") == "c"
    with pytest.raises(IndexError):
        client.complete("p")


def test_make_llm_client_scripted():
    client = make_llm_client("scripted", completions=["x"])
    assert isinstance(client, ScriptedLLMClient)
    assert client.complete("hi") == "x"


def test_gen0_batch_scores_and_ranks():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    evolver = _evolver(ScriptedLLMClient([batch_body]), population_size=8)
    pop = evolver.initialize_population()
    assert len(pop) == 8
    assert all(c.valid and c.reward_fn is not None for c in pop)
    ranked = evolver.score_population(pop)
    scores = [c.score for c in ranked]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_score1_sandbox_error_rejects_candidate_and_continues():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    evolver = _evolver(ScriptedLLMClient([batch_body]), population_size=8)
    pop = evolver.initialize_population()
    bad_id = pop[1].candidate_id

    def _score(reward_fn, *, candidate_id: str = ""):
        if candidate_id == bad_id:
            raise RewardSandboxError("runtime error in sandboxed compute(): ZeroDivisionError")
        return _score_by_smoke(reward_fn)

    evolver.score_fn = _score
    ranked = evolver.score_population(pop)
    assert len(ranked) == 8
    bad = next(c for c in ranked if c.candidate_id == bad_id)
    assert bad.score == float("-inf")
    assert bad.metadata["score1_rejected"]
    assert bad.metadata["score1_reject_reason"].startswith("score1_error: RewardSandboxError")
    assert all(c.score > float("-inf") for c in ranked if c.candidate_id != bad_id)


def test_gen0_replaces_invalid_with_extra_draws(tmp_path):
    completions = [_invalid_import_code(), _invalid_import_code()]
    completions += [_valid_code(float(i)) for i in range(1, 9)]
    log_path = os.path.join(str(tmp_path), "stage1_rejections.jsonl")
    evolver = StageIEvolver(
        ScriptedLLMClient(completions),
        score_fn=_score_by_smoke,
        validator=make_crowdnav_validator(),
        config=StageIConfig(population_size=8, max_invalid_replacements=8),
        rejection_log_path=log_path,
    )
    pop = evolver.initialize_population()
    assert len(pop) == 8
    assert all(c.valid for c in pop)
    rows = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    rejected = [r for r in rows if not r["accepted"]]
    assert len(rejected) >= 2
    assert all(r["rejection_category"] == "forbidden_import" for r in rejected)


def _ranked_gen0(evolver: StageIEvolver):
    return evolver.score_population(evolver.initialize_population())


def test_next_generation_with_elite_uses_top2_and_lower_half():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    completions = [batch_body]
    completions += [_valid_code(100.0 + i) for i in range(2)]  # crossover slots
    completions += [_valid_code(50.0 + i) for i in range(4)]  # mutation slots
    completions += [_valid_code(10.0)]  # random slot (elite takes one)
    evolver = _evolver(
        ScriptedLLMClient(completions), population_size=8, keep_runtime_elite=True,
        n_crossover=2, n_mutation=4,
    )
    ranked = _ranked_gen0(evolver)
    gen1 = evolver._next_generation(ranked)
    assert len(gen1) == 8
    assert gen1[0] is ranked[0]
    assert sum(1 for c in gen1 if c.origin == "crossover") == 2
    assert sum(1 for c in gen1 if c.origin == "mutation") == 4
    assert sum(1 for c in gen1 if c.origin == "random") == 1
    top2 = {ranked[0].candidate_id, ranked[1].candidate_id}
    lower = {c.candidate_id for c in ranked[4:]}
    for c in gen1:
        if c.origin == "crossover":
            assert set(c.parent_ids) == top2
        if c.origin == "mutation":
            assert len(c.parent_ids) == 1 and c.parent_ids[0] in lower


def test_next_generation_exact_242_without_runtime_elite():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    completions = [batch_body]
    completions += [_valid_code(100.0 + i) for i in range(2)]
    completions += [_valid_code(50.0 + i) for i in range(4)]
    completions += ["\n\n".join([_valid_code(10.0), _valid_code(11.0)])]
    evolver = _evolver(
        ScriptedLLMClient(completions), population_size=8, keep_runtime_elite=False,
        n_crossover=2, n_mutation=4,
    )
    gen1 = evolver._next_generation(_ranked_gen0(evolver))
    assert sum(1 for c in gen1 if c.origin == "crossover") == 2
    assert sum(1 for c in gen1 if c.origin == "mutation") == 4
    assert sum(1 for c in gen1 if c.origin == "random") == 2
    assert len(gen1) == 8


def test_fill_valid_soft_fails_empty_completion_and_pads_seed():
    class FlakyLLM(LLMClient):
        provider = "scripted"

        def __init__(self) -> None:
            self.n = 0

        def complete(self, prompt: str, *, max_tokens=None) -> str:
            self.n += 1
            if self.n <= 3:
                raise EmptyCompletionError("Groq returned an empty completion.")
            return _valid_code(1.0)

    evolver = _evolver(
        FlakyLLM(),
        population_size=2,
        max_invalid_replacements=2,
    )
    # Gen0 batch raises empty → regen → more empties → seed fallback fills.
    pop = evolver.initialize_population()
    assert len(pop) == 2
    assert all(c.valid for c in pop)


def test_is_recoverable_llm_error_matches_known_messages():
    assert is_recoverable_llm_error(EmptyCompletionError("x"))
    assert is_recoverable_llm_error(CompletionTruncatedError("truncated"))
    assert is_recoverable_llm_error(RuntimeError("Groq returned an empty completion."))
    assert not is_recoverable_llm_error(RuntimeError("auth failed 401"))
