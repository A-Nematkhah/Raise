"""
Tests for Stage I evolver + LLM client (mocked / scripted completions).
"""

from __future__ import annotations

import pytest

from crowd_nav.reward_search.evolver import StageIConfig, StageIEvolver
from crowd_nav.reward_search.llm import (
    ScriptedLLMClient,
    extract_python_code,
    make_llm_client,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.prompts import (
    D1_SYSTEM_PROMPT,
    D4_EXTERNAL_KNOWLEDGE,
    D5_SEED_FUNCTION,
    format_d1_initial,
    format_d2_crossover,
    format_d2_mutation,
)
from crowd_nav.reward_search.sandbox.runtime import default_smoke_states


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float({value})\n"
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


def test_extract_and_normalize_code():
    raw = "Here you go:\n```python\ndef cal_reward(state):\n    return float(1.0)\n```\n"
    code = extract_python_code(raw)
    assert "def cal_reward" in code
    norm = normalize_to_compute_reward(code)
    assert "def compute_reward(state, memory)" in norm
    assert "cal_reward" not in norm


def test_strip_scalar_subscripts_and_validate():
    from crowd_nav.reward_search.llm import strip_redundant_scalar_subscripts
    from crowd_nav.reward_search.sandbox.validator import RewardValidator

    bad = (
        "def compute_reward(state, memory):\n"
        "    dx = state.robot.px[0] - state.robot.gx[0]\n"
        "    dy = state.robot.py[0] - state.robot.gy[0]\n"
        "    dist = (dx * dx + dy * dy) ** 0.5\n"
        "    if state.dmin[0] < 0.2:\n"
        "        return float(-1.0)\n"
        "    return float(-dist)\n"
    )
    fixed = strip_redundant_scalar_subscripts(bad)
    assert "px[0]" not in fixed
    assert "dmin[0]" not in fixed
    assert "state.robot.px" in fixed
    # action[0] must remain indexable (tuple / ActionXY)
    with_action = (
        "def compute_reward(state, memory):\n"
        "    return float(state.action[0] + state.robot.vx[0])\n"
    )
    fixed_action = normalize_to_compute_reward(with_action)
    assert "state.action[0]" in fixed_action
    assert "vx[0]" not in fixed_action
    RewardValidator().validate_code(bad)
    RewardValidator().validate_code(with_action)


def test_prompt_templates_contain_appendix_anchors():
    assert "expert in reinforcement learning" in D1_SYSTEM_PROMPT
    user = format_d1_initial(reflection="keep exploring")
    assert "compute_reward" in user
    assert "RewardState" in user
    assert "keep exploring" in user
    assert "memory" in user
    assert "class MyReward" not in user
    assert "Crowd-robot navigation" in D4_EXTERNAL_KNOWLEDGE
    assert "def compute_reward(state, memory)" in D5_SEED_FUNCTION
    cross = format_d2_crossover("codeA", "codeB", "note")
    assert "codeA" in cross and "codeB" in cross
    assert "getattr" in cross and "state.robot.px" in cross
    mut = format_d2_mutation("elitist", "weakness")
    assert "elitist" in mut and "weakness" in mut
    assert "getattr" in mut
    assert "state, memory" in mut


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


def test_stage_i_builds_n8_and_runs_one_generation():
    # Gen0: batch (8 funcs in one reply) + regen if needed. Gen1: 8 valids.
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    completions = [batch_body]
    completions += [_valid_code(float(i) + 0.5) for i in range(8, 0, -1)]
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(
            population_size=8,
            generations=1,
            n_crossover=2,
            n_mutation=4,
            n_random=2,
            max_invalid_replacements=8,
        ),
    )
    final = evolver.run()
    assert len(final) == 8
    assert all(c.valid and c.reward_fn is not None for c in final)
    assert all(c.score is not None for c in final)
    # Ranked descending.
    scores = [c.score for c in final]
    assert scores == sorted(scores, reverse=True)
    assert len(evolver.history) == 2  # Gen0 + Gen1
    assert evolver.history[0].generation == 0
    assert evolver.history[1].generation == 1
    assert len(evolver.history[0].population) == 8
    assert evolver.history[1].reflection
    # Next-gen origins present.
    origins = {c.origin for c in evolver.history[1].population}
    assert "crossover" in origins
    assert "mutation" in origins
    assert "random" in origins


def test_global_best_survives_a_regressive_generation():
    batch = _valid_code(100.0)
    initial = [
        StageIEvolver(
            ScriptedLLMClient([_valid_code(0.0)]),
            score_fn=_score_by_smoke,
            config=StageIConfig(generations=0),
        )._make_candidate(_valid_code(float(index)), origin="initial")
        for index in range(8)
    ]
    initial[0].score = 100.0
    low = [
        StageIEvolver(
            ScriptedLLMClient([_valid_code(0.0)]),
            score_fn=_score_by_smoke,
            config=StageIConfig(generations=0),
        )._make_candidate(_valid_code(1.0), origin="mutation")
        for _ in range(8)
    ]
    for candidate in low:
        candidate.score = 1.0

    evolver = StageIEvolver(
        ScriptedLLMClient([batch]),
        score_fn=_score_by_smoke,
        config=StageIConfig(generations=1),
    )
    evolver.initialize_population = lambda: initial
    rankings = iter([initial, low])
    evolver.score_population = lambda population: next(rankings)
    evolver._next_generation = lambda ranked: low
    final = evolver.run()

    assert evolver.global_best is initial[0]
    assert evolver.global_best not in final

    from crowd_nav.reward_search.pipeline import EvoNavPipeline

    handed_to_stage2 = EvoNavPipeline._include_global_best(final, evolver.global_best)
    assert any(candidate.candidate_id == initial[0].candidate_id for candidate in handed_to_stage2)


def test_gen0_replaces_invalid_with_extra_draws():
    # Batch call returns one invalid; regen draws replace until 8 valids.
    batch_invalid = _invalid_import_code()
    completions = [batch_invalid]
    completions += [_invalid_import_code()]
    completions += [_valid_code(float(i)) for i in range(1, 9)]
    client = ScriptedLLMClient(completions)
    import tempfile
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "stage1_rejections.jsonl")
        evolver = StageIEvolver(
            client,
            score_fn=_score_by_smoke,
            config=StageIConfig(
                population_size=8,
                generations=0,
                n_crossover=2,
                n_mutation=4,
                n_random=2,
                max_invalid_replacements=8,
            ),
            rejection_log_path=log_path,
        )
        final = evolver.run()
        assert len(final) == 8
        assert all(c.valid for c in final)
        assert os.path.isfile(log_path)
        rows = [json.loads(line) for line in open(log_path, encoding="utf-8")]
        rejected = [r for r in rows if not r["accepted"]]
        assert len(rejected) >= 2
        assert all(r["rejection_category"] == "forbidden_import" for r in rejected)


def test_mutation_uses_lower_performer_and_crossover_uses_top2():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    completions = [batch_body]
    # Track prompts indirectly via distinct next-gen codes.
    completions += [_valid_code(100.0)] * 2  # crossover slots
    completions += [_valid_code(50.0)] * 4  # mutation slots
    completions += [_valid_code(10.0)] * 2  # random slots
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(
            population_size=8,
            generations=1,
            keep_runtime_elite=True,
        ),
    )
    evolver.run()
    gen1 = evolver.history[1].population
    assert sum(1 for c in gen1 if c.origin == "crossover") == 2
    assert sum(1 for c in gen1 if c.origin == "mutation") == 4
    assert sum(1 for c in gen1 if c.origin == "random") == 1
    assert any(
        candidate.candidate_id == evolver.history[0].population[0].candidate_id
        for candidate in gen1
    )
    for c in gen1:
        if c.origin == "crossover":
            assert len(c.parent_ids) == 2
        if c.origin == "mutation":
            assert len(c.parent_ids) == 1


def test_next_generation_exact_242_without_runtime_elite():
    batch_body = "\n\n".join(_valid_code(float(i)) for i in range(8, 0, -1))
    completions = [batch_body]
    completions += [_valid_code(100.0)] * 2
    completions += [_valid_code(50.0)] * 4
    completions += [_valid_code(10.0)] * 2
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(
            population_size=8,
            generations=1,
            keep_runtime_elite=False,
        ),
    )
    evolver.run()
    gen1 = evolver.history[1].population
    assert sum(1 for c in gen1 if c.origin == "crossover") == 2
    assert sum(1 for c in gen1 if c.origin == "mutation") == 4
    assert sum(1 for c in gen1 if c.origin == "random") == 2
    assert len(gen1) == 8
