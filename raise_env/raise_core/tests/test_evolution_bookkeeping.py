"""Shared retry budget, history-based fitness trend, diversity metric."""

from __future__ import annotations

from domains.highway import prompts as hwy_prompts
from domains.highway.reward_checks import make_highway_validator
from raise_core.candidate import RewardCandidate
from raise_core.explore import StageIConfig, StageIEvolver
from raise_core.feedback import FeedbackCardConfig, component_diversity
from raise_core.llm import LLMClient
from raise_core.raise_loop import ClosedLoopConfig, ClosedLoopRunner
from raise_core.raise_loop.checkpoint import load_checkpoint
from raise_core.raise_loop.logging_io import read_epochs


class _InvalidLLM(LLMClient):
    provider = "scripted"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, *, max_tokens=None):
        self.calls += 1
        return "```python\nimport os\ndef compute_reward(state, memory):\n    return 0.0\n```\n"


def _evaluated(cid: str, value: float, fitness: float) -> RewardCandidate:
    code = f"def compute_reward(state, memory):\n    return float(state.progress) + {value}\n"
    return RewardCandidate(cid, code, valid=True, origin="initial",
                           metadata={"fitness": fitness, "last_metrics": {"SR": 0.5}})


def test_generation_shares_one_retry_budget():
    llm = _InvalidLLM()
    evo = StageIEvolver(
        llm,
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_highway_validator(),
        config=StageIConfig(population_size=4, max_invalid_replacements=3),
        prompts=hwy_prompts,
        feedback=FeedbackCardConfig(enabled=False),
    )
    ranked = [_evaluated(f"ini_000{i}", float(i), 1.0 - 0.1 * i) for i in range(4)]
    gen1 = evo._next_generation(ranked)
    assert len(gen1) == 4
    # Crossover: 1 + 3 retries; mutation and random get their single draw only.
    assert llm.calls == 4 + 1 + 1
    assert {c.origin for c in gen1[1:]} == {"crossover_fallback", "mutation_fallback", "seed_fallback"}


def test_reflection_trend_comes_from_the_given_history():
    evo = StageIEvolver(
        _InvalidLLM(),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_highway_validator(),
        prompts=hwy_prompts,
    )
    note = evo._build_reflection([_evaluated("a", 1.0, 0.5)], generation=3,
                                 fitness_trend=[0.1, None, float("-inf"), 0.2])
    assert note.startswith("Gen3 best_fitness_trend=[0.100, 0.200, 0.500]")


def test_run_reflection_trend_matches_epoch_history(tmp_path):
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(population_size=4, generations=2, stage2_train_steps=32,
                           use_stub=True, output_dir=out, device="cpu",
                           feedback=FeedbackCardConfig(enabled=False))
    ClosedLoopRunner(cfg).run()
    best = [rec["evolve_best_fitness"] for rec in read_epochs(out)]
    reflection = load_checkpoint(out)["reflection"]
    last_note = reflection.split(" || ")[-1]
    assert last_note.startswith(f"Gen1 best_fitness_trend=[{best[0]:.3f}, {best[1]:.3f}]")


def _with_components(*keys: str) -> str:
    items = ", ".join(f"'{k}': 1.0" for k in keys)
    return (
        "def compute_reward(state, memory):\n"
        f"    parts = {{{items}}}\n"
        "    return sum(parts.values())\n"
    )


def test_component_diversity_is_mean_pairwise_jaccard_distance():
    a = _with_components("speed", "collision")
    b = _with_components("speed", "lane")
    assert component_diversity([a, a]) == 0.0
    # |{speed}| / |{speed, collision, lane}| = 1/3 shared.
    assert abs(component_diversity([a, b]) - (1 - 1 / 3)) < 1e-12
    assert component_diversity([a]) is None
    no_dict = "def compute_reward(state, memory):\n    return 0.0\n"
    assert component_diversity([no_dict, no_dict]) == 0.0


def test_epochs_record_component_diversity(tmp_path):
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(population_size=4, generations=1, stage2_train_steps=32,
                           use_stub=True, output_dir=out, device="cpu",
                           feedback=FeedbackCardConfig(enabled=False))
    ClosedLoopRunner(cfg).run()
    value = read_epochs(out)[0]["component_diversity"]
    assert value is not None and 0.0 <= value <= 1.0
