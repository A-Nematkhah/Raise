"""Next-generation operator counts: defaults, overrides, enforcement, records."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from domains.crowdnav import prompts as cn_prompts
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import StageIConfig, StageIEvolver, operator_counts
from raise_core.feedback import FeedbackCardConfig
from raise_core.llm import ScriptedLLMClient
from raise_core.pipeline import RaiseRunConfig
from raise_core.raise_loop import ClosedLoopConfig, ClosedLoopRunner
from raise_core.raise_loop.logging_io import read_epochs
from raise_core.refine import StubPolicyTrainer

_RAISE_ROOT = Path(__file__).resolve().parents[2]


def _counts(n, elitism, **given):
    ops = operator_counts(n, elitism=elitism, **given)
    return (ops.elite, ops.crossover, ops.mutation, ops.random)


@pytest.mark.parametrize(
    "n, elitism, expected",
    [
        (6, True, (1, 2, 2, 1)),
        (6, False, (0, 2, 3, 1)),
        (4, True, (1, 1, 1, 1)),
        (4, False, (0, 1, 2, 1)),
        (8, False, (0, 3, 4, 1)),
        (2, True, (1, 0, 1, 0)),
        (2, False, (0, 1, 1, 0)),
    ],
)
def test_default_split(n, elitism, expected):
    assert _counts(n, elitism) == expected


def test_first_unset_count_absorbs_the_remainder():
    assert _counts(6, True, n_random=0) == (1, 2, 3, 0)
    assert _counts(6, True, n_crossover=1) == (1, 1, 2, 2)
    assert _counts(8, True, n_crossover=2, n_mutation=4) == (1, 2, 4, 1)
    assert _counts(6, True, n_crossover=1, n_mutation=1, n_random=3) == (1, 1, 1, 3)


@pytest.mark.parametrize(
    "n, elitism, given",
    [
        (6, True, dict(n_crossover=2, n_mutation=2, n_random=2)),
        (6, True, dict(n_crossover=4, n_mutation=4)),
        (1, True, {}),
    ],
)
def test_counts_that_do_not_add_up_raise(n, elitism, given):
    with pytest.raises(ValueError):
        operator_counts(n, elitism=elitism, **given)


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float(state.progress) + {value}\n"
        "```\n"
    )


def test_default_generation_at_n6_has_every_operator():
    completions = ["\n\n".join(_valid_code(float(i)) for i in range(6, 0, -1))]
    completions += [_valid_code(100.0 + i) for i in range(5)]
    evo = StageIEvolver(
        ScriptedLLMClient(completions),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_crowdnav_validator(),
        config=StageIConfig(population_size=6, keep_runtime_elite=True),
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(enabled=False),
    )
    ranked = evo.score_population(evo.initialize_population())
    gen1 = evo._next_generation(ranked)
    origins = [c.origin for c in gen1[1:]]
    assert gen1[0] is ranked[0]
    assert sorted(origins) == ["crossover", "crossover", "mutation", "mutation", "random"]


def test_runner_rejects_counts_that_do_not_add_up(tmp_path):
    # Not use_stub: the fast profile would reset the counts for its smaller N.
    cfg = ClosedLoopConfig(population_size=6, generations=1, output_dir=str(tmp_path / "run"),
                           n_crossover=4, n_mutation=4)
    runner = ClosedLoopRunner(cfg, score_fn=lambda *_a, **_k: 0.0, trainer=StubPolicyTrainer())
    with pytest.raises(ValueError, match="do not add up"):
        runner.run()


def test_run_records_configured_and_actual_operators(tmp_path):
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(population_size=4, generations=2, stage2_train_steps=32,
                           use_stub=True, output_dir=out, device="cpu",
                           feedback=FeedbackCardConfig(enabled=False))
    result = ClosedLoopRunner(cfg).run()
    assert result.manifest["operators"] == {"elite": 1, "crossover": 1, "mutation": 1, "random": 1}
    gen0, gen1 = read_epochs(out)
    assert gen0["n_carried"] == 0 and sum(gen0["origin_counts"].values()) == 4
    assert gen1["n_carried"] >= 1
    assert gen1["n_carried"] + sum(gen1["origin_counts"].values()) == 4
    assert set(gen1["origin_counts"]) <= {
        "crossover", "mutation", "random", "crossover_fallback", "mutation_fallback",
        "seed_fallback",
    }


def test_pipeline_config_passes_counts_through():
    cfg = RaiseRunConfig(population=6, n_random=0)
    assert cfg.operators().to_dict() == {"elite": 1, "crossover": 2, "mutation": 3, "random": 0}


def test_cli_rejects_counts_that_do_not_add_up(monkeypatch, capsys):
    script = _RAISE_ROOT / "scripts" / "run_raise.py"
    monkeypatch.chdir(_RAISE_ROOT)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--allow-seed-llm", "--population", "6",
        "--n-crossover", "4", "--n-mutation", "4", "--output-dir", "results/_bad_ops",
    ])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 2
    assert "do not add up" in capsys.readouterr().err
    assert not (_RAISE_ROOT / "results" / "_bad_ops").exists()
