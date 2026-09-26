"""
Fast Stage II tests using StubPolicyTrainer (no real RL).

Run: pytest raise_core/tests/test_refine.py -q
"""

from __future__ import annotations

import logging

from raise_core.explore import RewardCandidate
from raise_core.llm import ScriptedLLMClient
from raise_core.sandbox import RewardValidator
from raise_core.refine import (
    ProxyMetrics,
    Stage2Config,
    Stage2Runner,
    StubPolicyTrainer,
    _v2_candidate_id,
)


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


def _make_population(n: int = 2) -> list:
    validator = RewardValidator()
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


def test_proxy_metrics_feedback_is_raw_not_rankings():
    m = ProxyMetrics(sr=0.8, cr=0.1, tr=0.1, nt=12.0, pl=15.0, itr=20.0, sd=0.3)
    text = m.feedback_text()
    assert "SR=0.8000" in text
    assert "rank" not in text.lower()
    assert set(m.as_dict()) == {"SR", "CR", "TR", "NT", "PL", "ITR", "SD"}


def test_stub_trainer_is_deterministic():
    pop = _make_population(1)
    trainer = StubPolicyTrainer()
    cfg = Stage2Config(population_size=1, rounds=1)
    a = trainer.train_and_eval(pop[0], round_index=0, config=cfg)
    b = trainer.train_and_eval(pop[0], round_index=0, config=cfg)
    assert a.as_dict() == b.as_dict()


def test_stage2_run_refines_to_v2_with_stub():
    n = 2
    pop = _make_population(n)
    client = ScriptedLLMClient([_valid_code(10.0), _valid_code(11.0)])
    runner = Stage2Runner(
        client,
        StubPolicyTrainer(),
        config=Stage2Config(
            population_size=n,
            rounds=1,
            train_env_steps=8,
            eval_episodes=2,
            horizon_steps=5,
            protect_elite_refine=False,
        ),
    )
    out = runner.run(pop)
    assert len(out) == n
    assert runner.best_trained is not None
    assert len(runner.history) == n
    # With elitism, best-ever genome may be re-injected without a _v2 id.
    assert any(r.refined and not r.kept_previous for r in runner.history)
    assert any(c.candidate_id.endswith("_v2") or c.origin == "refinement" for c in out)
    assert runner.validation_failures == []


def test_stage2_protects_elite_from_refine():
    n = 2
    pop = _make_population(n)
    client = ScriptedLLMClient([_valid_code(99.0), _valid_code(100.0)])
    runner = Stage2Runner(
        client,
        StubPolicyTrainer(),
        config=Stage2Config(
            population_size=n,
            rounds=1,
            protect_elite_refine=True,
        ),
    )
    out = runner.run(pop)
    assert runner.best_trained is not None
    assert any((c.metadata or {}).get("refine_skipped_elite") for c in out)
    # Elite genome is preserved for final selection even if later rounds degrade.
    assert (runner.best_trained.metadata or {}).get("trained_snapshot") is True


def test_failed_refinement_keeps_previous_and_logs(caplog):
    pop = _make_population(1)
    client = ScriptedLLMClient([_invalid_import_code()])
    runner = Stage2Runner(
        client,
        StubPolicyTrainer(),
        config=Stage2Config(
            population_size=1, rounds=1, protect_elite_refine=False
        ),
    )
    with caplog.at_level(logging.WARNING):
        out = runner.run(pop)
    assert out[0].candidate_id == "c0"
    assert out[0].code == pop[0].code
    assert out[0].metadata.get("refine_kept_previous") is True
    assert len(runner.validation_failures) == 1
    assert runner.validation_failures[0]["kept_previous"] is True
    assert runner.history[0].kept_previous is True
    assert "keeping previous" in caplog.text.lower() or "rejected" in caplog.text.lower()


def test_llm_error_keeps_previous():
    class BoomClient:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("llm down")

        def generate(self, prompt: str, n: int = 1):
            raise RuntimeError("llm down")

    pop = _make_population(1)
    runner = Stage2Runner(
        BoomClient(),  # type: ignore[arg-type]
        StubPolicyTrainer(),
        config=Stage2Config(
            population_size=1, rounds=1, protect_elite_refine=False
        ),
    )
    out = runner.run(pop)
    assert out[0].candidate_id == "c0"
    assert "llm_error" in (out[0].metadata.get("refine_error") or "")
    assert runner.history[0].kept_previous


def test_v2_id_helper():
    assert _v2_candidate_id("c3") == "c3_v2"
    assert _v2_candidate_id("c3_v2") == "c3_v3"
    assert _v2_candidate_id("c3_v3") == "c3_v4"


def test_table5_defaults():
    cfg = Stage2Config()
    assert cfg.population_size == 8
    assert cfg.rounds == 16
    # Paper Table 5 / §4.3.2: K2=8000 gradient steps.
    assert cfg.train_env_steps == 8_000
    assert cfg.k2_unit == "gradient_steps"
    assert cfg.inject_elite is False
    assert cfg.protect_elite_refine is False
    assert cfg.eval_episodes == 50
    assert cfg.horizon_steps == 100
    assert cfg.algo == "a2c"
    assert cfg.num_processes is None  # resolved at train time


def test_stage2_argv_resolves_auto_num_processes():
    from raise_core.parallelism import resolve_num_processes
    from raise_core.refine import _stage2_train_argv

    cfg = Stage2Config(num_processes=None)
    argv = _stage2_train_argv(cfg, "c0", 0)
    assert "--num-processes" in argv
    n = int(argv[argv.index("--num-processes") + 1])
    assert n == resolve_num_processes(None)
    assert n >= 1

    cfg1 = Stage2Config(num_processes=1)
    argv1 = _stage2_train_argv(cfg1, "c0", 0)
    assert int(argv1[argv1.index("--num-processes") + 1]) == 1


def test_stage2_a2c_forces_num_mini_batch_one():
    """A2C/SRNN require matching mini-batch size (train.py sets this to 1)."""
    from raise_core.refine import _stage2_train_argv

    cfg = Stage2Config(algo="a2c", num_processes=4, num_mini_batch=None)
    argv = _stage2_train_argv(cfg, "c0", 0)
    assert int(argv[argv.index("--num-mini-batch") + 1]) == 1
