"""
Fast Stage III tests using StubPolicyTrainer (no real RL, no LLM).

Run: pytest raise_core/tests/test_validate.py -q
"""

from __future__ import annotations

from pathlib import Path

from domains.highway.reward_checks import make_highway_validator
from raise_core.explore import RewardCandidate
from raise_core.validate import (
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
    StubPolicyTrainer,
)


def _make_population(n: int = 2) -> list:
    validator = make_highway_validator()
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
                metadata={
                    "fitness": 9.0,
                    "last_metrics": {"SR": 1.0},
                    "checkpoint_path": "stage2.zip",
                },
            )
        )
    return pop


def test_defaults_and_k3_constant():
    cfg = Stage3Config()
    assert cfg.train_env_steps == STAGE3_STEPS == 70_000
    assert cfg.eval_episodes == 30
    assert cfg.resume is True


def test_stage3_trains_each_candidate_once_and_picks_best_by_fitness(tmp_path: Path):
    pop = _make_population(3)
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_highway_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "stage3_train"),
            resume=False,
        ),
    )
    out = runner.run(pop)
    assert [c.candidate_id for c in out] == ["c0", "c1", "c2"]
    assert len(runner.history) == 3
    best = max(out, key=lambda c: c.metadata["fitness"])
    assert runner.best_trained is not None
    assert runner.best_trained.candidate_id == best.candidate_id
    for c in out:
        md = c.metadata
        # Stage II results never leak into Stage III selection.
        assert md["stage2_fitness"] == 9.0
        assert md["fitness"] != 9.0
        assert md["checkpoint_path"] is None
        assert md["stage3_trained"] is True
