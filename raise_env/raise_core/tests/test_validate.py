"""
Fast Stage III tests using StubPolicyTrainer (no real RL, no LLM).

Run: pytest raise_core/tests/test_validate.py -q
"""

from __future__ import annotations

from pathlib import Path

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import RewardCandidate
from raise_core.validate import (
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
    StubPolicyTrainer,
)


def _make_population(n: int = 2) -> list:
    validator = make_crowdnav_validator()
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
                    "training_dynamics": {
                        "source": "stage2",
                        "total_timesteps": 150_000,
                    },
                    "warm_start_checkpoint": "stage2_parent.zip",
                    "eval_generation": 3,
                    "fitness_history": [{"generation": 2, "fitness": 8.0}],
                    "duplicate_of": None,
                    "llm_fallback_clone": False,
                },
            )
        )
    return pop


def test_defaults_and_k3_constant():
    from raise_config import CONFIG

    cfg = Stage3Config()
    assert cfg.train_env_steps == STAGE3_STEPS == int(CONFIG.budget.stage3.train_steps)
    assert cfg.eval_episodes == 30
    assert cfg.resume is True


def test_stage3_trains_each_candidate_once_and_picks_best_by_fitness(tmp_path: Path):
    pop = _make_population(3)
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
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
        # Stale Stage II training/warm-start fields are stripped (P0-1).
        assert "training_dynamics" not in md
        assert "warm_start_checkpoint" not in md
        assert "eval_generation" not in md
        assert "fitness_history" not in md
        assert "duplicate_of" not in md
        assert "llm_fallback_clone" not in md
        assert md["stage2_training_dynamics"]["total_timesteps"] == 150_000
        assert md["stage2_warm_start_checkpoint"] == "stage2_parent.zip"
        assert md["stage2_eval_generation"] == 3


def test_stage3_input_preserves_stage2_aliases_without_leaking():
    from raise_core.validate import _stage3_input

    cand = _make_population(1)[0]
    out = _stage3_input(cand)
    md = out.metadata
    assert md["stage2_fitness"] == 9.0
    assert "fitness" not in md or md.get("fitness") is None
    assert "last_metrics" not in md
    assert "checkpoint_path" not in md
    assert "training_dynamics" not in md
    assert md["stage2_training_dynamics"]["source"] == "stage2"
    assert md["stage2_warm_start_checkpoint"] == "stage2_parent.zip"


def test_stage3_skips_collapsed_candidates(tmp_path: Path):
    pop = _make_population(2)
    pop[0].metadata["last_metrics"] = {"holdout": {"CR": 0.95, "SR": 0.0, "TR": 0.05}}
    pop[0].metadata["fitness"] = -0.2
    pop[1].metadata["last_metrics"] = {"holdout": {"CR": 0.3, "SR": 0.6, "TR": 0.1}}
    pop[1].metadata["fitness"] = 0.05
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "stage3_train"),
            resume=False,
            skip_collapsed=True,
            collapse_cr=0.8,
        ),
    )
    out = runner.run(pop)
    assert [c.candidate_id for c in out] == ["c1"]
    skipped = [r for r in runner.history if r.status == "skipped_collapsed"]
    assert [r.candidate_id for r in skipped] == ["c0"]


def test_stage3_all_collapsed_trains_everyone(tmp_path: Path):
    pop = _make_population(2)
    for c in pop:
        c.metadata["last_metrics"] = {"holdout": {"CR": 0.9, "SR": 0.0, "TR": 0.1}}
        c.metadata["fitness"] = -0.2
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "stage3_all"),
            resume=False,
            skip_collapsed=True,
            collapse_cr=0.8,
        ),
    )
    out = runner.run(pop)
    assert {c.candidate_id for c in out} == {"c0", "c1"}
    assert not any(r.status == "skipped_collapsed" for r in runner.history)
