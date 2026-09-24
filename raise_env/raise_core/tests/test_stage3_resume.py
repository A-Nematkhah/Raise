"""Stage III crash-safe resume (stub trainer — no real PPO)."""

from __future__ import annotations

import os
from pathlib import Path

from raise_core.explore import RewardCandidate
from raise_core.llm import ScriptedLLMClient
from raise_core.sandbox import RewardValidator
from raise_core.stage3_checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    stage3_dir_from_train_root,
)
from raise_core.validate import Stage3Config, Stage3Runner, StubPolicyTrainer


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float({value})\n"
        "```\n"
    )


def _make_population(n: int = 3) -> list:
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


def test_stage3_resume_skips_finished_candidates(tmp_path: Path):
    n = 3
    pop = _make_population(n)
    out_root = tmp_path / "stage3_train"
    scripts = [_valid_code(10.0 + i) for i in range(24)]
    cfg = Stage3Config(
        population_size=n,
        rounds=1,
        train_env_steps=100,
        eval_episodes=2,
        human_counts=(5,),
        output_root=str(out_root),
        resume=True,
        protect_elite_refine=False,
    )

    # Complete first two candidates, then write a mid-round checkpoint.
    runner = Stage3Runner(ScriptedLLMClient(scripts), StubPolicyTrainer(), config=cfg)
    full = runner.run_round(pop, round_index=0)
    done_two = full[:2]
    hist = runner._history_payload[:2]
    runner.history = runner.history[:2]
    runner._history_payload = list(hist)
    ckpt_dir = stage3_dir_from_train_root(str(out_root))
    save_checkpoint(
        ckpt_dir,
        build_checkpoint(
            status="running",
            phase="training",
            round_index=0,
            candidate_index=2,
            rounds=1,
            round_input_population=pop,
            round_output_partial=done_two,
            history=hist,
            best_trained=runner.best_trained,
            trained_snapshots=runner.trained_snapshots[:2],
            run_h_sweep=True,
            config={"output_root": str(out_root)},
        ),
    )

    resumed = Stage3Runner(
        ScriptedLLMClient(scripts), StubPolicyTrainer(), config=cfg
    )
    final = resumed.run(pop, run_h_sweep=True)
    assert len(final) == n
    assert len(resumed.history) == n
    assert [r.candidate_id for r in resumed.history] == ["c0", "c1", "c2"]
    done = load_checkpoint(ckpt_dir)
    assert done is not None
    assert done.get("status") == "done"
    assert os.path.isfile(os.path.join(ckpt_dir, "RESUME.json"))


def test_stage3_resume_disabled_starts_fresh(tmp_path: Path):
    n = 2
    pop = _make_population(n)
    out_root = tmp_path / "stage3_train"
    scripts = [_valid_code(1.0) for _ in range(16)]
    cfg = Stage3Config(
        population_size=n,
        rounds=1,
        human_counts=(5,),
        output_root=str(out_root),
        resume=True,
        protect_elite_refine=False,
    )
    Stage3Runner(ScriptedLLMClient(scripts), StubPolicyTrainer(), config=cfg).run(
        pop, run_h_sweep=False
    )
    cfg2 = Stage3Config(
        population_size=n,
        rounds=1,
        human_counts=(5,),
        output_root=str(out_root),
        resume=False,
        protect_elite_refine=False,
    )
    runner = Stage3Runner(
        ScriptedLLMClient(scripts), StubPolicyTrainer(), config=cfg2
    )
    out = runner.run(pop, run_h_sweep=False)
    assert len(out) == n
    assert len(runner.history) == n
