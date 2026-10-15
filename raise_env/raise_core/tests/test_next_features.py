"""NEXT.md features: multi-seed Stage III, top-k, tied blind eval, mutation filter, SH."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.explore import (
    RewardCandidate,
    StageIConfig,
    StageIEvolver,
    filter_mutation_pool,
    select_mutation_parents,
)
from raise_core.llm import ScriptedLLMClient
from raise_core.pipeline import RaiseRunConfig, _eval_tied_finalists
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import _RESUME_CONFIG_KEYS
from raise_core.validate import Stage3Config, Stage3Runner, StubPolicyTrainer


def _cand(cid: str, fitness: float, cr: float, code_i: int = 0) -> RewardCandidate:
    validator = make_crowdnav_validator()
    code = f"def compute_reward(state, memory):\n    return float({code_i}.0)\n"
    fn, err = validator.try_validate(code)
    assert fn is not None, err
    return RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=fn,
        valid=True,
        origin="initial",
        metadata={
            "fitness": fitness,
            "last_metrics": {
                "fitness": fitness,
                "holdout": {"CR": cr, "SR": max(0.0, 1.0 - cr), "TR": 0.0},
                "CR": cr,
                "holdout_episodes": [
                    {"case": i, "task_return": fitness} for i in range(4)
                ],
            },
        },
    )


def test_filter_mutation_pool_drops_collapsed_and_falls_back():
    pool = [
        _cand("ok", 0.1, 0.2, 0),
        _cand("bad", -0.2, 0.95, 1),
        _cand("ok2", 0.05, 0.3, 2),
    ]
    kept = filter_mutation_pool(pool, exclude_collapsed=True, collapse_cr=0.8)
    assert [c.candidate_id for c in kept] == ["ok", "ok2"]
    all_bad = [_cand("a", -0.2, 0.9, i) for i in range(3)]
    assert filter_mutation_pool(all_bad, exclude_collapsed=True) == all_bad
    assert filter_mutation_pool(pool, exclude_collapsed=False) == pool


def test_mutation_exclude_collapsed_changes_parent_picks():
    ranked = [
        _cand("top", 0.5, 0.1, 0),
        _cand("mid", 0.2, 0.2, 1),
        _cand("collapsed_low", -0.2, 0.95, 2),
        _cand("also_low", 0.0, 0.3, 3),
    ]
    # Without filter, lower half includes collapsed_low.
    raw = select_mutation_parents(
        ranked, 4, mode="lower", rng=random.Random(0)
    )
    assert any(c.candidate_id == "collapsed_low" for c in raw)
    filtered = filter_mutation_pool(ranked, exclude_collapsed=True, collapse_cr=0.8)
    picks = select_mutation_parents(
        filtered, 4, mode="lower", rng=random.Random(0)
    )
    assert all(c.candidate_id != "collapsed_low" for c in picks)


def test_stage3_top_k_keeps_best_stage2(tmp_path: Path):
    pop = [
        _cand("weak", 0.01, 0.2, 0),
        _cand("strong", 0.20, 0.2, 1),
        _cand("mid", 0.10, 0.2, 2),
    ]
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "s3"),
            resume=False,
            skip_collapsed=False,
            top_k=2,
        ),
    )
    out = runner.run(pop)
    assert {c.candidate_id for c in out} == {"strong", "mid"}
    skipped = [r for r in runner.history if r.status == "skipped_top_k"]
    assert [r.candidate_id for r in skipped] == ["weak"]


def test_stage3_multi_seed_stores_runs_and_mean(tmp_path: Path):
    pop = [_cand("only", 0.1, 0.2, 3)]
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "s3m"),
            resume=False,
            skip_collapsed=False,
            n_seeds=3,
        ),
    )
    out = runner.run(pop)
    assert len(out) == 1
    md = out[0].metadata
    assert md["stage3_n_seeds"] == 3
    assert len(md["stage3_seed_runs"]) == 3
    assert all(r["status"] == "ok" for r in md["stage3_seed_runs"])
    fits = [r["fitness"] for r in md["stage3_seed_runs"]]
    assert md["fitness"] == sum(fits) / 3
    assert runner.selection_summary()["n_seeds"] == 3


def test_eval_tied_finalists_report_only(tmp_path: Path):
    pop = [_cand("w", 0.2, 0.2, 0), _cand("t", 0.19, 0.2, 1)]
    for c in pop:
        c.metadata["checkpoint_path"] = None
        c.metadata["stage3_trained"] = True
        c.metadata["fitness"] = float(c.metadata["fitness"])
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            train_env_steps=8,
            eval_episodes=2,
            output_root=str(tmp_path / "tie"),
            resume=False,
        ),
    )
    # Pretend both were trained; stub reevaluate works without a zip.
    runner.trained = list(pop)
    runner.best_trained = pop[0]
    winner_fe = {
        "candidate_id": "w",
        "status": "ok",
        "fitness": 0.25,
        "episodes": 4,
        "metrics": {"SR": 0.5, "CR": 0.3, "TR": 0.2, "fitness": 0.25},
    }
    out = _eval_tied_finalists(
        runner,
        stage3_pop=pop,
        best=pop[0],
        selection={"tied_ids": ["t"]},
        winner_final_eval=winner_fe,
        episodes=4,
        seed_base=99,
        enabled=True,
    )
    assert out is not None
    assert out["note"].startswith("Report only")
    assert out["winner_id"] == "w"
    assert {e["candidate_id"] for e in out["evaluations"]} == {"w", "t"}
    assert out["evaluations"][0]["role"] == "winner"
    # Selection winner unchanged.
    assert runner.best_trained.candidate_id == "w"


def test_resume_keys_include_next_md_fields():
    assert "mutation_exclude_collapsed" in _RESUME_CONFIG_KEYS
    assert "stage2_successive_halving" in _RESUME_CONFIG_KEYS
    assert ClosedLoopConfig().stage2_successive_halving is False
    assert ClosedLoopConfig().mutation_exclude_collapsed is False
    assert RaiseRunConfig().stage3_seeds == 1
    assert RaiseRunConfig().stage3_top_k is None
    assert RaiseRunConfig().eval_tied_finalists is True
    snap = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(n_seeds=2, top_k=3),
    )._config_snapshot()
    assert snap["n_seeds"] == 2
    assert snap["top_k"] == 3


def test_stagei_config_accepts_mutation_exclude():
    cfg = StageIConfig(mutation_exclude_collapsed=True, collapse_cr=0.8)
    assert cfg.mutation_exclude_collapsed is True


def test_successive_halving_marks_survivors(tmp_path: Path):
    from raise_core.feedback import FeedbackCardConfig
    from raise_core.raise_loop import ClosedLoopRunner
    from raise_core.refine import StubPolicyTrainer

    out = str(tmp_path / "sh")
    cfg = ClosedLoopConfig(
        population_size=4,
        generations=1,
        stage2_train_steps=30,
        use_stub=True,
        output_dir=out,
        device="cpu",
        stage2_successive_halving=True,
        feedback=FeedbackCardConfig(enabled=False),
    )
    result = ClosedLoopRunner(cfg, trainer=StubPolicyTrainer()).run()
    # At least one candidate should carry successive_halving metadata.
    marked = [
        c
        for c in result.population
        if isinstance((c.metadata or {}).get("successive_halving"), dict)
    ]
    assert marked, "expected phase-2 survivors to be marked"
    sh = marked[0].metadata["successive_halving"]
    assert sh["phase1_steps"] == 10
    assert sh["phase2_steps"] == 20
    assert sh["k2"] == 30
    assert marked[0].metadata.get("train_steps") == 30
