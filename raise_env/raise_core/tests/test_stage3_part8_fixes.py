"""Stage III blind eval seeds, tie reporting, final eval, drift and CLI wiring."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core import pipeline as pipeline_mod
from raise_core import validate as validate_mod
from raise_core.candidate import RewardCandidate
from raise_core.domains import make_stage3_trainer_for_domain
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
from raise_core.selection import candidate_fitness_se, statistical_ties
from raise_core.validate import (
    FINAL_EVAL_SEED_OFFSET,
    STAGE3_EVAL_SEED_OFFSET,
    Stage3Config,
    Stage3Runner,
    StubPolicyTrainer,
)

_RAISE_ROOT = Path(__file__).resolve().parents[2]


def _cand(cid: str, fitness: float | None = None, se: float | None = None) -> RewardCandidate:
    code = f"def compute_reward(state, memory):\n    return float({len(cid)}.0)\n"
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is not None, err
    md = {}
    if fitness is not None:
        md = {"fitness": fitness, "last_metrics": {"fitness": fitness, "fitness_se": se}}
    return RewardCandidate(
        candidate_id=cid, code=code, reward_fn=reward_fn, valid=True, metadata=md
    )


class _Recording:
    """Wraps the Stage III stub and records every config it sees."""

    def __init__(self, inner, *, fail_train: bool = False, fail_reeval: bool = False):
        self.inner = inner
        self.fail_train = fail_train
        self.fail_reeval = fail_reeval
        self.train_configs = []
        self.reeval_configs = []

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        self.train_configs.append(config)
        if self.fail_train:
            raise RuntimeError("PPO diverged")
        return self.inner.train_and_eval(candidate, round_index=round_index, config=config)

    def reevaluate(self, candidate, *, round_index, config):
        self.reeval_configs.append(config)
        if self.fail_reeval:
            raise FileNotFoundError("model.zip missing")
        return self.inner.reevaluate(candidate, round_index=round_index, config=config)


def _run_fast(tmp_path, monkeypatch, **trainer_kwargs):
    holder = {}

    def factory(pack, use_stub=False):
        holder["t"] = _Recording(
            make_stage3_trainer_for_domain(pack, use_stub=use_stub), **trainer_kwargs
        )
        return holder["t"]

    monkeypatch.setattr(pipeline_mod, "make_stage3_trainer_for_domain", factory)
    out = tmp_path / "run"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return cfg, arts, manifest, holder["t"], out


def test_stage3_and_final_eval_use_unseen_seed_families(tmp_path, monkeypatch):
    cfg, arts, manifest, trainer, out = _run_fast(tmp_path, monkeypatch)
    stage3_base = cfg.seed + STAGE3_EVAL_SEED_OFFSET
    assert {c.eval_seed_base for c in trainer.train_configs} == {stage3_base}
    # Closed-loop generation g evaluates on seed + g * 1000.
    assert all(stage3_base != cfg.seed + g * 1000 for g in range(cfg.generations + 1))

    assert len(trainer.reeval_configs) == 1
    final_cfg = trainer.reeval_configs[0]
    assert final_cfg.eval_seed_base == cfg.seed + FINAL_EVAL_SEED_OFFSET
    assert final_cfg.eval_episodes == cfg.final_eval_episodes

    final = manifest["final_eval"]
    assert final["status"] == "ok"
    assert final["candidate_id"] == arts.best_stage3.candidate_id
    assert final["seed_base"] == cfg.seed + FINAL_EVAL_SEED_OFFSET
    assert "metrics" not in final
    on_disk = json.loads((out / "final_candidate.json").read_text(encoding="utf-8"))
    assert on_disk["metadata"]["final_eval"]["metrics"]["fitness"] == final["fitness"]

    sel = manifest["stage3_selection"]
    assert sel["status"] == "ok"
    assert sel["best_id"] == arts.best_stage3.candidate_id
    assert sel["eval_seed_base"] == stage3_base


def test_stage3_population_skips_duplicate_genomes():
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(resume=False),
    )
    first = _cand("aa")
    duplicate = _cand("bb")
    distinct = _cand("ccc")

    prepared = runner._prepare_population([first, duplicate, distinct])

    assert [candidate.candidate_id for candidate in prepared] == ["aa", "ccc"]
    skipped = [row for row in runner.history if row.status == "skipped_duplicate"]
    assert len(skipped) == 1
    assert skipped[0].candidate_id == "bb"
    assert skipped[0].metrics["duplicate_of"] == "aa"


def test_final_eval_keeps_the_selection_score(tmp_path, monkeypatch):
    _, arts, _, _, _ = _run_fast(tmp_path, monkeypatch)
    md = arts.best_stage3.metadata
    fits = [c.metadata["fitness"] for c in arts.stage3_population]
    assert md["fitness"] == max(fits)
    assert md["final_eval"]["status"] == "ok"


def test_failed_final_eval_is_recorded_not_raised(tmp_path, monkeypatch):
    _, arts, manifest, _, _ = _run_fast(tmp_path, monkeypatch, fail_reeval=True)
    assert manifest["final_eval"]["status"] == "failed"
    assert "FileNotFoundError" in manifest["final_eval"]["error"]
    assert manifest["best_stage3_id"] == arts.best_stage3.candidate_id


def test_all_failed_stage3_is_labelled(tmp_path, monkeypatch):
    _, arts, manifest, trainer, _ = _run_fast(tmp_path, monkeypatch, fail_train=True)
    assert manifest["stage3_selection"]["status"] == "all_failed"
    assert manifest["stage3_selection"]["n_failed"] == len(trainer.train_configs)
    assert manifest["best_stage3_id"] == manifest["best_stage2_id"]
    assert manifest["final_eval"] is None
    assert trainer.reeval_configs == []


def test_final_eval_disabled_with_zero_episodes(tmp_path):
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(output_root=str(tmp_path / "s3"), resume=False),
    )
    assert runner.final_eval(_cand("c0", 0.1), episodes=0, seed_base=1) is None


def test_stage3_warm_start_has_its_own_flag(tmp_path):
    cfg = RaiseRunConfig(output_dir=str(tmp_path / "run"))
    cfg.apply_fast_profile()
    cfg.stage3_warm_start = True
    assert RaisePipeline(cfg)._stage3_config().warm_start is True
    cfg.stage3_warm_start = False
    cfg.warm_start = True
    assert RaisePipeline(cfg)._stage3_config().warm_start is False


def test_statistical_ties_use_the_larger_se():
    best = _cand("best", -0.524, 0.196)
    near = _cand("near", -0.598, 0.190)
    far = _cand("far", -1.393, 0.113)
    untrained = _cand("none")
    assert statistical_ties([best, near, far, untrained], best) == ["near"]
    assert candidate_fitness_se(untrained) == 0.0
    # Without SE only exact ties count.
    a, b = _cand("a", 0.5), _cand("bb", 0.5)
    assert statistical_ties([a, b, _cand("ccc", 0.49)], a) == ["bb"]


def test_selection_summary_reports_ties(tmp_path):
    runner = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(output_root=str(tmp_path / "s3"), resume=False),
    )
    runner.trained = [_cand("best", -0.5, 0.2), _cand("near", -0.6, 0.2)]
    runner.best_trained = runner.trained[0]
    summary = runner.selection_summary()
    assert summary["tied_ids"] == ["near"]
    assert summary["fitness_se"] == 0.2


def test_resume_warns_when_eval_mode_changes(tmp_path, monkeypatch):
    out_root = str(tmp_path / "stage3_train")
    pop = [_cand("c0"), _cand("c11")]
    Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(output_root=out_root, eval_mode="both"),
    ).run(pop)

    warnings = []
    monkeypatch.setattr(
        validate_mod.console, "warn", lambda msg, **kw: warnings.append(msg)
    )
    Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=Stage3Config(output_root=out_root, eval_mode="holdout_only"),
    ).run(pop)
    assert any("eval_mode" in w for w in warnings)


def test_cli_exposes_stage3_and_robot_flags(monkeypatch, capsys):
    script = _RAISE_ROOT / "scripts" / "run_raise.py"
    monkeypatch.chdir(_RAISE_ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--print-config",
            "--domain",
            "crowdnav",
            "--robot-kinematics",
            "unicycle",
            "--action-mode",
            "continuous",
            "--stage3-warm-start",
            "--final-eval-episodes",
            "60",
        ],
    )
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        assert int(exc.code or 0) == 0
    out = capsys.readouterr().out
    assert "domain='crowdnav'" in out
    assert "robot_kinematics='unicycle'" in out
    assert "action_mode='continuous'" in out
    assert "stage3_warm_start=True" in out
    assert "final_eval_episodes=60" in out
