"""Fast end-to-end pipeline tests (stub trainers, seed LLM)."""

from __future__ import annotations

import json
import os

import torch

from raise_core import pipeline as pipeline_mod
from raise_core.domains import make_stage3_trainer
from raise_core.evo_selection.config import SelectionConfig
from raise_core.evo_selection.llm_ranker import LLMRanker, RankResult
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
from raise_core.refine import Stage2Config
from raise_core.validate import Stage3Config


def test_pipeline_fast(tmp_path):
    out = tmp_path / "run"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    assert arts.best_stage1 is not None
    assert arts.best_stage2 is not None
    assert arts.best_stage3 is not None
    for name in (
        "config.json",
        "manifest.json",
        "best_stage1.json",
        "best_stage2.json",
        "best_stage3.json",
        "final_candidate.json",
        "stage2_population.json",
        "stage3_population.json",
        "seed_reward.py",
        "closed_loop/epochs.jsonl",
        "closed_loop/REPORT.txt",
        "stage3/checkpoint.json",
    ):
        assert (out / name).is_file(), name
    final = json.loads((out / "final_candidate.json").read_text(encoding="utf-8"))
    assert "compute_reward" in final["code"]
    assert final["valid"] is True
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["domain"] == "crowdnav"
    assert man["best_stage3_id"] == arts.best_stage3.candidate_id
    s2 = json.loads((out / "stage2_population.json").read_text(encoding="utf-8"))
    assert s2["history"], "per-generation Stage II rows feed the plots"
    # Stage III best is the highest Stage III fitness.
    fits = [(c.metadata or {})["fitness"] for c in arts.stage3_population]
    assert (arts.best_stage3.metadata or {})["fitness"] == max(fits)


def test_pipeline_llm_rank_selects_stage3_winner_without_api_calls(
    tmp_path, monkeypatch
):
    stages = []

    def rank(_self, cards, *, fallback_order, stage, **kwargs):
        stages.append(stage)
        ranking = list(reversed(fallback_order))
        return RankResult(
            ranking=ranking,
            scores={candidate_id: 0.5 for candidate_id in ranking},
            weaknesses={candidate_id: [] for candidate_id in ranking},
            rationale={
                candidate_id: "Balances completion and safety."
                for candidate_id in ranking
            },
        )

    monkeypatch.setattr(LLMRanker, "rank", rank)
    cfg = RaiseRunConfig(output_dir=str(tmp_path / "llm_rank_pipeline"))
    cfg.apply_fast_profile()
    cfg.selection = SelectionConfig(
        mode="llm_rank",
        top_k=1,
        n_votes=1,
        p_restart=0.0,
        seed=cfg.seed,
    )
    artifacts = RaisePipeline(cfg).run()

    assert stages == ["stage2", "stage2", "stage3"]
    assert artifacts.best_stage3.candidate_id == artifacts.manifest["best_stage3_id"]
    assert artifacts.manifest["stage3_selection"]["best_id"] == artifacts.best_stage3.candidate_id
    # LLM may reorder only within {measured best} ∪ ties; a full reverse
    # outside that set must keep the measured Stage III best.
    assert artifacts.manifest["stage3_selection"]["llm_restricted_to_ties"] is True
    assert "llm_ranking" in artifacts.manifest["stage3_selection"]


def test_limit_cpu_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    before = torch.get_num_threads()
    try:
        pipeline_mod.limit_cpu_threads(0)
        assert os.environ["OMP_NUM_THREADS"] == "8"
        assert torch.get_num_threads() == before
        pipeline_mod.limit_cpu_threads(1)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


class _MetadataReassigningStage3:
    """Stub Stage III that reassigns candidate.metadata like CrowdNavPPOTrainer."""

    def __init__(self, inner, seen=None):
        self.inner = inner
        self.seen = seen if seen is not None else []

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        self.seen.append(config)
        out = self.inner.train_and_eval(
            candidate, round_index=round_index, config=config, **kwargs
        )
        md = dict(candidate.metadata or {})
        md["fitness"] = -123.0
        md["last_metrics"] = {"SR": 0.0, "CR": 1.0, "TR": 0.0, "fitness": -123.0}
        candidate.metadata = md
        return out


def test_stage3_does_not_mutate_stage2_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _MetadataReassigningStage3(
            make_stage3_trainer(pack, use_stub=use_stub)
        ),
    )
    out = tmp_path / "run_hw"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    on_disk = json.loads((out / "best_stage2.json").read_text(encoding="utf-8"))
    mem_md = arts.best_stage2.metadata or {}
    disk_md = on_disk.get("metadata") or {}
    assert mem_md.get("fitness") == disk_md.get("fitness")
    assert mem_md.get("fitness") != -123.0
    assert mem_md.get("last_metrics") == disk_md.get("last_metrics")


def test_stage3_receives_trainer_config(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _MetadataReassigningStage3(
            make_stage3_trainer(pack, use_stub=use_stub), seen
        ),
    )
    cfg = RaiseRunConfig(output_dir=str(tmp_path / "run_hw_cfg"))
    cfg.apply_fast_profile()
    cfg.eval_mode = "holdout_only"
    cfg.warm_start = True
    cfg.n_envs = 2
    cfg.robot_kinematics = "unicycle"
    cfg.action_mode = "continuous"
    RaisePipeline(cfg).run()
    assert seen
    s3 = seen[0]
    assert s3.eval_mode == "holdout_only"
    # Stage II warm start does not leak into Stage III's K3 comparison.
    assert s3.warm_start is False
    assert s3.n_envs == 2
    assert s3.train_env_steps == cfg.stage3_train_steps
    assert s3.ppo_n_steps == 256
    assert s3.eval_deterministic is False
    assert (s3.robot_kinematics, s3.action_mode) == ("unicycle", "continuous")
    assert s3.eval_split == "val"


def test_warm_start_off_by_default():
    assert RaiseRunConfig().warm_start is False
    assert RaiseRunConfig().stage3_warm_start is False
    assert Stage2Config().warm_start is False
    assert Stage3Config().warm_start is False


def test_resume_preserves_original_config_and_created_utc(tmp_path):
    out = tmp_path / "run_resume_cfg"
    cfg1 = RaiseRunConfig(output_dir=str(out), seed=111)
    cfg1.apply_fast_profile()
    arts1 = RaisePipeline(cfg1).run()
    created = arts1.manifest["created_utc"]
    original = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert original["seed"] == 111

    cfg2 = RaiseRunConfig(
        output_dir=str(out), seed=222, resume=True, allow_config_drift=True
    )
    cfg2.apply_fast_profile()
    arts2 = RaisePipeline(cfg2).run()
    kept = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert kept["seed"] == 111
    resumes = list(out.glob("config.resume_*.json"))
    assert resumes, "resume should write a stamped config snapshot"
    stamped = json.loads(resumes[0].read_text(encoding="utf-8"))
    assert stamped["seed"] == 222

    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["created_utc"] == created
    assert man["config"]["seed"] == 111
    assert isinstance(man.get("resumes"), list) and man["resumes"]
    assert man["resumes"][-1]["argv"]
    assert arts2.manifest["created_utc"] == created
