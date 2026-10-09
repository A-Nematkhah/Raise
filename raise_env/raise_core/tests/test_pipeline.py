"""Fast end-to-end pipeline tests (stub trainers, seed LLM)."""

from __future__ import annotations

import json

from raise_core import pipeline as pipeline_mod
from raise_core.domains import make_stage3_trainer_for_domain
from raise_core.pipeline import RaisePipeline, RaiseRunConfig


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
    assert man["domain"] == "highway"
    assert man["best_stage3_id"] == arts.best_stage3.candidate_id
    s2 = json.loads((out / "stage2_population.json").read_text(encoding="utf-8"))
    assert s2["history"], "per-generation Stage II rows feed the plots"
    # Stage III best is the highest Stage III fitness.
    fits = [(c.metadata or {})["fitness"] for c in arts.stage3_population]
    assert (arts.best_stage3.metadata or {})["fitness"] == max(fits)


class _MetadataReassigningStage3:
    """Stub Stage III that reassigns candidate.metadata like HighwayPPOTrainer."""

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
        "make_stage3_trainer_for_domain",
        lambda pack, use_stub=False: _MetadataReassigningStage3(
            make_stage3_trainer_for_domain(pack, use_stub=use_stub)
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


def test_stage3_receives_highway_config(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer_for_domain",
        lambda pack, use_stub=False: _MetadataReassigningStage3(
            make_stage3_trainer_for_domain(pack, use_stub=use_stub), seen
        ),
    )
    cfg = RaiseRunConfig(output_dir=str(tmp_path / "run_hw_cfg"))
    cfg.apply_fast_profile()
    cfg.highway_eval_mode = "holdout_only"
    cfg.highway_warm_start = False
    cfg.highway_n_envs = 2
    RaisePipeline(cfg).run()
    assert seen
    s3 = seen[0]
    assert s3.highway_eval_mode == "holdout_only"
    assert s3.highway_warm_start is False
    assert s3.highway_n_envs == 2
    assert s3.train_env_steps == cfg.stage3_train_steps
    assert s3.highway_n_steps == 64
    assert s3.highway_eval_deterministic is False
