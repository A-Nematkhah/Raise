"""Fast tests for Algorithm 1 pipeline + reporting helpers."""

from __future__ import annotations

import json
import os

from raise_core import pipeline as pipeline_mod
from raise_core.domains import make_stage3_trainer_for_domain
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
from domains.crowdnav.reporting import (
    EpisodeRecord,
    format_table2_row,
    summarize_episodes,
    write_json,
)
from raise_core.scoring import make_smoke_score_fn
from domains.crowdnav.state import LegacyReward


def test_smoke_score_fn_finite():
    fn = make_smoke_score_fn()
    legacy = LegacyReward(
        success_reward=10,
        collision_penalty=-20,
        discomfort_dist=0.25,
        discomfort_penalty_factor=10,
        time_step=0.25,
        pot_factor=2.0,
    )
    score = fn(legacy, candidate_id="x")
    value = float(score)
    assert value == value  # not NaN
    assert value > float("-inf")
    assert 0.0 <= float(score.degenerate_fraction) <= 1.0


def test_summarize_episodes_mean_std():
    eps = []
    for seed in (1, 2):
        for k in range(4):
            eps.append(
                EpisodeRecord(
                    episode_index=k,
                    seed=seed,
                    outcome="success" if k < 3 else "collision",
                    success=1 if k < 3 else 0,
                    collision=0 if k < 3 else 1,
                    timeout=0,
                    nav_time=10.0,
                    path_length=12.0,
                    intrusion_ratio_pct=5.0,
                    min_dist_during_intrusion=0.3,
                    steps=40,
                    method="toy",
                    randomize=False,
                )
            )
    bundle = summarize_episodes(eps, method="toy", randomize=False)
    assert abs(bundle.sr.mean - 0.75) < 1e-9
    assert bundle.sr.n == 2  # two seeds
    assert "episodes" in bundle.to_dict()
    assert len(bundle.to_dict()["episodes"]) == 8
    assert "SR" in format_table2_row("toy", bundle)


def test_pipeline_fast(tmp_path):
    out = tmp_path / "run"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    assert arts.best_stage1 is not None
    assert arts.best_stage2 is not None
    assert arts.best_stage3 is not None
    for name in (
        "manifest.json",
        "best_stage1.json",
        "best_stage2.json",
        "final_candidate.json",
        "stage1_population.json",
        "stage2_population.json",
        "stage3_population.json",
        "seed_reward.py",
    ):
        assert (out / name).is_file(), name
    final = json.loads((out / "final_candidate.json").read_text(encoding="utf-8"))
    assert "compute_reward" in final["code"]
    assert final["valid"] is True
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man.get("surrogate", {}).get("enabled") is False


class _MetadataReassigningStage3:
    """Stub Stage III that reassigns candidate.metadata like HighwayPPOTrainer."""

    def __init__(self, inner):
        self.inner = inner

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        out = self.inner.train_and_eval(
            candidate, round_index=round_index, config=config, **kwargs
        )
        md = dict(candidate.metadata or {})
        md["fitness"] = -123.0
        md["last_metrics"] = {"SR": 0.0, "CR": 1.0, "TR": 0.0, "fitness": -123.0}
        candidate.metadata = md
        return out

    def evaluate_at_human_counts(self, candidate, bundle, *, config, human_counts=None):
        return self.inner.evaluate_at_human_counts(
            candidate, bundle, config=config, human_counts=human_counts
        )


def test_pipeline_fast_highway_stage3_keeps_stage2_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer_for_domain",
        lambda pack, use_stub=False: _MetadataReassigningStage3(
            make_stage3_trainer_for_domain(pack, use_stub=use_stub)
        ),
    )
    out = tmp_path / "run_hw"
    cfg = RaiseRunConfig(output_dir=str(out), domain="highway")
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    on_disk = json.loads((out / "best_stage2.json").read_text(encoding="utf-8"))
    mem_md = arts.best_stage2.metadata or {}
    disk_md = on_disk.get("metadata") or {}
    assert mem_md.get("fitness") == disk_md.get("fitness")
    assert mem_md.get("last_metrics") == disk_md.get("last_metrics")
    s2_disk = {
        c["candidate_id"]: (c.get("metadata") or {}).get("fitness")
        for c in json.loads(
            (out / "stage2_population.json").read_text(encoding="utf-8")
        )["population"]
    }
    for cand in arts.stage2_population:
        assert (cand.metadata or {}).get("fitness") == s2_disk[cand.candidate_id]


class _ConfigCapturingStage3(_MetadataReassigningStage3):
    def __init__(self, inner, seen):
        super().__init__(inner)
        self.seen = seen

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        self.seen.append(config)
        return self.inner.train_and_eval(
            candidate, round_index=round_index, config=config, **kwargs
        )


def test_pipeline_highway_stage3_receives_highway_config(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer_for_domain",
        lambda pack, use_stub=False: _ConfigCapturingStage3(
            make_stage3_trainer_for_domain(pack, use_stub=use_stub), seen
        ),
    )
    cfg = RaiseRunConfig(output_dir=str(tmp_path / "run_hw_cfg"), domain="highway")
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
    assert s3.skip_final_refine is True
    assert s3.highway_n_steps == 64
    assert s3.highway_eval_deterministic is False


def test_pipeline_fast_with_surrogate(tmp_path):
    from raise_core.surrogate.gate import surrogate_model_ready

    model_dir = "artifacts/surrogate"
    if not surrogate_model_ready(model_dir):
        import pytest

        pytest.skip("no local artifacts/surrogate/model.joblib — run bootstrap --fast first")

    out = tmp_path / "run_surr"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    cfg.surrogate_model_dir = model_dir
    cfg.surrogate_gate_stage3 = True
    cfg.surrogate_drop_fraction = 0.5
    cfg.surrogate_min_keep = 1
    cfg.active_learning = False
    arts = RaisePipeline(cfg).run()
    assert arts.best_stage3 is not None
    assert (out / "surrogate_preds_stage1.json").is_file()
    assert (out / "surrogate_preds_stage2.json").is_file()
    assert (out / "surrogate_gate_stage3.json").is_file()
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["surrogate"]["enabled"] is True
    assert man["surrogate"]["gate"] is not None


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "a" / "b.json"
    write_json(str(path), {"x": 1, "episodes": [{"k": 0}]})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["x"] == 1
    assert data["episodes"][0]["k"] == 0
