"""Unit tests for paper-scale presets, checkpointing, and cross-seed aggregation.

Does **not** invoke ``scripts/run_evonav_paper_scale.py`` or paper K3 training.
"""

from __future__ import annotations

import os

from crowd_nav.reward_search.checkpointing import (
    CheckpointKey,
    CheckpointStore,
    CostLogger,
)
from crowd_nav.reward_search.paper_scale import (
    aggregate_across_seeds,
    build_paper_scale_report,
)
from crowd_nav.reward_search.pipeline import EvoNavRunConfig
from crowd_nav.reward_search.presets import (
    PAPER_DEFAULT_SEEDS,
    PAPER_K3,
    apply_paper_scale,
    load_paper_scale_yaml,
    parse_seeds_arg,
)


def test_paper_scale_yaml_tables_3_to_6():
    spec = load_paper_scale_yaml()
    assert spec.N == 8
    assert spec.G1 == 10
    assert spec.M == 100
    assert spec.N_traj == 10
    assert spec.K2 == 8000
    assert spec.G2 == 16
    assert spec.E2 == 50
    assert spec.K3 == PAPER_K3 == int(1e7)
    assert spec.G3 == 3
    assert spec.E3 == 500
    assert list(spec.seeds) == list(PAPER_DEFAULT_SEEDS)
    assert len(spec.seeds) == 5
    note = spec.methodology_note()
    assert "across seeds" in note
    assert "5" in note or "does not state" in note.lower()


def test_apply_paper_scale_distinct_from_fast():
    cfg = EvoNavRunConfig()
    cfg.apply_fast_profile()
    assert cfg.fast is True
    assert cfg.stage3_train_steps < 100

    paper = EvoNavRunConfig()
    apply_paper_scale(paper)
    assert paper.fast is False
    assert paper.stage1_population == 8
    assert paper.stage1_generations == 10
    assert paper.stage2_rounds == 16
    assert paper.stage2_train_steps == 8000
    assert paper.stage2_k2_unit == "gradient_steps"
    assert paper.elitism is False
    assert paper.final_rank == "llm"
    assert paper.stage3_rounds == 3
    assert paper.stage3_train_steps == int(1e7)
    assert paper.stage3_eval_episodes == 500
    assert paper.stage2_use_stub is False
    assert paper.stage3_use_stub is False
    assert paper.score1_mode == "dataset"


def test_pipeline_defaults_match_paper_fidelity():
    cfg = EvoNavRunConfig()
    assert cfg.stage2_train_steps == 8_000
    assert cfg.stage2_k2_unit == "gradient_steps"
    assert cfg.elitism is False
    assert cfg.final_rank == "llm"


def test_parse_seeds_default_five():
    assert parse_seeds_arg(None) == list(PAPER_DEFAULT_SEEDS)
    assert parse_seeds_arg([1, 2]) == [1, 2]


def test_aggregate_across_seeds_not_episodes():
    rows = [
        {"seed": 425, "metrics": {"SR": 0.80, "CR": 0.10}},
        {"seed": 426, "metrics": {"SR": 0.60, "CR": 0.20}},
    ]
    agg = aggregate_across_seeds(rows)
    assert agg["n_seeds"] == 2
    assert agg["aggregation"] == "across_seeds"
    assert abs(agg["metrics"]["SR"]["mean"] - 0.7) < 1e-9
    assert agg["metrics"]["SR"]["n"] == 2


def test_checkpoint_resume_roundtrip(tmp_path):
    cost = CostLogger(str(tmp_path / "cost_log.json"))
    store = CheckpointStore(str(tmp_path / "ckpts"), cost_logger=cost, device="cpu")
    key = CheckpointKey(seed=425, stage="stage2", round=0, candidate_id="c000")
    assert not store.has(key)
    store.save(key, {"metrics": {"SR": 0.5}}, wall_seconds=12.0, resumed=False)
    assert store.has(key)
    loaded = store.load(key)
    assert loaded["payload"]["metrics"]["SR"] == 0.5
    store.mark_stage_done(425, "stage3", {"ok": True})
    assert store.seed_complete(425)
    assert os.path.isfile(tmp_path / "cost_log.json")
    totals = cost.totals()
    assert totals["n_events"] >= 1
    assert totals["wall_seconds_total"] >= 12.0


def test_build_report_documents_seed_methodology(tmp_path):
    spec = load_paper_scale_yaml()
    rows = [
        {"seed": s, "metrics": {"SR": 0.5 + 0.01 * i, "CR": 0.1}}
        for i, s in enumerate(spec.seeds)
    ]
    report = build_paper_scale_report(
        spec=spec,
        seeds=spec.seeds,
        per_seed_rows=rows,
        cost_log_path=str(tmp_path / "cost_log.json"),
        output_dir=str(tmp_path),
    )
    assert "methodology" in report
    assert report["methodology"]["seed_count_default"] == 5
    assert report["methodology"]["variance_source"] == "across_seeds"
    assert report["hyperparameters"]["K3"] == int(1e7)
    assert report["table1_style_aggregate"]["n_seeds"] == 5


def test_ci_guard_in_paper_scale_script():
    """The paper-scale entry point must refuse CI environments."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    script = os.path.join(root, "scripts", "run_evonav_paper_scale.py")
    assert os.path.isfile(script)
    text = open(script, encoding="utf-8").read()
    assert "CI" in text
    assert "Refusing paper-scale run under CI" in text
    assert "Refusing to run a non-fast pipeline" in text
    assert "--allow-seed-llm" in text


def test_paper_scale_yaml_documents_seed_fail_closed():
    spec = load_paper_scale_yaml()
    assert spec.llm_provider == "seed"
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "paper_scale.yaml"
    )
    text = open(os.path.abspath(yaml_path), encoding="utf-8").read()
    assert "allow-seed-llm" in text or "fail-closes" in text or "fail-closed" in text.lower() or "Intentionally not a real LLM" in text

