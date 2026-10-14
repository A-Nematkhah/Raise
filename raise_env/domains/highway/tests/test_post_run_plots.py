"""Offline plot helpers for highway closed-loop artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from domains.highway.post_run import post_run_summary, write_highway_run_artifacts
from raise_core.pipeline import RaisePipeline, RaiseRunConfig

_ROOT = Path(__file__).resolve().parents[3]


def _plot_module():
    path = _ROOT / "scripts" / "plot_raise_run.py"
    spec = importlib.util.spec_from_file_location("plot_raise_run", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _epoch_rows():
    """Shape written by ClosedLoopRunner (gen0: top Score1 != fitness-best)."""
    return [
        {
            "epoch": 0,
            "best_score1": 0.62,
            "evolve_best_id": "ini_0000",
            "evolve_best_score1": 0.41,
            "evolve_best_fitness": -0.91,
            "evolve_best_fitness_se": 0.21,
            "n_labeled_total": 6,
            "label_metrics": {"ini_0000": {"fitness": -0.91, "fitness_se": 0.21}},
        },
        {
            # Older row without evolve_best_fitness_se: SE comes from label_metrics.
            "epoch": 1,
            "best_score1": 0.66,
            "evolve_best_id": "cro_0007",
            "evolve_best_score1": 0.66,
            "evolve_best_fitness": -0.22,
            "n_labeled_total": 11,
            "label_metrics": {"cro_0007": {"fitness": -0.22, "fitness_se": 0.18}},
        },
        {
            # Carried elite: not in label_metrics and no SE recorded.
            "epoch": 2,
            "best_score1": 0.7,
            "evolve_best_id": "cro_0007",
            "evolve_best_score1": 0.66,
            "evolve_best_fitness": -0.53,
            "n_labeled_total": 16,
            "label_metrics": {},
        },
    ]


def _write_epochs(run_dir: Path, rows) -> None:
    cl = run_dir / "closed_loop"
    cl.mkdir(parents=True)
    with open(cl / "epochs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_closed_loop_series_tracks_the_fitness_best_genome():
    s = _plot_module().closed_loop_series(_epoch_rows())
    assert s["epochs"] == [0, 1, 2]
    assert s["best_id"] == ["ini_0000", "cro_0007", "cro_0007"]
    assert s["score1_best"] == [0.41, 0.66, 0.66]
    assert s["score1_max"] == [0.62, 0.66, 0.7]
    assert s["fitness_se"][:2] == [0.21, 0.18]
    assert s["fitness_se"][2] != s["fitness_se"][2]  # NaN: no bar


def test_closed_loop_series_falls_back_to_best_score1():
    rows = [{"epoch": 0, "best_score1": 0.4, "evolve_best_fitness": -1.0}]
    s = _plot_module().closed_loop_series(rows)
    assert s["score1_best"] == [0.4]


def test_plot_closed_loop_epochs(tmp_path: Path):
    _write_epochs(tmp_path, _epoch_rows())
    out = tmp_path / "closed_loop_epochs.png"
    assert _plot_module().plot_closed_loop_epochs(str(tmp_path), str(out)) is True
    assert out.is_file() and out.stat().st_size > 0


def test_epoch_rows_record_fitness_best_se(tmp_path: Path):
    out = tmp_path / "run"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    RaisePipeline(cfg).run()
    lines = (out / "closed_loop" / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert rows and all("evolve_best_fitness_se" in r for r in rows)

    art = write_highway_run_artifacts(str(out), skip_viz=True)
    names = {Path(p).name for p in art["plots"]}
    assert "closed_loop_epochs.png" in names
    summary = post_run_summary(art)
    assert summary["ok"] is True and summary["n_plots"] == len(art["plots"])
    assert summary["plots_error"] is None


def test_post_run_summary_surfaces_failures():
    summary = post_run_summary(
        {
            "plots": [],
            "plots_error": "no epochs",
            "viz": {"ok": False, "reason": "no model.zip"},
            "ok": False,
        }
    )
    assert summary == {
        "ok": False,
        "n_plots": 0,
        "plots_error": "no epochs",
        "viz_ok": False,
        "viz_reason": "no model.zip",
        "viz_error": None,
        "viz_seed_base": None,
    }
