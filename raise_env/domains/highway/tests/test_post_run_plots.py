"""Offline plot helpers for highway closed-loop artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def test_plot_closed_loop_epochs(tmp_path: Path):
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "plot_raise_run.py"
    spec = importlib.util.spec_from_file_location("plot_raise_run", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cl = tmp_path / "closed_loop"
    cl.mkdir()
    rows = [
        {
            "epoch": 1,
            "best_score1": 0.4,
            "n_labeled_dataset": 6,
            "stats": {"soft_success_mean": 0.1, "scalar_mean": -0.2},
        },
        {
            "epoch": 2,
            "best_score1": 0.55,
            "n_labeled_dataset": 12,
            "stats": {"soft_success_mean": 0.3, "scalar_mean": 0.1},
        },
    ]
    with open(cl / "epochs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    out = tmp_path / "closed_loop_epochs.png"
    assert mod.plot_closed_loop_epochs(str(tmp_path), str(out)) is True
    assert out.is_file() and out.stat().st_size > 0
