"""P1-6: Score1 vs fitness analysis helper."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_score1_vs_fitness import analyze_run, format_report_line, main


def _write_pop(path: Path, rows):
    path.write_text(
        json.dumps({"candidates": rows}, indent=2) + "\n", encoding="utf-8"
    )


def test_analyze_score1_vs_fitness_on_synthetic_run(tmp_path: Path):
    cl = tmp_path / "closed_loop"
    cl.mkdir()
    _write_pop(
        cl / "pop_epoch_000.json",
        [
            {
                "candidate_id": "a",
                "score": 0.9,
                "metadata": {
                    "fitness": 0.1,
                    "last_metrics": {"holdout": {"SR": 0.5, "CR": 0.2}},
                },
            },
            {
                "candidate_id": "b",
                "score": 0.2,
                "metadata": {
                    "fitness": -0.2,
                    "last_metrics": {"holdout": {"SR": 0.0, "CR": 0.9}},
                },
            },
            {
                "candidate_id": "c",
                "score": 0.5,
                "metadata": {
                    "fitness": 0.05,
                    "last_metrics": {"holdout": {"SR": 0.4, "CR": 0.3}},
                },
            },
        ],
    )
    summary = analyze_run(tmp_path)
    assert summary["n_trained"] == 3
    assert summary["all"]["n"] == 3
    assert summary["collapsed_cr_ge_0.8_score1"]["n"] == 1
    line = format_report_line(summary)
    assert "Spearman=" in line
    assert main([str(tmp_path)]) == 0
