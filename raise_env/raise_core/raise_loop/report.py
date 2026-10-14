"""Human-readable closed-loop run reports (terminal + REPORT.txt)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from raise_core.json_io import atomic_write_text
from raise_core.raise_loop.logging_io import (
    closed_loop_dir,
    read_epochs,
    write_json,
)


def _fmt(value: Any, spec: str) -> str:
    try:
        return format(float(value), spec) if value is not None else "n/a"
    except (TypeError, ValueError):
        return str(value)


def format_epoch_summary(rec: Dict[str, Any]) -> str:
    """One scannable line for a closed-loop epoch."""
    g = int(rec.get("epoch", -1))
    labeled = rec.get("labeled_ids") or []
    return (
        f"epoch {g}: best={rec.get('evolve_best_id') or '?'} "
        f"fitness={_fmt(rec.get('evolve_best_fitness'), '.3f')} "
        f"Score1={_fmt(rec.get('evolve_best_score1'), '.4f')} | "
        f"trained {len(labeled)}/{rec.get('n_population', '?')} "
        f"ok={rec.get('n_label_ok', 0)} fail={rec.get('n_label_failed', 0)}"
    )


def build_closed_loop_report(
    output_dir: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    epochs: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Multi-line plain-text report for operators."""
    rows = list(epochs) if epochs is not None else read_epochs(output_dir)
    lines: List[str] = [
        "Closed-loop run report",
        f"output: {os.path.abspath(output_dir)}",
        f"epochs: {len(rows)}",
        "-" * 60,
    ]
    if not rows:
        lines.append("(no epochs.jsonl yet)")
    else:
        for rec in rows:
            lines.append(format_epoch_summary(rec))
        last = rows[-1]
        lines.append("-" * 60)
        lines.append(
            f"last best: {last.get('evolve_best_id')} "
            f"fitness={_fmt(last.get('evolve_best_fitness'), '.3f')}"
        )
    if manifest:
        lines.append("-" * 60)
        lines.append(f"mode: {manifest.get('mode', 'n/a')}")
        if manifest.get("n_epochs") is not None:
            lines.append(f"manifest n_epochs: {manifest.get('n_epochs')}")
        if manifest.get("n_labeled_total") is not None:
            lines.append(f"manifest n_trained: {manifest.get('n_labeled_total')}")
    lines.append("-" * 60)
    lines.append("files:")
    lines.append(f"  {os.path.join(output_dir, 'closed_loop', 'epochs.jsonl')}")
    lines.append(f"  {os.path.join(output_dir, 'closed_loop', 'REPORT.txt')}")
    lines.append(f"  {os.path.join(output_dir, 'closed_loop', 'manifest.json')}")
    return "\n".join(lines) + "\n"


def write_closed_loop_report(
    output_dir: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    text = build_closed_loop_report(output_dir, manifest=manifest)
    path = atomic_write_text(os.path.join(closed_loop_dir(output_dir), "REPORT.txt"), text)
    # Also keep a JSON summary for tooling.
    rows = read_epochs(output_dir)
    write_json(
        os.path.join(closed_loop_dir(output_dir), "report_summary.json"),
        {
            "n_epochs": len(rows),
            "epochs": [
                {
                    "epoch": r.get("epoch"),
                    "best_id": r.get("evolve_best_id"),
                    "best_fitness": r.get("evolve_best_fitness"),
                    "best_score1": r.get("evolve_best_score1"),
                    "n_to_label": r.get("n_to_label"),
                    "n_label_ok": r.get("n_label_ok"),
                    "n_label_failed": r.get("n_label_failed"),
                }
                for r in rows
            ],
            "manifest": manifest or {},
        },
    )
    return path
