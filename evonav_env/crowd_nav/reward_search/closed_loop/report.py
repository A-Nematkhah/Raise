"""Human-readable closed-loop run reports (terminal + REPORT.txt)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from crowd_nav.reward_search.closed_loop.logging_io import (
    closed_loop_dir,
    read_epochs,
    write_json,
)


def _gate_line(gate: Dict[str, Any]) -> str:
    if gate.get("soft"):
        reason = gate.get("reason") or "soft"
        return f"gate=soft({reason})"
    n_kept = gate.get("n_kept")
    n_dropped = gate.get("n_dropped")
    dropped = gate.get("dropped_ids") or []
    drop_s = ",".join(str(x) for x in dropped[:4])
    if len(dropped) > 4:
        drop_s += ",..."
    extra = f" dropped=[{drop_s}]" if drop_s else ""
    return f"gate=hard keep={n_kept} drop={n_dropped}{extra}"


def _al_line(al: Dict[str, Any]) -> str:
    if not al.get("enabled"):
        return "al=off"
    return (
        f"al=on stage2={al.get('n_al_stage2', 0)} "
        f"s1_req={al.get('n_stage1_requests', 0)}"
    )


def format_epoch_summary(rec: Dict[str, Any]) -> str:
    """One scannable line for a closed-loop epoch."""
    g = int(rec.get("epoch", -1))
    best = rec.get("best_id") or "?"
    score = rec.get("best_score1")
    try:
        score_s = f"{float(score):.4f}" if score is not None else "n/a"
    except (TypeError, ValueError):
        score_s = str(score)
    labeled = rec.get("labeled_ids") or []
    label_s = ",".join(str(x) for x in labeled)
    gate = rec.get("gate") or {}
    al = rec.get("al") or {}
    refit = "yes" if rec.get("refit") else "no"
    return (
        f"epoch {g}: best={best} Score1={score_s} | "
        f"label {len(labeled)}/{rec.get('n_population', '?')} [{label_s}] | "
        f"{_gate_line(gate)} | {_al_line(al)} | "
        f"dataset={rec.get('n_labeled_dataset', '?')} refit={refit} "
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
            f"final dataset labels: {last.get('n_labeled_dataset')} | "
            f"last best: {last.get('best_id')} "
            f"Score1={last.get('best_score1')}"
        )
        hard = sum(1 for r in rows if not (r.get("gate") or {}).get("soft", True))
        soft = len(rows) - hard
        al_on = sum(1 for r in rows if (r.get("al") or {}).get("enabled"))
        refits = sum(1 for r in rows if r.get("refit"))
        lines.append(
            f"summary: soft_gate_epochs={soft} hard_gate_epochs={hard} "
            f"al_epochs={al_on} refits={refits}"
        )
    if manifest:
        lines.append("-" * 60)
        lines.append(f"mode: {manifest.get('mode', 'n/a')}")
        if manifest.get("n_epochs") is not None:
            lines.append(f"manifest n_epochs: {manifest.get('n_epochs')}")
        if manifest.get("n_labeled_total") is not None:
            lines.append(f"manifest n_labeled: {manifest.get('n_labeled_total')}")
    lines.append("-" * 60)
    lines.append("files:")
    lines.append(f"  {os.path.join(output_dir, 'closed_loop', 'epochs.jsonl')}")
    lines.append(f"  {os.path.join(output_dir, 'closed_loop', 'REPORT.txt')}")
    lines.append(f"  {os.path.join(output_dir, 'manifest.json')}")
    return "\n".join(lines) + "\n"


def write_closed_loop_report(
    output_dir: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    text = build_closed_loop_report(output_dir, manifest=manifest)
    path = os.path.join(closed_loop_dir(output_dir), "REPORT.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    # Also keep a JSON summary for tooling.
    rows = read_epochs(output_dir)
    write_json(
        os.path.join(closed_loop_dir(output_dir), "report_summary.json"),
        {
            "n_epochs": len(rows),
            "epochs": [
                {
                    "epoch": r.get("epoch"),
                    "best_id": r.get("best_id"),
                    "best_score1": r.get("best_score1"),
                    "n_to_label": r.get("n_to_label"),
                    "gate_soft": (r.get("gate") or {}).get("soft"),
                    "n_dropped": (r.get("gate") or {}).get("n_dropped"),
                    "al_enabled": (r.get("al") or {}).get("enabled"),
                    "n_al_stage2": (r.get("al") or {}).get("n_al_stage2"),
                    "n_labeled_dataset": r.get("n_labeled_dataset"),
                    "refit": r.get("refit"),
                }
                for r in rows
            ],
            "manifest": manifest or {},
        },
    )
    return path


def log_closed_loop_report(output_dir: str, *, manifest: Optional[Dict[str, Any]] = None) -> str:
    """Write REPORT.txt and print it via console status lines."""
    from crowd_nav.reward_search import console

    path = write_closed_loop_report(output_dir, manifest=manifest)
    text = build_closed_loop_report(output_dir, manifest=manifest)
    console.banner("Closed-loop report")
    for line in text.strip().splitlines():
        console.status(line, stage="closed-loop")
    console.status(f"Wrote {path}", stage="closed-loop")
    return path
