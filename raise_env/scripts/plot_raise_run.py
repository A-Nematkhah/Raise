#!/usr/bin/env python
"""
Plot RAISE metrics from a run directory (offline, no GPU).

Reads stage*_population.json / best_stage*.json and writes PNGs under
``<run-dir>/plots/`` (or ``--output-dir``).

Examples::

    python scripts/plot_raise_run.py --run-dir results/run_1to2h
    python scripts/plot_raise_run.py --run-dir results/run_easy --show
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _optional_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    return _load_json(path)


def _scalar(m: Dict[str, Any]) -> float:
    sr = float(m.get("SR", 0.0))
    cr = float(m.get("CR", 0.0))
    tr = float(m.get("TR", 0.0))
    return sr - cr - 0.5 * tr


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_stage1_score1(history: Sequence[Dict[str, Any]], out_path: str) -> bool:
    if not history:
        return False
    import matplotlib.pyplot as plt

    gens = [int(h.get("generation", i)) for i, h in enumerate(history)]
    scores = [float(h.get("best_score", float("nan"))) for h in history]
    labels = [str(h.get("best_id", "?")) for h in history]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(gens, scores, marker="o", color="#1f4e79", linewidth=2)
    for g, s, lab in zip(gens, scores, labels):
        ax.annotate(lab, (g, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Score1")
    ax.set_title("Stage I — best Score1 per generation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_stage_metrics(
    history: Sequence[Dict[str, Any]],
    *,
    stage_label: str,
    out_path: str,
) -> bool:
    if not history:
        return False
    import matplotlib.pyplot as plt
    import numpy as np

    by_round: Dict[int, List[Dict[str, Any]]] = {}
    for rec in history:
        r = int(rec.get("round_index", 0))
        by_round.setdefault(r, []).append(rec)
    rounds = sorted(by_round)
    if not rounds:
        return False

    # Flatten: x positions with gaps between rounds
    names: List[str] = []
    srs: List[float] = []
    crs: List[float] = []
    trs: List[float] = []
    scalars: List[float] = []
    xticks: List[float] = []
    x = 0.0
    round_centers: List[Tuple[float, str]] = []
    for r in rounds:
        start = x
        for rec in by_round[r]:
            m = rec.get("metrics") or {}
            cid = str(rec.get("candidate_id", "?"))
            refined = bool(rec.get("refined"))
            kept = bool(rec.get("kept_previous"))
            tag = cid
            if refined:
                tag += "*"
            elif kept:
                tag += "·"
            names.append(tag)
            srs.append(float(m.get("SR", 0.0)))
            crs.append(float(m.get("CR", 0.0)))
            trs.append(float(m.get("TR", 0.0)))
            scalars.append(_scalar(m))
            xticks.append(x)
            x += 1.0
        end = x - 1.0
        round_centers.append(((start + end) / 2.0, f"R{r}"))
        x += 0.8  # gap

    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.55 * len(names) + 2), 7.0), sharex=True)
    width = 0.25
    xs = np.asarray(xticks)
    axes[0].bar(xs - width, srs, width, label="SR", color="#2ca02c")
    axes[0].bar(xs, crs, width, label="CR", color="#d62728")
    axes[0].bar(xs + width, trs, width, label="TR", color="#ff7f0e")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(f"{stage_label} — SR / CR / TR by candidate (* refined, · kept)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(xs, scalars, marker="o", color="#1f4e79", linewidth=1.8)
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("SR − CR − 0.5·TR")
    axes[1].set_title(f"{stage_label} — scalar score")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    for cx, lab in round_centers:
        axes[1].annotate(
            lab,
            xy=(cx, axes[1].get_ylim()[0]),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#555555",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_stage_summary(run_dir: str, out_path: str) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np

    rows: List[Tuple[str, Dict[str, float]]] = []
    for label, fname in (
        ("Stage I\n(Score1 only)", "best_stage1.json"),
        ("Stage II", "best_stage2.json"),
        ("Stage III", "best_stage3.json"),
    ):
        payload = _optional_json(os.path.join(run_dir, fname))
        if payload is None:
            continue
        md = payload.get("metadata") or {}
        metrics = md.get("last_metrics") or {}
        if not metrics and label.startswith("Stage I"):
            score = payload.get("score")
            if score is None:
                continue
            rows.append((f"{label}\n{payload.get('candidate_id', '?')}", {"Score1": float(score)}))
            continue
        if not metrics:
            continue
        cid = str(payload.get("candidate_id", "?"))
        rows.append((f"{label}\n{cid}", {k: float(metrics[k]) for k in ("SR", "CR", "TR") if k in metrics}))

    if not rows:
        return False

    # Prefer a single SR/CR/TR panel when Stage II/III exist; Score1 as text inset if alone.
    metric_keys = ["SR", "CR", "TR"]
    has_rates = any(any(k in m for k in metric_keys) for _, m in rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if has_rates:
        labels = [lab for lab, m in rows if any(k in m for k in metric_keys)]
        data = [[m.get(k, 0.0) for k in metric_keys] for _, m in rows if any(k in m for k in metric_keys)]
        x = np.arange(len(labels))
        width = 0.25
        colors = {"SR": "#2ca02c", "CR": "#d62728", "TR": "#ff7f0e"}
        for i, k in enumerate(metric_keys):
            vals = [row[i] for row in data]
            ax.bar(x + (i - 1) * width, vals, width, label=k, color=colors[k])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("Best candidates — Stage II / III metrics")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    else:
        labels = [lab.replace("\n", " ") for lab, _ in rows]
        vals = [list(m.values())[0] for _, m in rows]
        ax.bar(range(len(vals)), vals, color="#1f4e79")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Score1")
        ax.set_title("Best Stage I Score1")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_rejection_categories(run_dir: str, out_path: str) -> bool:
    path = os.path.join(run_dir, "stage1_rejections.jsonl")
    if not os.path.isfile(path):
        return False
    counts: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("accepted") is True:
                continue
            cat = str(
                rec.get("rejection_category")
                or rec.get("category")
                or rec.get("reason")
                or "unknown"
            )
            if cat in ("accepted", "ok", "none"):
                continue
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return False
    import matplotlib.pyplot as plt

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(labels) + 1.5)))
    ax.barh(range(len(vals)), vals, color="#7f7f7f")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Stage I sandbox rejections by category")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def run_plots(run_dir: str, output_dir: str, *, show: bool = False) -> List[str]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    _ensure_dir(output_dir)

    s1 = _optional_json(os.path.join(run_dir, "stage1_population.json"))
    if s1 is not None:
        path = os.path.join(output_dir, "stage1_score1.png")
        if plot_stage1_score1(s1.get("history") or [], path):
            written.append(path)

    s2 = _optional_json(os.path.join(run_dir, "stage2_population.json"))
    if s2 is not None:
        path = os.path.join(output_dir, "stage2_metrics.png")
        if plot_stage_metrics(s2.get("history") or [], stage_label="Stage II", out_path=path):
            written.append(path)

    s3 = _optional_json(os.path.join(run_dir, "stage3_population.json"))
    if s3 is not None:
        path = os.path.join(output_dir, "stage3_metrics.png")
        if plot_stage_metrics(s3.get("history") or [], stage_label="Stage III", out_path=path):
            written.append(path)
        h_sweep = s3.get("h_sweep") or []
        if h_sweep:
            path = os.path.join(output_dir, "stage3_h_sweep.png")
            if _plot_h_sweep(h_sweep, path):
                written.append(path)

    path = os.path.join(output_dir, "best_summary.png")
    if plot_stage_summary(run_dir, path):
        written.append(path)

    path = os.path.join(output_dir, "stage1_rejections.png")
    if plot_rejection_categories(run_dir, path):
        written.append(path)

    if show and written:
        for p in written:
            img = plt.imread(p)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(p))
        plt.show()
    return written


def _plot_h_sweep(h_sweep: Sequence[Any], out_path: str) -> bool:
    # Accept either list of {human_num, metrics} or nested by_human_count dicts.
    points: List[Tuple[int, Dict[str, float]]] = []
    for item in h_sweep:
        if not isinstance(item, dict):
            continue
        if "by_human_count" in item:
            for h, m in (item.get("by_human_count") or {}).items():
                if isinstance(m, dict):
                    points.append((int(h), {k: float(m.get(k, 0.0)) for k in ("SR", "CR", "TR")}))
        elif "human_num" in item and "metrics" in item:
            m = item["metrics"] or {}
            points.append(
                (int(item["human_num"]), {k: float(m.get(k, 0.0)) for k in ("SR", "CR", "TR")})
            )
    if not points:
        return False
    import matplotlib.pyplot as plt

    points.sort(key=lambda t: t[0])
    hs = [h for h, _ in points]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(hs, [m["SR"] for _, m in points], "o-", label="SR", color="#2ca02c")
    ax.plot(hs, [m["CR"] for _, m in points], "s-", label="CR", color="#d62728")
    ax.plot(hs, [m["TR"] for _, m in points], "^-", label="TR", color="#ff7f0e")
    ax.set_xlabel("Human count H")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Stage III — Table 6 H-sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RAISE run metrics to PNG")
    parser.add_argument("--run-dir", required=True, help="results/<run> directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write PNGs (default: <run-dir>/plots)",
    )
    parser.add_argument("--show", action="store_true", help="Also open figures interactively")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.output_dir or os.path.join(run_dir, "plots"))
    written = run_plots(run_dir, out_dir, show=bool(args.show))
    if not written:
        print(f"No plottable artifacts found under {run_dir}", file=sys.stderr)
        return 2
    print(f"Wrote {len(written)} figure(s) to {out_dir}:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
