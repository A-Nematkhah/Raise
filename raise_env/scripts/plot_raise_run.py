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

import os as _os
import sys as _sys

_RAISE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _RAISE_ROOT not in _sys.path:
    _sys.path.insert(0, _RAISE_ROOT)

from raise_core.script_utils import bootstrap_raise_env

bootstrap_raise_env(chdir=True)
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from raise_core.console import ensure_utf8_stdio
from raise_core.json_io import read_json_dict


def _optional_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    return read_json_dict(path)


def _scalar(m: Dict[str, Any]) -> float:
    for key in ("fitness", "selection_scalar"):
        if m.get(key) is not None:
            try:
                return float(m[key])
            except (TypeError, ValueError):
                pass
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

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10.0, 0.7 * len(names) + 2), 7.5), sharex=True
    )
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
    axes[1].set_ylabel("selection scalar")
    axes[1].set_title(f"{stage_label} — scalar score (cached domain fitness when available)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=90, ha="center", fontsize=7)
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
        ("Best Score1\n(Stage-II metrics shown)", "best_stage1.json"),
        ("Stage II", "best_stage2.json"),
        ("Stage III", "best_stage3.json"),
    ):
        payload = _optional_json(os.path.join(run_dir, fname))
        if payload is None:
            continue
        md = payload.get("metadata") or {}
        metrics = md.get("last_metrics") or {}
        if not metrics and "Score1" in label:
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


def _read_epochs_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "closed_loop", "epochs.jsonl")
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _float_or_nan(value: Any) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _evolve_best_se(row: Dict[str, Any]) -> float:
    se = row.get("evolve_best_fitness_se")
    if se is None:
        # Rows written before evolve_best_fitness_se existed.
        best_id = str(row.get("evolve_best_id") or "")
        se = ((row.get("label_metrics") or {}).get(best_id) or {}).get("fitness_se")
    return _float_or_nan(se)


def closed_loop_series(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Per-epoch series of the fitness-best genome (``evolve_best_*``).

    ``score1_best`` is that genome's Score1, so both panels describe the same
    candidate; ``score1_max`` (the population's top Score1) is kept separately.
    """
    score1_best = []
    for r in rows:
        value = r.get("evolve_best_score1")
        score1_best.append(_float_or_nan(r.get("best_score1") if value is None else value))
    return {
        "epochs": [int(r.get("epoch", i)) for i, r in enumerate(rows)],
        "best_id": [str(r.get("evolve_best_id") or "?") for r in rows],
        "score1_best": score1_best,
        "score1_max": [_float_or_nan(r.get("best_score1")) for r in rows],
        "fitness": [_float_or_nan(r.get("evolve_best_fitness")) for r in rows],
        "fitness_se": [_evolve_best_se(r) for r in rows],
        "trained": [_float_or_nan(r.get("n_labeled_total") or 0) for r in rows],
    }


def plot_closed_loop_epochs(run_dir: str, out_path: str) -> bool:
    """Score1 / fitness ± 2 SE of the fitness-best genome, trained count, per epoch."""
    rows = _read_epochs_jsonl(run_dir)
    if not rows:
        return False
    import matplotlib.pyplot as plt

    s = closed_loop_series(rows)
    epochs = s["epochs"]
    trained = s["trained"]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    axes[0].plot(
        epochs, s["score1_best"], "o-", color="#1f4e79", linewidth=1.8,
        label="Score1 of fitness-best",
    )
    axes[0].plot(
        epochs, s["score1_max"], "--", color="#1f4e79", alpha=0.45, linewidth=1.2,
        label="max Score1 (any genome)",
    )
    axes[0].set_ylabel("Score1")
    axes[0].set_title("Closed-loop — fitness-best genome per epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    yerr = [2.0 * se if se == se else 0.0 for se in s["fitness_se"]]
    axes[1].errorbar(
        epochs, s["fitness"], yerr=yerr, fmt="^-", color="#9467bd",
        linewidth=1.6, capsize=4, label="fitness ± 2 SE",
    )
    for x, y, cid in zip(epochs, s["fitness"], s["best_id"]):
        if y == y:
            axes[1].annotate(
                cid, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7
            )
    axes[1].set_ylabel("Best fitness")
    axes[1].set_title(
        "Each epoch is scored on its own seeds; overlapping bars are not an improvement",
        fontsize=8,
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].step(epochs, trained, where="mid", color="#ff7f0e", linewidth=1.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Trained (cumulative)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_best_crowdnav_detail(run_dir: str, out_path: str) -> bool:
    """SR/CR/TR + nav time / path length / danger for best Stage II/III (CrowdNav fields)."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows: List[Tuple[str, Dict[str, float]]] = []
    for label, fname in (("Stage II", "best_stage2.json"), ("Stage III", "best_stage3.json")):
        payload = _optional_json(os.path.join(run_dir, fname))
        if payload is None:
            continue
        m = (payload.get("metadata") or {}).get("last_metrics") or {}
        if not m:
            continue
        cid = str(payload.get("candidate_id", "?"))
        rows.append(
            (
                f"{label}\n{cid}",
                {
                    "SR": float(m.get("SR", 0.0)),
                    "CR": float(m.get("CR", 0.0)),
                    "TR": float(m.get("TR", 0.0)),
                    "nav_time": float(m.get("NT", 0.0) or 0.0),
                    "path_length": float(m.get("PL", 0.0) or 0.0),
                    "danger_freq": float(m.get("ITR", 0.0) or 0.0),
                    "min_dist_danger": float(m.get("SD", 0.0) or 0.0),
                    "scalar": _scalar(m),
                },
            )
        )
    if not rows:
        return False
    # Only emit this panel when CrowdNav navigation fields are present.
    if not any(r[1]["nav_time"] > 0 or r[1]["path_length"] > 0 for r in rows):
        return False

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    labels = [lab for lab, _ in rows]
    x = np.arange(len(labels))
    width = 0.25
    for i, (k, color) in enumerate(
        (("SR", "#2ca02c"), ("CR", "#d62728"), ("TR", "#ff7f0e"))
    ):
        axes[0].bar(x + (i - 1) * width, [m[k] for _, m in rows], width, label=k, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Best — rates")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    width_m = 0.35
    for i, (k, color) in enumerate((("nav_time", "#1f77b4"), ("path_length", "#8c564b"))):
        axes[1].bar(x + (i - 0.5) * width_m, [m[k] for _, m in rows], width_m, label=k, color=color)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("seconds / metres")
    axes[1].set_title("Best — nav time & path length")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    width_f = 0.25
    for i, (k, color) in enumerate(
        (("danger_freq", "#17becf"), ("min_dist_danger", "#bcbd22"), ("scalar", "#9467bd"))
    ):
        axes[2].bar(x + (i - 1) * width_f, [m[k] for _, m in rows], width_f, label=k, color=color)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Best — danger + fitness")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_rejection_categories(run_dir: str, out_path: str) -> bool:
    path = os.path.join(run_dir, "stage1_rejections.jsonl")
    if not os.path.isfile(path):
        path = os.path.join(run_dir, "closed_loop", "stage1_rejections.jsonl")
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
        s1_history = s1.get("history") or []
    else:
        epochs_path = os.path.join(run_dir, "closed_loop", "epochs.jsonl")
        s1_history = []
        if os.path.isfile(epochs_path):
            with open(epochs_path, encoding="utf-8") as epochs_file:
                s1_history = [
                    {
                        "generation": row.get("epoch"),
                        "best_score": row.get("best_score1"),
                        "best_id": row.get("evolve_best_id"),
                    }
                    for line in epochs_file
                    if line.strip()
                    for row in [json.loads(line)]
                ]
    if s1_history:
        path = os.path.join(output_dir, "stage1_score1.png")
        if plot_stage1_score1(s1_history, path):
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

    path = os.path.join(output_dir, "best_summary.png")
    if plot_stage_summary(run_dir, path):
        written.append(path)

    path = os.path.join(output_dir, "stage1_rejections.png")
    if plot_rejection_categories(run_dir, path):
        written.append(path)

    # Closed-loop: also look under run_dir and nested closed_loop/ for rejections.
    path = os.path.join(output_dir, "closed_loop_epochs.png")
    if plot_closed_loop_epochs(run_dir, path):
        written.append(path)

    path = os.path.join(output_dir, "best_crowdnav_detail.png")
    if plot_best_crowdnav_detail(run_dir, path):
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


def main() -> int:
    ensure_utf8_stdio()
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
