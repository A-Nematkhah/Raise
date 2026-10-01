#!/usr/bin/env python
"""Analyze Phase 0-1 label reliability output (label_reliability_highway.py)."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

MODE_SPEED_SPLIT = 22.5  # m/s: ~20.2 cruise (A) vs ~25 crash-prone (B)


def _mode(metrics: Dict[str, float]) -> str:
    return "A" if float(metrics.get("mean_speed", 0.0)) < MODE_SPEED_SPLIT else "B"


def _icc1(matrix: np.ndarray) -> float:
    """One-way random ICC(1): share of variance explained by the candidate."""
    n, k = matrix.shape
    grand = matrix.mean()
    ms_between = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / max(1, n - 1)
    ms_within = ((matrix - matrix.mean(axis=1, keepdims=True)) ** 2).sum() / max(1, n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 1e-12 else float("nan")


def _pairwise_rho(by_seed: Dict[int, Dict[str, float]], ids: List[str]) -> List[float]:
    out = []
    for a, b in itertools.combinations(sorted(by_seed), 2):
        xa = [by_seed[a][i] for i in ids]
        xb = [by_seed[b][i] for i in ids]
        if len(set(xa)) < 2 or len(set(xb)) < 2:
            out.append(float("nan"))
            continue
        out.append(float(spearmanr(xa, xb).statistic))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/_label_reliability_20260930")
    parser.add_argument("--run", default="results/highway_4h_20260930_124808")
    parser.add_argument("--labels", default="labels.jsonl", help="labels.jsonl or labels_reeval.jsonl")
    args = parser.parse_args()

    rows = []
    for name in sorted(os.listdir(args.out)):
        path = os.path.join(args.out, name, args.labels)
        if not (name.startswith("seed_") and os.path.isfile(path)):
            continue
        new_rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        base_path = os.path.join(args.out, name, "labels.jsonl")
        if args.labels != "labels.jsonl" and os.path.isfile(base_path):
            base_native = {
                r["candidate_id"]: r["native"]
                for r in (json.loads(l) for l in open(base_path, encoding="utf-8") if l.strip())
            }
            for r in new_rows:
                r["native"] = r.get("native") or base_native.get(r["candidate_id"], {})
        for r in new_rows:
            if not r.get("native") and r.get("zero_train"):
                r["native"] = r["fixed_eval"]
        rows += [r for r in new_rows if r.get("native")]
    if not rows:
        print("no rows yet")
        return 1

    orig = {
        c["candidate_id"]: c["metadata"]["last_metrics"]
        for c in json.load(open(os.path.join(args.run, "stage2_population.json"), encoding="utf-8"))[
            "population"
        ]
    }
    table = defaultdict(dict)
    for r in rows:
        table[r["candidate_id"]][r["seed"]] = r
    seeds = sorted({r["seed"] for r in rows})
    ids = sorted(cid for cid in table if all(s in table[cid] for s in seeds))
    print(f"seeds={seeds} complete candidates={len(ids)}/{len(table)}\n")

    print(f"{'cand':9s} {'orig':>12s} " + " ".join(f"{'s'+str(s)+' nat|fix':>22s}" for s in seeds) + "  H(entropy)")
    flips = 0
    for cid in ids:
        o = orig.get(cid, {})
        cells = []
        modes = set()
        for s in seeds:
            r = table[cid][s]
            n, f = r["native"], r["fixed_eval"]
            modes.update({_mode(n), _mode(f)})
            cells.append(f"{n['SR']:.2f}/{_mode(n)} | {f['SR']:.2f}/{_mode(f)}")
        if o:
            modes.add(_mode(o))
        flips += int(len(modes) > 1)
        ents = [table[cid][s]["actions"]["mean_entropy"] for s in seeds]
        print(
            f"{cid:9s} {o.get('SR', float('nan')):.2f}/{_mode(o) if o else '?':>1s}      "
            + " ".join(f"{c:>22s}" for c in cells)
            + "  " + ",".join(f"{e:.2f}" for e in ents)
        )

    views = ["native", "fixed_eval"]
    if all("fixed_eval_stochastic" in table[cid][s] for cid in ids for s in seeds):
        views.append("fixed_eval_stochastic")
        print("\nstochastic fixed eval (SR/mode, mean_speed):")
        for cid in ids:
            cells = []
            for s in seeds:
                m = table[cid][s]["fixed_eval_stochastic"]
                cells.append(f"{m['SR']:.2f}/{_mode(m)} v={float(m.get('mean_speed', 0.0)):.1f}")
            print(f"  {cid:9s} " + "  ".join(f"{c:>18s}" for c in cells))
        stoch_gap = [
            abs(table[cid][s]["fixed_eval"]["SR"] - table[cid][s]["fixed_eval_stochastic"]["SR"])
            for cid in ids
            for s in seeds
        ]
        print(f"deterministic vs stochastic |dSR|: mean={np.mean(stoch_gap):.3f} max={np.max(stoch_gap):.3f}")

    for view in views:
        for key in ("SR", "fitness"):
            by_seed = {s: {cid: float(table[cid][s][view].get(key, 0.0)) for cid in ids} for s in seeds}
            rhos = _pairwise_rho(by_seed, ids)
            mat = np.array([[by_seed[s][cid] for s in seeds] for cid in ids])
            mean_rho = float(np.nanmean(rhos)) if np.isfinite(rhos).any() else float("nan")
            print(
                f"\n[{view:21s} {key:7s}] pairwise rho={['%.2f' % x for x in rhos]} "
                f"mean={mean_rho:.2f}  ICC(1)={_icc1(mat):.2f}"
            )

    eval_gap = [
        abs(table[cid][s]["native"]["SR"] - table[cid][s]["fixed_eval"]["SR"]) for cid in ids for s in seeds
    ]
    print(f"\nsame model, native vs fixed eval |dSR|: mean={np.mean(eval_gap):.3f} max={np.max(eval_gap):.3f}")
    print(f"candidates with a mode flip (incl. original label): {flips}/{len(ids)}")
    all_actions = defaultdict(float)
    for r in rows:
        for a, frac in r["actions"]["action_frac"].items():
            all_actions[a] += frac / len(rows)
    print("mean deterministic action share:", {k: round(v, 3) for k, v in sorted(all_actions.items())})
    nat_rhos = _pairwise_rho(
        {s: {cid: float(table[cid][s]["native"]["SR"]) for cid in ids} for s in seeds}, ids
    )
    rho_nat = float(np.nanmean(nat_rhos)) if np.isfinite(nat_rhos).any() else float("nan")
    if not np.isfinite(rho_nat):
        print("\nDECISION GATE: N/A (labels constant across candidates -> no ranking signal)")
        return 0
    noisy = (rho_nat < 0.5) or (flips > len(ids) / 3.0)
    print(f"\nDECISION GATE (rho<0.5 or >1/3 flips): {'NOISY -> fix label noise first' if noisy else 'OK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
