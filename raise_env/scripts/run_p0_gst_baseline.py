#!/usr/bin/env python
"""Resume/fix CrowdNav++ GST baseline into an existing p0_comparison.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def _scalar(m):
    return float(m.get("SR", 0) - m.get("CR", 0) - 0.5 * m.get("TR", 0))


def _mean_std(vals):
    import math

    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return {"n": n, "mean": mean, "std": math.sqrt(var)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        default="results/run_scaled_h5_gst/evals/p0_comparison.json",
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=425)
    parser.add_argument("--human-num", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import crowd_sim  # noqa: F401
    from domains.crowdnav.reporting import evaluate_saved_model, write_json

    seeds = [int(args.base_seed) + i for i in range(int(args.n_seeds))]
    model_dir = "trained_models/GST_predictor_non_rand"
    ckpt = "41200.pt"
    rows = []
    for seed in seeds:
        print(f"[GST] seed={seed} E={args.episodes} H={args.human_num} device={args.device}", flush=True)
        t0 = time.perf_counter()
        bundle = evaluate_saved_model(
            model_dir,
            checkpoint=ckpt,
            n_episodes=int(args.episodes),
            seed=seed,
            method="CrowdNav++_GST",
            randomize=False,
            device=args.device,
            human_num=int(args.human_num),
        )
        md = {
            "SR": float(bundle.sr.mean),
            "CR": float(bundle.cr.mean),
            "TR": float(bundle.tr.mean),
            "NT": float(bundle.nt.mean),
            "PL": float(bundle.pl.mean),
            "ITR": float(bundle.itr.mean),
            "SD": float(bundle.sd.mean),
        }
        row = {
            "method": "CrowdNav++_GST",
            "seed": seed,
            "human_num": int(args.human_num),
            "episodes": int(args.episodes),
            "wall_seconds": time.perf_counter() - t0,
            "metrics": md,
            "scalar_SR_minus_CR_minus_0.5TR": _scalar(md),
            "model_dir": model_dir,
            "checkpoint": ckpt,
        }
        rows.append(row)
        print(
            f"[GST] SR={md['SR']:.3f} CR={md['CR']:.3f} TR={md['TR']:.3f} "
            f"scalar={row['scalar_SR_minus_CR_minus_0.5TR']:.3f}",
            flush=True,
        )

    keys = ("SR", "CR", "TR", "NT", "PL", "ITR", "SD")
    agg = {k: _mean_std([float(r["metrics"][k]) for r in rows]) for k in keys}
    agg["scalar"] = _mean_std([float(r["scalar_SR_minus_CR_minus_0.5TR"]) for r in rows])
    block = {
        "method": "CrowdNav++_GST",
        "per_seed": rows,
        "aggregate": agg,
    }

    path = args.comparison
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    report["methods"]["CrowdNav++_GST"] = block
    ranking = []
    for key, b in report["methods"].items():
        if b.get("skipped"):
            ranking.append({"key": key, "skipped": True, "reason": b.get("reason")})
            continue
        a = (b.get("aggregate") or {}).get("scalar") or {}
        ranking.append(
            {
                "key": key,
                "method": b.get("method", key),
                "scalar_mean": a.get("mean"),
                "scalar_std": a.get("std"),
                "SR_mean": ((b.get("aggregate") or {}).get("SR") or {}).get("mean"),
                "CR_mean": ((b.get("aggregate") or {}).get("CR") or {}).get("mean"),
                "TR_mean": ((b.get("aggregate") or {}).get("TR") or {}).get("mean"),
            }
        )
    ok = [r for r in ranking if not r.get("skipped")]
    ok.sort(key=lambda r: float(r.get("scalar_mean") or float("-inf")), reverse=True)
    report["ranking_by_scalar"] = ok + [r for r in ranking if r.get("skipped")]
    write_json(path, report)
    print("Updated ranking:", flush=True)
    for r in report["ranking_by_scalar"]:
        if r.get("skipped"):
            print(f"  SKIP {r['key']}", flush=True)
        else:
            print(
                f"  {r['method']:<28} SR={r['SR_mean']:.3f} CR={r['CR_mean']:.3f} "
                f"scalar={r['scalar_mean']:.3f}±{r['scalar_std']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
