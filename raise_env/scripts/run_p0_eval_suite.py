#!/usr/bin/env python
"""
P0 evaluation suite for thesis/audit follow-up.

Locks in:
  1) RAISE Stage III best-ever (R0 mut_0060_v2) vs last-round (R1) policies
  2) Classical / CrowdNav++ baselines under the SAME protocol as the scaled run:
     H=5, GST inferred, without_random, E episodes, multiple eval seeds

Does NOT retrain. Writes a single comparison JSON.

Example::

    python scripts/run_p0_eval_suite.py \\
        --run-dir results/run_scaled_h5_gst \\
        --episodes 150 --n-seeds 3 --device cuda
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "eval_raise_checkpoint",
    os.path.join(_ROOT, "scripts", "eval_raise_checkpoint.py"),
)
_eval_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_mod)
_scalar = _eval_mod._scalar
run_eval = _eval_mod.run_eval


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean_std(vals: List[float]) -> Dict[str, float]:
    import math

    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    mean = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return {"n": n, "mean": mean, "std": math.sqrt(var)}


def _aggregate_seed_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("SR", "CR", "TR", "NT", "PL", "ITR", "SD")
    out: Dict[str, Any] = {"per_seed": rows, "aggregate": {}}
    for k in keys:
        vals = [float((r.get("metrics") or {}).get(k, 0.0)) for r in rows]
        out["aggregate"][k] = _mean_std(vals)
    scalars = [float(r.get("scalar_SR_minus_CR_minus_0.5TR", 0.0)) for r in rows]
    out["aggregate"]["scalar"] = _mean_std(scalars)
    return out


def eval_raise_multi_seed(
    *,
    run_dir: str,
    candidate_id: str,
    round_index: int,
    episodes: int,
    seeds: List[int],
    device: str,
    human_num: int,
) -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        print(f"\n=== RAISE {candidate_id} r{round_index:02d} seed={seed} E={episodes} ===")
        row = run_eval(
            run_dir=run_dir,
            candidate_id=candidate_id,
            round_index=round_index,
            episodes=episodes,
            device_pref=device,
            human_num=human_num,
            seed=seed,
            output_json=os.path.join(
                run_dir,
                "evals",
                f"eval_r{round_index:02d}_{candidate_id}_E{episodes}_s{seed}.json",
            ),
        )
        rows.append(row)
    return {
        "method": f"RAISE_{candidate_id}_r{round_index:02d}",
        "candidate_id": candidate_id,
        "round_index": round_index,
        **_aggregate_seed_rows(rows),
    }


def eval_baseline_multi_seed(
    *,
    name: str,
    model_dir: str,
    checkpoint: str,
    episodes: int,
    seeds: List[int],
    device: str,
    human_num: int,
    randomize: bool = False,
) -> Dict[str, Any]:
    from domains.crowdnav.reporting import evaluate_saved_model

    rows = []
    for seed in seeds:
        print(f"\n=== Baseline {name} seed={seed} E={episodes} H={human_num} ===")
        t0 = time.perf_counter()
        bundle = evaluate_saved_model(
            model_dir,
            checkpoint=checkpoint,
            n_episodes=episodes,
            seed=seed,
            method=name,
            randomize=randomize,
            device=device,
            human_num=human_num,
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
            "method": name,
            "seed": seed,
            "human_num": human_num,
            "episodes": episodes,
            "wall_seconds": time.perf_counter() - t0,
            "metrics": md,
            "scalar_SR_minus_CR_minus_0.5TR": _scalar(md),
            "model_dir": model_dir,
            "checkpoint": checkpoint,
        }
        rows.append(row)
        print(
            f"[{name}] seed={seed} SR={md['SR']:.3f} CR={md['CR']:.3f} "
            f"TR={md['TR']:.3f} scalar={row['scalar_SR_minus_CR_minus_0.5TR']:.3f}"
        )
    return {"method": name, **_aggregate_seed_rows(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 fair eval suite")
    parser.add_argument("--run-dir", default="results/run_scaled_h5_gst")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=425)
    parser.add_argument("--human-num", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only re-eval RAISE R0/R1",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Comparison JSON path (default: <run-dir>/evals/p0_comparison.json)",
    )
    args = parser.parse_args()

    import crowd_sim  # noqa: F401
    from domains.crowdnav.reporting import write_json

    run_dir = os.path.abspath(args.run_dir)
    seeds = [int(args.base_seed) + i for i in range(int(args.n_seeds))]
    out_path = args.output or os.path.join(run_dir, "evals", "p0_comparison.json")

    report: Dict[str, Any] = {
        "created_utc": _utc_now(),
        "protocol": {
            "run_dir": run_dir,
            "human_num": int(args.human_num),
            "predict_method": "inferred",
            "randomization_regime": "without_random",
            "eval_episodes": int(args.episodes),
            "eval_seeds": seeds,
            "device": args.device,
            "notes": (
                "Eval seeds vary episode RNG only; policies were trained at seed 425. "
                "This is NOT multi-seed training. Full P0 training seeds remain pending."
            ),
        },
        "methods": {},
        "limitations": [
            "n_train_seeds=1 (only eval-seed aggregation)",
            "Baselines pretrained at paper H=20; here evaluated with H pinned to 5",
            "No fresh LegacyReward PPO train in this suite",
            "DS-RNN skipped (not reproduced locally)",
        ],
    }

    # 1) Best-ever R0
    report["methods"]["RAISE_best_ever_r00"] = eval_raise_multi_seed(
        run_dir=run_dir,
        candidate_id="mut_0060_v2",
        round_index=0,
        episodes=int(args.episodes),
        seeds=seeds,
        device=args.device,
        human_num=int(args.human_num),
    )

    # 2) Last-round R1 (same genome folder r01_mut_0060_v2)
    report["methods"]["RAISE_last_round_r01"] = eval_raise_multi_seed(
        run_dir=run_dir,
        candidate_id="mut_0060_v2",
        round_index=1,
        episodes=int(args.episodes),
        seeds=seeds,
        device=args.device,
        human_num=int(args.human_num),
    )

    if not args.skip_baselines:
        baselines = [
            ("SF", "trained_models/SF_no_rand", "00000.pt"),
            ("ORCA", "trained_models/ORCA_no_rand", "00000.pt"),
            ("CrowdNav++_GST", "trained_models/GST_predictor_non_rand", "41200.pt"),
        ]
        for name, path, ckpt in baselines:
            if not os.path.isdir(path):
                report["methods"][name] = {
                    "method": name,
                    "skipped": True,
                    "reason": f"missing {path}",
                }
                continue
            try:
                report["methods"][name] = eval_baseline_multi_seed(
                    name=name,
                    model_dir=path,
                    checkpoint=ckpt,
                    episodes=int(args.episodes),
                    seeds=seeds,
                    device=args.device,
                    human_num=int(args.human_num),
                    randomize=False,
                )
            except Exception as exc:  # noqa: BLE001
                report["methods"][name] = {
                    "method": name,
                    "skipped": True,
                    "reason": str(exc),
                }

    # Ranking table by mean scalar
    ranking = []
    for key, block in report["methods"].items():
        if block.get("skipped"):
            ranking.append({"key": key, "skipped": True, "reason": block.get("reason")})
            continue
        agg = (block.get("aggregate") or {}).get("scalar") or {}
        ranking.append(
            {
                "key": key,
                "method": block.get("method", key),
                "scalar_mean": agg.get("mean"),
                "scalar_std": agg.get("std"),
                "SR_mean": ((block.get("aggregate") or {}).get("SR") or {}).get("mean"),
                "CR_mean": ((block.get("aggregate") or {}).get("CR") or {}).get("mean"),
                "TR_mean": ((block.get("aggregate") or {}).get("TR") or {}).get("mean"),
            }
        )
    ranking_ok = [r for r in ranking if not r.get("skipped")]
    ranking_ok.sort(key=lambda r: float(r.get("scalar_mean") or float("-inf")), reverse=True)
    report["ranking_by_scalar"] = ranking_ok + [r for r in ranking if r.get("skipped")]

    write_json(out_path, report)
    print("\n======== P0 COMPARISON (mean over eval seeds) ========")
    for r in report["ranking_by_scalar"]:
        if r.get("skipped"):
            print(f"  SKIP {r['key']}: {r.get('reason')}")
            continue
        print(
            f"  {r['method']:<28} "
            f"SR={r['SR_mean']:.3f} CR={r['CR_mean']:.3f} TR={r['TR_mean']:.3f} "
            f"scalar={r['scalar_mean']:.3f}±{r['scalar_std']:.3f}"
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
