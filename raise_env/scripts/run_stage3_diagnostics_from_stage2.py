#!/usr/bin/env python
"""Re-train Stage-2 diagnostic candidates at Stage-3 K3 budget with diagnostics.

From raise_env/:

  python scripts/run_stage3_diagnostics_from_stage2.py \\
    --run-dir results/highway_4h_20260926_211502
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

from raise_core.presets import CLOSED_LOOP_PROFILES  # noqa: E402

DIR_RE = re.compile(r"^r(\d+)_(.+)$")
TARGET_IDS = {
    "ini_0000",
    "ini_0001",
    "ini_0002",
    "ini_0003",
    "ini_0004",
    "ini_0005",
    "mut_0009",
    "mut_0014",
    "mut_0018",
    "mut_0019",
    "mut_0020",
    "cro_0006",
    "cro_0007",
    "cro_0011",
    "cro_0012",
}


def _stage2_dirs(run_dir: str) -> list[tuple[int, str, str]]:
    root = os.path.join(run_dir, "closed_loop", "stage2_train")
    out: list[tuple[int, str, str]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        m = DIR_RE.match(name)
        if not m:
            continue
        round_i, cid = int(m.group(1)), m.group(2)
        if cid not in TARGET_IDS:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append((round_i, cid, path))
    return out


def _ensure_codes(run_dir: str) -> dict[str, str]:
    recovered = os.path.join(run_dir, "recovered_stage2_diag_candidates.json")
    if not os.path.isfile(recovered):
        env = {**os.environ, "PYTHONPATH": _ROOT}
        rc = subprocess.call(
            [sys.executable, os.path.join(_SCRIPTS, "_recover_diag_candidates.py"), run_dir],
            cwd=_ROOT,
            env=env,
        )
        if not os.path.isfile(recovered):
            raise FileNotFoundError(
                f"Code recovery failed (exit={rc}); missing {recovered}"
            )
    payload = json.load(open(recovered, encoding="utf-8"))
    codes = {cid: row["code"] for cid, row in payload.items() if cid in TARGET_IDS}
    missing = sorted(TARGET_IDS - set(codes))
    if missing:
        raise RuntimeError(
            "Missing reward codes for: "
            + ", ".join(missing)
            + ". Provide recovered_stage2_diag_candidates.json or population "
            "JSON entries that include full compute_reward source."
        )
    return codes


def main() -> int:
    profile = dict(CLOSED_LOOP_PROFILES["highway_4h"])
    k3 = int(profile["stage3_k3"])
    base_seed = int(profile["seed"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/highway_4h_20260926_211502",
        help="Completed Stage-2 diagnostics run directory",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: <run-dir>/stage3_train_diagnostics",
    )
    parser.add_argument("--k3", type=int, default=k3, help=f"Stage3 budget (default {k3})")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: only first N candidates (smoke)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip candidates that already have both diagnostics JSONL files",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_root = (
        os.path.abspath(args.out_dir)
        if str(args.out_dir).strip()
        else os.path.join(run_dir, "stage3_train_diagnostics")
    )
    os.makedirs(out_root, exist_ok=True)

    print("=== Stage3 diagnostics from Stage2 candidates ===")
    print(f"  run_dir: {run_dir}")
    print(f"  out_dir: {out_root}")
    print(f"  K3 (train_env_steps): {int(args.k3)}")
    print(f"  base_seed: {base_seed}  (train seed = base_seed + round_index)")
    print("  diagnostics_groundtruth: True")
    print("  warm_start: False (isolate budget; scratch train at K3)")
    print("=================================================")

    pairs = _stage2_dirs(run_dir)
    if not pairs:
        print("No Stage2 diagnostic dirs found.", file=sys.stderr)
        return 2
    if int(args.limit) > 0:
        pairs = pairs[: int(args.limit)]

    try:
        codes = _ensure_codes(run_dir)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2

    from domains.highway.adapter import HighwayPPOTrainer
    from raise_core.domains import load_domain, make_validator_for_domain

    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    trainer = HighwayPPOTrainer(stage="stage3")

    t_all = time.perf_counter()
    done = 0
    for round_i, cid, s2_path in pairs:
        out_name = f"r{int(round_i):02d}_{cid}"
        out_dir = os.path.join(out_root, out_name)
        gt = os.path.join(out_dir, "diagnostics_groundtruth.jsonl")
        ro = os.path.join(out_dir, "diagnostics_rollout.jsonl")
        if args.skip_existing and os.path.isfile(gt) and os.path.isfile(ro):
            print(f"skip existing {cid} -> {out_dir}")
            done += 1
            continue

        reward_fn = validator.validate_code(codes[cid])
        cand = SimpleNamespace(
            candidate_id=cid,
            reward_fn=reward_fn,
            code=codes[cid],
            metadata={},
        )
        cfg = SimpleNamespace(
            train_env_steps=int(args.k3),
            eval_episodes=4,
            seed=base_seed,
            device="cpu",
            output_root=out_root,
            highway_n_envs=1,
            highway_warm_start=False,
            highway_eval_mode="holdout_only",
            highway_diagnostics_groundtruth=True,
        )
        print(
            f"\n>>> Stage3-diag {cid} round={round_i} "
            f"seed={base_seed + round_i} K3={int(args.k3)} "
            f"(stage2 ref: {s2_path})"
        )
        t0 = time.perf_counter()
        trainer.train_and_eval(cand, round_index=round_i, config=cfg)
        print(f"<<< done {cid} in {time.perf_counter() - t0:.1f}s -> {out_dir}")
        done += 1

    print(
        f"\nFinished {done}/{len(pairs)} candidates in "
        f"{time.perf_counter() - t_all:.1f}s -> {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
