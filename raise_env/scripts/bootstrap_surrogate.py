#!/usr/bin/env python
"""
Bootstrap surrogate dataset + model.

Implementation: ``raise_core.surrogate.bootstrap.run_bootstrap``.

Match the closed-loop Stage II env when warm-starting a RAISE run.

CrowdNav (12h defaults)::

  python scripts/bootstrap_surrogate.py
  python scripts/run_raise_12h.py --warm-surrogate artifacts/surr_warm

Highway::

  python scripts/bootstrap_surrogate.py --domain highway --llm seed
  python scripts/run_raise_highway_4h.py --warm-surrogate artifacts/highway_surr_warm
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401
import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Stage-II surrogate")
    parser.add_argument(
        "--domain",
        default="crowdnav",
        choices=("crowdnav", "highway"),
        help="Domain pack (default crowdnav; use highway for highway-fast-v0)",
    )
    parser.add_argument("--stage1-dataset", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model-out", default=None)
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=40,
        help="Bootstrap labels to collect (default 40; gate needs ≥24)",
    )
    parser.add_argument(
        "--stage2-train-steps",
        type=int,
        default=None,
        help="K2 budget per bootstrap label",
    )
    parser.add_argument(
        "--k2-unit",
        default=None,
        choices=("gradient_steps", "env_steps"),
    )
    parser.add_argument(
        "--llm",
        default="groq",
        help="LLM provider (default groq — match 12h run)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Train device (default cuda — match 12h run)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help=(
            "Stage II ShmemVecEnv workers (default 2 → phase=train; "
            "1 forces phase=test). Highway uses 1."
        ),
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=5,
        help="Crowd size — CrowdNav only (default 5)",
    )
    parser.add_argument(
        "--regime",
        default="with_random",
        choices=("with_random", "without_random"),
        help="Randomization regime — CrowdNav only",
    )
    parser.add_argument(
        "--predict-method",
        default="inferred",
        choices=("inferred", "truth", "none"),
        help="GST obs mode — CrowdNav only",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Episode horizon steps (CrowdNav; highway uses env duration)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Stage II eval episodes (default: domain-specific)",
    )
    parser.add_argument("--fast", action="store_true", help="stub trainer + smoke Score1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args()

    domain = str(args.domain).strip().lower()
    if domain == "highway":
        stage1 = args.stage1_dataset or "domains/highway/data/stage1_dataset"
        out = args.out or "artifacts/highway_surr_warm/dataset"
        model_out = args.model_out or "artifacts/highway_surr_warm/model"
        k2 = int(args.stage2_train_steps) if args.stage2_train_steps is not None else 15_000
        k2_unit = args.k2_unit or "env_steps"
        nproc = 1
    else:
        stage1 = args.stage1_dataset or "domains/crowdnav/data/stage1_dataset"
        out = args.out or "artifacts/surr_warm/dataset"
        model_out = args.model_out or "artifacts/surr_warm/model"
        k2 = int(args.stage2_train_steps) if args.stage2_train_steps is not None else 8000
        k2_unit = args.k2_unit or "gradient_steps"
        nproc = int(args.num_processes)

    # Windows spawn workers re-import Config → get_args() reads sys.argv.
    sys.argv = [sys.argv[0], "--seed", str(args.seed)]
    if str(args.device).lower() != "cuda":
        sys.argv.append("--no-cuda")

    from raise_core.surrogate.bootstrap import run_bootstrap

    result = run_bootstrap(
        domain=domain,
        stage1_dataset_path=stage1,
        out_dir=out,
        model_dir=model_out,
        n_candidates=int(args.n_candidates),
        stage2_train_steps=k2,
        k2_unit=str(k2_unit),
        use_stub=bool(args.fast),
        seed=int(args.seed),
        force=bool(args.force),
        llm_provider=str(args.llm),
        device=str(args.device),
        num_processes=nproc,
        human_num=int(args.human_num),
        predict_method=str(args.predict_method),
        randomization_regime=str(args.regime),
        horizon_steps=int(args.horizon),
        eval_episodes=args.eval_episodes,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "domain", "out_dir", "model_dir", "n")
                if k in result
            },
            indent=2,
        )
    )
    if result.get("status") == "ok":
        print("metrics:", json.dumps(result.get("metrics"), indent=2))
        man = result.get("manifest") or {}
        print(
            "env:",
            json.dumps(
                {
                    "domain": man.get("domain"),
                    "human_num": man.get("human_num"),
                    "regime": man.get("randomization_regime"),
                    "predict_method": man.get("predict_method"),
                    "K2": man.get("stage2_train_steps"),
                    "k2_unit": man.get("k2_unit"),
                },
                indent=2,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
