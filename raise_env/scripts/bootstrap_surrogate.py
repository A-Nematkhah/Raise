#!/usr/bin/env python
"""
Bootstrap surrogate dataset + model.

Implementation: ``crowd_nav.reward_search.surrogate.bootstrap.run_bootstrap``.
Design: ``crowd_nav/reward_search/surrogate/PLAN.md`` (locked v1).

Match the closed-loop Stage II env when warm-starting a RAISE run.
Defaults already match the 12h PROFILE (K2=8000, human_num=5, with_random):

  python scripts/bootstrap_surrogate.py
  python scripts/run_raise_12h.py --warm-surrogate artifacts/surr_warm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Stage-II surrogate")
    parser.add_argument("--stage1-dataset", default="data/stage1_dataset")
    parser.add_argument("--out", default="artifacts/surr_warm/dataset")
    parser.add_argument("--model-out", default="artifacts/surr_warm/model")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=40,
        help="Bootstrap labels to collect (default 40; gate needs ≥24)",
    )
    parser.add_argument(
        "--stage2-train-steps",
        type=int,
        default=8000,
        help="K2 budget per bootstrap label (default 8000 gradient steps)",
    )
    parser.add_argument(
        "--k2-unit",
        default="gradient_steps",
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
            "1 forces phase=test)"
        ),
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=5,
        help="Crowd size — match closed-loop (12h default 5; paper 20)",
    )
    parser.add_argument(
        "--regime",
        default="with_random",
        choices=("with_random", "without_random"),
        help="Randomization regime — match closed-loop (12h default with_random)",
    )
    parser.add_argument(
        "--predict-method",
        default="inferred",
        choices=("inferred", "truth", "none"),
        help="GST obs mode — match closed-loop (default inferred)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Episode horizon steps (default 100)",
    )
    parser.add_argument("--fast", action="store_true", help="stub trainer + smoke Score1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args()

    # Windows spawn workers re-import Config → get_args() reads sys.argv.
    # Isolate bootstrap flags (same pattern as run_raise / run_stage3_smoke).
    sys.argv = [sys.argv[0], "--seed", str(args.seed)]
    if str(args.device).lower() != "cuda":
        sys.argv.append("--no-cuda")

    from crowd_nav.reward_search.surrogate.bootstrap import run_bootstrap

    result = run_bootstrap(
        stage1_dataset_path=args.stage1_dataset,
        out_dir=args.out,
        model_dir=args.model_out,
        n_candidates=int(args.n_candidates),
        stage2_train_steps=int(args.stage2_train_steps),
        k2_unit=str(args.k2_unit),
        use_stub=bool(args.fast),
        seed=int(args.seed),
        force=bool(args.force),
        llm_provider=str(args.llm),
        device=str(args.device),
        num_processes=int(args.num_processes),
        human_num=int(args.human_num),
        predict_method=str(args.predict_method),
        randomization_regime=str(args.regime),
        horizon_steps=int(args.horizon),
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "out_dir", "model_dir", "n") if k in result},
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
