#!/usr/bin/env python
"""
Bootstrap surrogate dataset + model.

Implementation: ``crowd_nav.reward_search.surrogate.bootstrap.run_bootstrap``.
Design: ``crowd_nav/reward_search/surrogate/PLAN.md`` (locked v1).
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
    parser.add_argument("--out", default="data/surrogate_dataset")
    parser.add_argument("--model-out", default="artifacts/surrogate")
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--stage2-train-steps", type=int, default=8000)
    parser.add_argument(
        "--k2-unit",
        default="gradient_steps",
        choices=("gradient_steps", "env_steps"),
    )
    parser.add_argument("--llm", default="seed", help="LLM provider (seed|groq|...)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Stage II ShmemVecEnv workers (default 1 to avoid RAM OOM on Windows)",
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
    )
    print(json.dumps({k: result[k] for k in ("status", "out_dir", "model_dir", "n") if k in result}, indent=2))
    if result.get("status") == "ok":
        print("metrics:", json.dumps(result.get("metrics"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
