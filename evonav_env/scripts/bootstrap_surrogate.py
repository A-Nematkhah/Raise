#!/usr/bin/env python
"""
Bootstrap surrogate dataset + model (stub CLI).

Implementation: ``crowd_nav.reward_search.surrogate.bootstrap.run_bootstrap``.
Design: ``crowd_nav/reward_search/surrogate/PLAN.md``.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap surrogate (not implemented yet)")
    parser.add_argument("--stage1-dataset", default="data/stage1_dataset")
    parser.add_argument("--out", default="data/surrogate_dataset")
    parser.add_argument("--model-out", default="artifacts/surrogate")
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--stage2-train-steps", type=int, default=8000)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args()

    from crowd_nav.reward_search.surrogate.bootstrap import run_bootstrap

    try:
        run_bootstrap(
            stage1_dataset_path=args.stage1_dataset,
            out_dir=args.out,
            model_dir=args.model_out,
            n_candidates=args.n_candidates,
            stage2_train_steps=args.stage2_train_steps,
            use_stub=bool(args.fast),
            seed=int(args.seed),
            force=bool(args.force),
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        print("Read: crowd_nav/reward_search/surrogate/PLAN.md", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
