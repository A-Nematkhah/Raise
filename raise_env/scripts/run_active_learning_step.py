#!/usr/bin/env python
"""
One active-learning step.

Design: ``crowd_nav/reward_search/active_learning/PLAN.md`` (locked v1).
Requires a fitted surrogate under ``--surrogate``.
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
    parser = argparse.ArgumentParser(description="Active learning step")
    parser.add_argument("--surrogate", default="artifacts/surrogate")
    parser.add_argument("--queue-root", default="data/active_learning")
    parser.add_argument("--surrogate-dataset", default="data/surrogate_dataset")
    parser.add_argument("--candidates", default=None, help="stage1_population.json path")
    parser.add_argument("--stage1-dataset", default="data/stage1_dataset")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--fast", action="store_true", help="stub Stage II + smoke Score1")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--promote-threshold", type=float, default=0.0)
    args = parser.parse_args()

    from crowd_nav.reward_search.active_learning.loop import run_active_learning_step

    summary = run_active_learning_step(
        surrogate_model_dir=args.surrogate,
        queue_root=args.queue_root,
        max_queries=int(args.max_queries),
        refit_every=int(args.refit_every),
        force_refit=bool(args.force_refit),
        candidates_path=args.candidates,
        surrogate_dataset=args.surrogate_dataset,
        use_stub=bool(args.fast),
        seed=int(args.seed),
        promote_threshold=float(args.promote_threshold),
        stage1_dataset_path=args.stage1_dataset,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "executed"}, indent=2))
    if summary.get("status") == "error":
        print(summary.get("message", "error"), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
