#!/usr/bin/env python
"""
One active-learning step (stub CLI).

Design: ``crowd_nav/reward_search/active_learning/PLAN.md``.
Requires a fitted surrogate under ``--surrogate``.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Active learning step (not implemented yet)")
    parser.add_argument("--surrogate", default="artifacts/surrogate")
    parser.add_argument("--queue-root", default="data/active_learning")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    from crowd_nav.reward_search.active_learning.loop import run_active_learning_step

    try:
        run_active_learning_step(
            surrogate_model_dir=args.surrogate,
            queue_root=args.queue_root,
            max_queries=int(args.max_queries),
            refit_every=int(args.refit_every),
            force_refit=bool(args.force_refit),
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        print("Read: crowd_nav/reward_search/active_learning/PLAN.md", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
