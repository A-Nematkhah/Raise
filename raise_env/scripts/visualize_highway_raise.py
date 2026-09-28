#!/usr/bin/env python
"""Offline plots + GIF for a highway RAISE run directory.

Examples::

    python scripts/visualize_highway_raise.py --run-dir results/highway_4h_...
    python scripts/plot_raise_run.py --run-dir results/highway_4h_...
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

os.chdir(_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Alias of --skip-viz")
    args = parser.parse_args()

    from domains.highway.post_run import write_highway_run_artifacts

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 1
    report = write_highway_run_artifacts(
        run_dir,
        episodes=int(args.episodes),
        seed=int(args.seed),
        skip_viz=bool(args.skip_viz or args.plots_only),
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
