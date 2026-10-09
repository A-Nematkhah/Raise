#!/usr/bin/env python
"""Upgrade a highway Stage I dataset to the current state schema in place.

Recomputes ``ego.lane_index`` from ``ego.y`` and clears ``timeout`` on
off-road frames. The original files are copied to a sibling
``stage1_dataset_backup_<timestamp>/`` directory first.

Usage (from raise_env/):
  python scripts/migrate_highway_stage1_dataset.py
  python scripts/migrate_highway_stage1_dataset.py --dataset path/to/stage1_dataset
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

import argparse  # noqa: E402

from domains.highway.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DATASET,
    STATE_SCHEMA_VERSION,
    migrate_highway_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_STAGE1_DATASET)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    n = migrate_highway_dataset(args.dataset, backup=not args.no_backup)
    if n == 0:
        print(f"{args.dataset}: already at state_schema_version={STATE_SCHEMA_VERSION}")
    else:
        print(f"{args.dataset}: migrated {n} trajectories to state_schema_version={STATE_SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
