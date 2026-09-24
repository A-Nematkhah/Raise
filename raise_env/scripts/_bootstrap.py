"""Ensure ``domains/crowdnav/runtime`` is on ``sys.path`` (see ``raise_paths``)."""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPTS, ".."))
CROWDNAV_RUNTIME = os.path.join(ROOT, "domains", "crowdnav", "runtime")

for _p in (ROOT, CROWDNAV_RUNTIME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export canonical helpers when available.
try:
    from raise_paths import (  # noqa: E402
        CROWDNAV_ACTIVE_LEARNING,
        CROWDNAV_STAGE1_DATASET,
        HIGHWAY_STAGE1_DATASET,
        ensure_raise_paths,
    )
except ImportError:  # pragma: no cover
    CROWDNAV_STAGE1_DATASET = "domains/crowdnav/data/stage1_dataset"
    CROWDNAV_ACTIVE_LEARNING = "domains/crowdnav/data/active_learning"
    HIGHWAY_STAGE1_DATASET = "domains/highway/data/stage1_dataset"

    def ensure_raise_paths() -> str:
        return ROOT
