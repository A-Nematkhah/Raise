"""
Ensure domain runtime trees are importable.

Physical layout (post domain isolation)::

    domains/crowdnav/runtime/{crowd_sim,crowd_nav,rl,gst_updated}
    domains/crowdnav/data/
    domains/highway/data/

Legacy top-level imports (``import crowd_sim``, ``import crowd_nav``, ``import rl``)
keep working by putting ``domains/crowdnav/runtime`` on ``sys.path``.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
CROWDNAV_RUNTIME = os.path.join(ROOT, "domains", "crowdnav", "runtime")
CROWDNAV_DATA = os.path.join(ROOT, "domains", "crowdnav", "data")
HIGHWAY_DATA = os.path.join(ROOT, "domains", "highway", "data")

# Default relative paths (from raise_env cwd) for datasets / GST trees.
CROWDNAV_STAGE1_DATASET = "domains/crowdnav/data/stage1_dataset"
CROWDNAV_ACTIVE_LEARNING = "domains/crowdnav/data/active_learning"
CROWDNAV_SURROGATE_DATASET = "domains/crowdnav/data/surrogate_dataset"
HIGHWAY_STAGE1_DATASET = "domains/highway/data/stage1_dataset"
HIGHWAY_ACTIVE_LEARNING = "domains/highway/data/active_learning"
HIGHWAY_SURROGATE_DATASET = "domains/highway/data/surrogate_dataset"
GST_RUNTIME_PREFIX = "domains/crowdnav/runtime/gst_updated"


def ensure_raise_paths() -> str:
    """Insert raise_env root + CrowdNav runtime onto ``sys.path``. Idempotent."""
    for path in (ROOT, CROWDNAV_RUNTIME):
        if path not in sys.path:
            sys.path.insert(0, path)
    return ROOT


# Importing this module always arms paths (scripts / conftest / domain packs).
ensure_raise_paths()
