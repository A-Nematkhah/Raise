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


def ensure_raise_paths() -> str:
    """Insert raise_env root + CrowdNav runtime onto ``sys.path``. Idempotent."""
    for path in (ROOT, CROWDNAV_RUNTIME):
        if path not in sys.path:
            sys.path.insert(0, path)
    return ROOT


# Importing this module always arms paths (scripts / conftest / domain packs).
ensure_raise_paths()
