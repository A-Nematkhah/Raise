"""Put the ``raise_env`` root (and vendored third-party packages) on ``sys.path``.

Scripts / conftest / ``domains.crowdnav`` import this. The vendored CrowdNav
simulator uses absolute ``crowd_sim.*`` imports, so its parent directory must
be importable as-is.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
CROWDNAV_VENDOR_DIR = os.path.join(ROOT, "third_party", "crowdnav")


def ensure_raise_paths() -> str:
    """Insert the raise_env root and the vendored CrowdNav dir onto ``sys.path``. Idempotent."""
    for path in (CROWDNAV_VENDOR_DIR, ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    return ROOT


ensure_raise_paths()
