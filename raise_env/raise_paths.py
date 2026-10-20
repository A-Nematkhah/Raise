"""Put the ``raise_env`` root and the active vendored simulator on ``sys.path``.

Scripts / conftest / ``domains.crowdnav`` import this. Only CrowdNav++ is
vendored, and its ``crowd_sim`` package is the active simulator implementation.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
CROWDNAVPP_VENDOR_DIR = os.path.join(ROOT, "third_party", "crowdnavpp")


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_inside(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath((_normalized(path), _normalized(directory))) == _normalized(
            directory
        )
    except ValueError:
        return False


def ensure_raise_paths() -> str:
    """Expose the RAISE root and CrowdNav++, rejecting conflicting imports."""
    loaded = sys.modules.get("crowd_sim")
    loaded_from = getattr(loaded, "__file__", None)
    if loaded_from and not _is_inside(loaded_from, CROWDNAVPP_VENDOR_DIR):
        raise RuntimeError(
            "A conflicting crowd_sim package was imported before raise_paths; "
            "restart Python to load the vendored CrowdNav++ package cleanly."
        )

    for path in (ROOT, CROWDNAVPP_VENDOR_DIR):
        normalized = _normalized(path)
        sys.path[:] = [entry for entry in sys.path if not entry or _normalized(entry) != normalized]
        sys.path.insert(0, path)
    return ROOT


ensure_raise_paths()
