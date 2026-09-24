"""CrowdNav domain pack (default RAISE backend)."""

from __future__ import annotations

# Arm ``import crowd_sim`` / ``crowd_nav`` / ``rl`` via runtime/ on sys.path.
try:
    import raise_paths as _raise_paths  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from domains.crowdnav.pack import get_pack

__all__ = ["get_pack"]
