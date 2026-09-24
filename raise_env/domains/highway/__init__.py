"""Highway domain pack."""

from __future__ import annotations

try:
    import raise_paths as _raise_paths  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from domains.highway.pack import get_pack

__all__ = ["get_pack"]
