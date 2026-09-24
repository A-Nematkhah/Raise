"""Compatibility shim — use ``domains.highway``."""

from domains.highway import *  # noqa: F401,F403
from domains.highway import get_pack  # noqa: F401

__all__ = ["get_pack"]
