"""Compatibility shim — use ``domains.crowdnav``."""

from domains.crowdnav import *  # noqa: F401,F403
from domains.crowdnav import get_pack  # noqa: F401

__all__ = ["get_pack"]
