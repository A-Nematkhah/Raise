"""
Surrogate model package (cheap predictor of Stage II short metrics).

See ``PLAN.md`` — contract locked 2026-09-20.
Default targets: SR, CR, TR (CrowdNav). Highway adds mean_speed /
mean_progress / soft_success via ``target_keys_for_domain``.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "1"

from raise_core.surrogate.model import (  # noqa: E402
    SurrogateModel,
    SurrogatePrediction,
)
from raise_core.surrogate.targets import (  # noqa: E402
    DEFAULT_TARGET_KEYS,
    HIGHWAY_TARGET_KEYS,
    target_keys_for_domain,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "DEFAULT_TARGET_KEYS",
    "HIGHWAY_TARGET_KEYS",
    "target_keys_for_domain",
    "SurrogateModel",
    "SurrogatePrediction",
]
