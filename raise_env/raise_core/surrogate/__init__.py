"""
Surrogate model package (cheap predictor of Stage II short metrics).

Default targets: SR, CR, TR (CrowdNav). Highway adds mean_speed /
mean_progress / soft_success via ``target_keys_for_domain``.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "1"

# Bump a domain's entry whenever its reward-state semantics change, so surrogate
# datasets/models built on the old states are not silently reused.
_DOMAIN_FEATURE_SCHEMA_VERSIONS = {"highway": "2"}


def feature_schema_version_for_domain(domain: str | None) -> str:
    key = str(domain or "crowdnav").strip().lower()
    return _DOMAIN_FEATURE_SCHEMA_VERSIONS.get(key, FEATURE_SCHEMA_VERSION)

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
    "feature_schema_version_for_domain",
    "DEFAULT_TARGET_KEYS",
    "HIGHWAY_TARGET_KEYS",
    "target_keys_for_domain",
    "SurrogateModel",
    "SurrogatePrediction",
]
