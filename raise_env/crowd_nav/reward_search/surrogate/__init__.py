"""
Surrogate model package (cheap predictor of Stage II short metrics).

See ``PLAN.md`` — contract locked 2026-09-20.
Primary targets: SR, CR, TR. Do not wire into ``EvoNavPipeline`` until
bootstrap + offline fit are green.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "1"

from crowd_nav.reward_search.surrogate.model import (  # noqa: E402
    SurrogateModel,
    SurrogatePrediction,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "SurrogateModel",
    "SurrogatePrediction",
]
