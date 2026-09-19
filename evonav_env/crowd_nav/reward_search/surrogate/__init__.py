"""
Surrogate model package (cheap predictor of expensive stage metrics).

See ``PLAN.md`` in this directory for the full technical design.
Implementation starts from stubs below; do not wire into ``EvoNavPipeline``
until bootstrap + offline fit are green.
"""

from __future__ import annotations

__all__ = [
    "FEATURE_SCHEMA_VERSION",
]

FEATURE_SCHEMA_VERSION = "1"
