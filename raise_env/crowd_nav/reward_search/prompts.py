"""
RAISE Appendix D prompt templates — **backward-compatible re-export**.

Canonical source: ``crowd_nav.domains.crowdnav.prompts`` (CrowdNav domain pack).
Importing from this module must remain byte-identical for Algorithm 1 baseline.
"""

from __future__ import annotations

from crowd_nav.domains.crowdnav.prompts import (  # noqa: F401
    D1_BATCH_USER_PROMPT,
    D1_SYSTEM_PROMPT,
    D1_USER_PROMPT,
    D2_CROSSOVER_PROMPT,
    D2_MUTATION_PROMPT,
    D3_SYSTEM_PROMPT,
    D3_USER_PROMPT,
    D4_EXTERNAL_KNOWLEDGE,
    D5_SEED_FUNCTION,
    PROMPT_MEMORY_EXAMPLE,
    format_d1_initial,
    format_d1_initial_batch,
    format_d2_crossover,
    format_d2_mutation,
    format_d3_refinement,
    format_d3_repair,
)

__all__ = [
    "D1_BATCH_USER_PROMPT",
    "D1_SYSTEM_PROMPT",
    "D1_USER_PROMPT",
    "D2_CROSSOVER_PROMPT",
    "D2_MUTATION_PROMPT",
    "D3_SYSTEM_PROMPT",
    "D3_USER_PROMPT",
    "D4_EXTERNAL_KNOWLEDGE",
    "D5_SEED_FUNCTION",
    "PROMPT_MEMORY_EXAMPLE",
    "format_d1_initial",
    "format_d1_initial_batch",
    "format_d2_crossover",
    "format_d2_mutation",
    "format_d3_refinement",
    "format_d3_repair",
]
