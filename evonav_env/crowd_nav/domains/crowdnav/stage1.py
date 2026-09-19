"""
CrowdNav Stage I — Score1 / smoke scoring resolved through the domain pack.

Implementation stays in ``crowd_nav.reward_search.{scoring,dataset,rules}``
(baseline-locked). This module is the pack-owned entry point so the pipeline
does not hard-code CrowdNav Score1 imports.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_DATASET = "data/stage1_dataset"


def resolve_dataset_path(dataset_path: Optional[str] = None) -> str:
    """Prefer explicit path; else CrowdNav default."""
    if dataset_path is not None and str(dataset_path).strip():
        return str(dataset_path).strip()
    return DEFAULT_STAGE1_DATASET


def make_score_fn(
    *,
    mode: str = "dataset",
    dataset_path: Optional[str] = None,
) -> Tuple[Callable[..., Any], Optional[Any]]:
    """
    Build Stage I ``score_fn`` for CrowdNav.

    Returns ``(score_fn, dataset_or_None)``. Dataset is ``None`` for smoke mode.
    """
    key = str(mode).strip().lower()
    if key == "smoke":
        from crowd_nav.reward_search.scoring import make_smoke_score_fn

        logger.warning(
            "Using make_smoke_score_fn (opt-in fast fixture) — not paper Score1"
        )
        return make_smoke_score_fn(), None

    if key != "dataset":
        raise ValueError(f"Unknown score1_mode: {mode!r}")

    from crowd_nav.reward_search.dataset import load_stage1_dataset
    from crowd_nav.reward_search.scoring import make_score1_fn

    path = resolve_dataset_path(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage I dataset not found at {path}. "
            f"Run: python scripts/collect_stage1_dataset.py --out {path}"
        )
    dataset = load_stage1_dataset(path)
    logger.info(
        "Loaded Stage I dataset from %s (%d scenarios)", path, len(dataset)
    )
    return make_score1_fn(dataset), dataset
