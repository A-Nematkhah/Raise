"""
Bootstrap: build label population → Stage-II-short labels → fit surrogate (stub).

CLI entry will live at ``scripts/bootstrap_surrogate.py``.
"""

from __future__ import annotations

from typing import Any, Optional


def run_bootstrap(
    *,
    stage1_dataset_path: str = "data/stage1_dataset",
    out_dir: str = "data/surrogate_dataset",
    model_dir: str = "artifacts/surrogate",
    n_candidates: int = 60,
    stage2_train_steps: int = 8_000,
    use_stub: bool = False,
    seed: int = 425,
    force: bool = False,
) -> dict[str, Any]:
    """
    End-to-end surrogate bootstrap (idempotent unless ``force``).

    See ``PLAN.md`` §5 for the step list.
    """
    raise NotImplementedError(
        "surrogate.bootstrap.run_bootstrap — see surrogate/PLAN.md"
    )
