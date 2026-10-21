"""Helpers for ``evaluate_crowdnav_generalization`` (importable from tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_stage3_checkpoint(raw: Any, run_dir: Path) -> Path:
    if not raw:
        raise ValueError("final_candidate.json has no Stage III checkpoint_path")
    path = Path(str(raw))
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_dir / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Stage III PPO checkpoint not found; checked: "
        + ", ".join(str(candidate.resolve()) for candidate in candidates)
    )


__all__ = ["resolve_stage3_checkpoint"]
