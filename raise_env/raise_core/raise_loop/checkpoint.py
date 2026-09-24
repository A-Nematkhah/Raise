"""
Persist / restore RAISE closed-loop run state.

Used for crash-safe resume (``RESUME.json`` + ``checkpoint.json`` under the run dir).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from raise_core.explore import RewardCandidate
from domains.crowdnav.reporting import candidate_to_dict, load_candidate_dict

_CHECKPOINT_NAME = "checkpoint.json"
_RESUME_NAME = "RESUME.json"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _serialize_pop(population: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    if not population:
        return []
    out: List[Dict[str, Any]] = []
    for cand in population:
        if hasattr(cand, "candidate_id"):
            out.append(candidate_to_dict(cand))
        elif isinstance(cand, dict):
            out.append(dict(cand))
    return out


def deserialize_population(
    rows: Optional[Sequence[Dict[str, Any]]],
    *,
    validator: Any = None,
) -> List[RewardCandidate]:
    """Rebuild ``RewardCandidate`` objects (re-validates code with ``validator``)."""
    if not rows:
        return []
    return [
        load_candidate_dict(dict(row), validator=validator)
        for row in rows
        if isinstance(row, dict)
    ]


def build_checkpoint(
    *,
    status: str,
    phase: str,
    epoch: int,
    next_epoch: int,
    population: Optional[Sequence[Any]] = None,
    ranked: Optional[Sequence[Any]] = None,
    to_label: Optional[Sequence[Any]] = None,
    labeled_ids_this_epoch: Optional[Sequence[str]] = None,
    global_best: Any = None,
    reflection: str = "",
    labels_since_refit: int = 0,
    n_labeled: int = 0,
    history: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable checkpoint payload."""
    payload: Dict[str, Any] = {
        "schema_version": "1",
        "status": str(status),
        "phase": str(phase),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "population": _serialize_pop(population),
        "ranked": _serialize_pop(ranked),
        "to_label": _serialize_pop(to_label),
        "labeled_ids_this_epoch": [str(x) for x in (labeled_ids_this_epoch or [])],
        "global_best": candidate_to_dict(global_best) if global_best is not None else None,
        "reflection": str(reflection or ""),
        "labels_since_refit": int(labels_since_refit),
        "n_labeled": int(n_labeled),
        "history": list(history or []),
        "config": dict(config or {}),
        "updated_at": _utc_now(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_checkpoint(run_dir: str, payload: Dict[str, Any]) -> str:
    """Write ``checkpoint.json`` and a small ``RESUME.json`` pointer; return ckpt path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    resume = {
        "checkpoint": _CHECKPOINT_NAME,
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "epoch": payload.get("epoch"),
        "updated_at": payload.get("updated_at") or _utc_now(),
    }
    with open(os.path.join(run_dir, _RESUME_NAME), "w", encoding="utf-8") as fh:
        json.dump(resume, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from ``run_dir`` if present."""
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None
