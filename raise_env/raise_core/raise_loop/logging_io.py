"""Crash-safe JSON / JSONL helpers for closed-loop epochs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from raise_core.json_io import atomic_write_json, atomic_write_text, utc_now_z

logger = logging.getLogger(__name__)

_EPOCHS_FILE = "epochs.jsonl"


def closed_loop_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, "closed_loop")
    os.makedirs(path, exist_ok=True)
    return path


def append_epoch_record(output_dir: str, record: Dict[str, Any]) -> str:
    """Record one epoch; rows of this epoch or later are replaced, not duplicated.

    A run killed after writing epoch ``g`` but before its checkpoint repeats
    epoch ``g`` on resume, so the rewritten file keeps exactly one row per
    epoch. The file is replaced atomically.
    """
    path = os.path.join(closed_loop_dir(output_dir), _EPOCHS_FILE)
    row = dict(record)
    row.setdefault("ts", utc_now_z())
    epoch = row.get("epoch")
    kept = [r for r in read_epochs(output_dir) if epoch is None or _epoch_of(r) < int(epoch)]
    text = "".join(
        json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in [*kept, row]
    )
    return atomic_write_text(path, text)


def _epoch_of(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("epoch"))
    except (TypeError, ValueError):
        return -1


def write_json(path: str, payload: Any) -> None:
    atomic_write_json(path, payload, default=str)


def read_epochs(output_dir: str) -> List[Dict[str, Any]]:
    """Epoch rows in file order; a malformed line (torn write) is skipped with a warning."""
    path = os.path.join(closed_loop_dir(output_dir), _EPOCHS_FILE)
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
