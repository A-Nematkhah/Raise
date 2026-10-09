"""Append-only JSONL helpers for closed-loop epochs."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from raise_core.json_io import utc_now_z



def closed_loop_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, "closed_loop")
    os.makedirs(path, exist_ok=True)
    return path


def append_epoch_record(output_dir: str, record: Dict[str, Any]) -> str:
    root = closed_loop_dir(output_dir)
    path = os.path.join(root, "epochs.jsonl")
    row = dict(record)
    row.setdefault("ts", utc_now_z())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str))
        fh.write("\n")
    return path


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def read_epochs(output_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(closed_loop_dir(output_dir), "epochs.jsonl")
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
