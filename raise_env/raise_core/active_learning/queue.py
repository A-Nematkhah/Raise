"""
Active-learning queue persistence (jsonl).

Default: ``data/active_learning/queue.jsonl`` + ``done.jsonl`` + ``manifest.json``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from raise_core.active_learning.query import QueryItem

_QUEUE = "queue.jsonl"
_DONE = "done.jsonl"
_STEPS = "steps.jsonl"
_MANIFEST = "manifest.json"


def default_queue_root() -> str:
    return "data/active_learning"


def _ensure_dir(root: str) -> None:
    os.makedirs(root, exist_ok=True)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _queue_path(root: str) -> str:
    return os.path.join(root, _QUEUE)


def _done_path(root: str) -> str:
    return os.path.join(root, _DONE)


def _steps_path(root: str) -> str:
    return os.path.join(root, _STEPS)


def _manifest_path(root: str) -> str:
    return os.path.join(root, _MANIFEST)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            fh.write("\n")


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
        fh.write("\n")


def read_manifest(root: str) -> Dict[str, Any]:
    path = _manifest_path(root)
    if not os.path.isfile(path):
        return {
            "n_done_since_refit": 0,
            "n_done_total": 0,
            "n_enqueued_total": 0,
        }
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def write_manifest(root: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(root)
    with open(_manifest_path(root), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def pending_items(root: str) -> List[QueryItem]:
    return [QueryItem.from_dict(row) for row in _load_jsonl(_queue_path(root))]


def enqueue(root: str, items: List[QueryItem]) -> None:
    """Append query items to the pending queue."""
    if not items:
        return
    _ensure_dir(root)
    for item in items:
        _append_jsonl(_queue_path(root), item.to_dict())
    man = read_manifest(root)
    man["n_enqueued_total"] = int(man.get("n_enqueued_total") or 0) + len(items)
    man["updated_at"] = _utc_now()
    write_manifest(root, man)


def dequeue_batch(root: str, *, limit: int = 5) -> List[QueryItem]:
    """
    Pop up to ``limit`` highest-priority pending items and rewrite the queue.
    """
    rows = _load_jsonl(_queue_path(root))
    if not rows or limit <= 0:
        return []
    items = [QueryItem.from_dict(r) for r in rows]
    items.sort(key=lambda q: q.priority, reverse=True)
    taken = items[: int(limit)]
    remaining = items[int(limit) :]
    _write_jsonl(_queue_path(root), [q.to_dict() for q in remaining])
    return taken


def mark_done(root: str, item: QueryItem, *, result: Dict[str, Any]) -> None:
    """Append a finished query + result blob to ``done.jsonl``."""
    _ensure_dir(root)
    row = {
        "finished_at": _utc_now(),
        "item": item.to_dict(),
        "result": dict(result or {}),
    }
    _append_jsonl(_done_path(root), row)
    man = read_manifest(root)
    man["n_done_total"] = int(man.get("n_done_total") or 0) + 1
    if str(item.kind) == "stage2_label" and str(result.get("status")) == "ok":
        man["n_done_since_refit"] = int(man.get("n_done_since_refit") or 0) + 1
    man["updated_at"] = _utc_now()
    write_manifest(root, man)


def append_step(root: str, summary: Dict[str, Any]) -> None:
    _ensure_dir(root)
    row = dict(summary)
    row.setdefault("ts", _utc_now())
    _append_jsonl(_steps_path(root), row)


def reset_refit_counter(root: str) -> None:
    man = read_manifest(root)
    man["n_done_since_refit"] = 0
    man["last_refit_at"] = _utc_now()
    write_manifest(root, man)


def recent_fingerprints(root: str, *, limit: int = 50) -> List[List[float]]:
    """Fingerprints from recent done stage2_label results / payloads."""
    rows = _load_jsonl(_done_path(root))
    fps: List[List[float]] = []
    for row in reversed(rows):
        item = row.get("item") or {}
        payload = item.get("payload") or {}
        fp = payload.get("behavior_fingerprint")
        if isinstance(fp, list) and fp:
            try:
                fps.append([float(x) for x in fp])
            except (TypeError, ValueError):
                continue
        if len(fps) >= limit:
            break
    return fps
