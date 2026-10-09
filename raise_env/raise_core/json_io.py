"""Crash-safe JSON writes shared by run-state checkpoints."""

from __future__ import annotations

import json
import os
import time
from typing import Any

_REPLACE_RETRIES = 5
_REPLACE_BACKOFF_S = 0.05


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> str:
    """Write ``payload`` to ``path`` via temp file + ``os.replace``.

    A crash mid-write leaves the previous file intact instead of truncated JSON.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Windows: os.replace raises PermissionError while another process
        # (viewer, antivirus) briefly holds the target open.
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


__all__ = ["atomic_write_json"]
