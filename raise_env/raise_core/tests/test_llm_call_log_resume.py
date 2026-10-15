"""P2-3: LLMCallLog serial continues across process restarts."""

from __future__ import annotations

import json
from pathlib import Path

from raise_core.llm_call_log import CALLS_FILE, LLMCallLog


def test_call_id_serial_continues_from_existing_jsonl(tmp_path: Path):
    log_dir = tmp_path / "closed_loop"
    log_dir.mkdir()
    path = log_dir / CALLS_FILE
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"event": "request", "call_id": "20261014T195131_0048_random"})
            + "\n"
        )
        fh.write(
            json.dumps({"event": "response", "call_id": "20261014T195131_0048_random"})
            + "\n"
        )
    log = LLMCallLog(str(log_dir))
    nxt = log.next_call_id("mutation")
    assert nxt.endswith("_0049_mutation")
