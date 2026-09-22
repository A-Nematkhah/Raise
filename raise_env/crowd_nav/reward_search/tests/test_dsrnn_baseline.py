"""Unit tests for DS-RNN Table 1 Option A / Option B helpers."""

from __future__ import annotations

import json
import os

from crowd_nav.reward_search.dsrnn_baseline import (
    DEFAULT_DSRNN_NO_RAND_DIR,
    DEFAULT_DSRNN_RAND_DIR,
    DSRNN_NOT_REPRODUCED_MSG,
    dsrnn_placeholder,
    format_dsrnn_table1_row,
    resolve_checkpoint,
    resolve_dsrnn_dirs,
)


def test_placeholder_has_no_numeric_metrics():
    entry = dsrnn_placeholder()
    assert entry["skipped"] is True
    assert entry["not_reproduced"] is True
    assert entry["reason"] == DSRNN_NOT_REPRODUCED_MSG
    for key in ("SR", "CR", "TR", "NT", "PL", "ITR", "SD", "sr", "cr"):
        assert key not in entry


def test_skip_dsrnn_mode():
    mode, dirs = resolve_dsrnn_dirs(skip_dsrnn=True)
    assert mode == "skip"
    assert dirs["no_rand"] == DEFAULT_DSRNN_NO_RAND_DIR
    assert dirs["rand"] == DEFAULT_DSRNN_RAND_DIR


def test_evaluate_mode_defaults():
    mode, dirs = resolve_dsrnn_dirs(skip_dsrnn=False)
    assert mode == "evaluate"
    assert dirs["no_rand"] == DEFAULT_DSRNN_NO_RAND_DIR
    assert dirs["rand"] == DEFAULT_DSRNN_RAND_DIR


def test_format_placeholder_row_mentions_paper_table():
    entry = dsrnn_placeholder(reason=f"{DSRNN_NOT_REPRODUCED_MSG} (--skip-dsrnn)")
    line = format_dsrnn_table1_row(entry, randomize=False)
    assert "DS-RNN" in line
    assert "not reproduced locally" in line
    assert "paper Table 1" in line
    # Must not look like a fabricated mean±std metric line.
    assert "+/-" not in line
    assert "±" not in line


def test_missing_checkpoint_returns_none(tmp_path):
    empty = tmp_path / "ds_rnn_no_rand"
    empty.mkdir()
    assert resolve_checkpoint(str(empty), "auto") is None
    assert resolve_checkpoint(str(empty), "00000.pt") is None


def test_auto_picks_latest_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "00000.pt").write_bytes(b"a")
    (ckpt_dir / "00100.pt").write_bytes(b"b")
    assert resolve_checkpoint(str(tmp_path), "auto") == "00100.pt"
    assert resolve_checkpoint(str(tmp_path), "00000.pt") == "00000.pt"
    assert resolve_checkpoint(str(tmp_path), "99999.pt") is None


def test_placeholder_json_roundtrip_never_fills_metrics(tmp_path):
    """Option B rows must serialize without inventing SR/CR values."""
    entry = dsrnn_placeholder()
    path = tmp_path / "dsrnn.json"
    path.write_text(json.dumps({"DS-RNN": entry}), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))["DS-RNN"]
    assert loaded["not_reproduced"] is True
    assert "SR" not in loaded
    assert loaded["reason"] == DSRNN_NOT_REPRODUCED_MSG
