"""P2-1: provenance.json and file logging."""

from __future__ import annotations

import json
from pathlib import Path

from raise_core import console
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
from raise_core.provenance import write_provenance


def test_write_provenance_creates_file(tmp_path: Path):
    path = write_provenance(str(tmp_path), torch_threads=1)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "argv" in data
    assert "git" in data
    assert "versions" in data
    assert data["torch_num_threads"] == 1
    # Second call appends resumes.
    write_provenance(str(tmp_path), torch_threads=1)
    data2 = json.loads(Path(path).read_text(encoding="utf-8"))
    assert len(data2["resumes"]) >= 2


def test_configure_run_logging_writes_file(tmp_path: Path):
    log_path = console.configure_run_logging(output_dir=str(tmp_path))
    assert log_path and Path(log_path).is_file()


def test_pipeline_fast_writes_provenance(tmp_path: Path):
    out = tmp_path / "run"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    RaisePipeline(cfg).run()
    assert (out / "provenance.json").is_file()
