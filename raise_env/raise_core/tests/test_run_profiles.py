"""``run_raise.py --profile`` expansion (replaces the former run_raise_* wrappers)."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest

from raise_core.presets import profile_to_run_raise_argv

_RAISE_ROOT = Path(__file__).resolve().parents[2]
_RUN_RAISE = _RAISE_ROOT / "scripts" / "run_raise.py"


def _print_config(argv: list, monkeypatch, capsys) -> dict:
    monkeypatch.chdir(_RAISE_ROOT)
    monkeypatch.setattr(sys, "argv", [str(_RUN_RAISE), *argv, "--print-config"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(_RUN_RAISE), run_name="__main__")
    assert exc.value.code == 0
    rows = {}
    for line in capsys.readouterr().out.splitlines():
        key, _, val = line.partition("=")
        rows[key] = val.strip("'\"")
    return rows


def test_closed_loop_profile_stamps_output_and_nests_surrogate(monkeypatch, capsys):
    cfg = _print_config(["--profile", "12h"], monkeypatch, capsys)
    out = cfg["output_dir"]
    assert out.startswith("results/raise_12h_")
    assert os.path.normpath(cfg["surrogate"]) == os.path.normpath(
        os.path.join(out, "surrogate_model")
    )
    assert os.path.normpath(cfg["surrogate_dataset"]) == os.path.normpath(
        os.path.join(out, "surrogate_dataset")
    )
    assert cfg["closed_loop"] == "True"
    assert cfg["closed_loop_proxy_feedback"] == "True"
    assert cfg["no_h_sweep"] == "True"


def test_profile_resume_uses_given_output_dir(monkeypatch, capsys):
    out = "results/raise_1h_20260101_000000"
    cfg = _print_config(
        ["--profile", "1h", "--output-dir", out, "--llm", "seed"], monkeypatch, capsys
    )
    assert cfg["output_dir"] == out
    assert cfg["llm"] == "seed"
    assert os.path.normpath(cfg["surrogate"]) == os.path.normpath(
        os.path.join(out, "surrogate_model")
    )


def test_18h_profile_matches_former_wrapper():
    argv = profile_to_run_raise_argv("18h")
    assert argv[argv.index("--num-processes") + 1] == "1"
    assert argv[argv.index("--final-rank") + 1] == "llm"
    assert "--no-h-sweep" in argv


def test_smoke_profile_is_tiny_real_trainer_run():
    argv = profile_to_run_raise_argv("smoke")
    assert "--fast" not in argv and "--stage3-stub" not in argv
    assert "--allow-seed-llm" in argv
    assert argv[argv.index("--stage2-train-steps") + 1] == "200"
    assert argv[argv.index("--stage3-train-steps") + 1] == "200"


def test_paper_scale_profile_not_flattened():
    with pytest.raises(ValueError):
        profile_to_run_raise_argv("paper_scale")
