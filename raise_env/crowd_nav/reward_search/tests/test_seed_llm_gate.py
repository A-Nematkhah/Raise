"""Fail-closed guards for --llm seed on non-fast entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_EVONAV_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _EVONAV_ROOT / "scripts"


def _run_script(script_name: str, argv: list[str], monkeypatch) -> int:
    """Execute a scripts/*.py main with controlled argv; return exit code."""
    script = _SCRIPTS / script_name
    assert script.is_file(), script
    monkeypatch.chdir(_EVONAV_ROOT)
    # Prepend root so `import crowd_nav` works the same as CLI from evonav_env/.
    root_str = str(_EVONAV_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    monkeypatch.setattr(sys, "argv", [str(script), *argv])
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_run_evonav_refuses_seed_without_fast(monkeypatch, capsys):
    """Non-fast + default seed LLM must exit before GST/dataset/simulator."""
    out_dir = _EVONAV_ROOT / "results" / "_seed_refuse_test"
    code = _run_script(
        "run_evonav.py",
        ["--output-dir", str(out_dir.relative_to(_EVONAV_ROOT))],
        monkeypatch,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Refusing to run a non-fast pipeline" in err
    assert "--allow-seed-llm" in err
    assert not (out_dir / "manifest.json").exists()
    assert not (out_dir / "config.json").exists()


def test_run_evonav_fast_still_allows_seed(monkeypatch):
    code = _run_script(
        "run_evonav.py",
        ["--fast", "--output-dir", "results/_seed_fast_ok", "--device", "cpu"],
        monkeypatch,
    )
    assert code == 0
    assert (_EVONAV_ROOT / "results" / "_seed_fast_ok" / "manifest.json").is_file()


def test_paper_scale_refuses_seed_yaml_default(monkeypatch, capsys):
    code = _run_script(
        "run_evonav_paper_scale.py",
        ["--output-dir", "results/_paper_seed_refuse"],
        monkeypatch,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Refusing to run a non-fast pipeline" in err


def test_paper_scale_dry_run_stubs_bypass_seed_gate(monkeypatch):
    """--dry-run-stubs is the paper-scale equivalent of --fast for the seed gate."""
    code = _run_script(
        "run_evonav_paper_scale.py",
        [
            "--dry-run-stubs",
            "--seeds",
            "425",
            "--output-dir",
            "results/_paper_seed_dry",
            "--device",
            "cpu",
        ],
        monkeypatch,
    )
    assert code == 0
