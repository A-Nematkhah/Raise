"""Fail-closed guards for --llm seed on non-fast entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_RAISE_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _RAISE_ROOT / "scripts"


def _run_script(script_name: str, argv: list[str], monkeypatch) -> int:
    """Execute a scripts/*.py main with controlled argv; return exit code."""
    script = _SCRIPTS / script_name
    assert script.is_file(), script
    monkeypatch.chdir(_RAISE_ROOT)
    # Prepend root so imports work the same as CLI from raise_env/.
    root_str = str(_RAISE_ROOT)
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


def test_run_raise_refuses_seed_without_fast(monkeypatch, capsys, tmp_path):
    """Non-fast + default seed LLM must exit before dataset/PPO work."""
    out_dir = tmp_path / "seed_refuse"
    code = _run_script(
        "run_raise.py",
        ["--output-dir", str(out_dir)],
        monkeypatch,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Refusing to run a non-fast pipeline" in err
    assert "--allow-seed-llm" in err
    assert not (out_dir / "manifest.json").exists()
    assert not (out_dir / "config.json").exists()


def test_run_raise_fast_still_allows_seed(monkeypatch, tmp_path):
    out_dir = tmp_path / "seed_fast_ok"
    code = _run_script(
        "run_raise.py",
        ["--fast", "--output-dir", str(out_dir), "--device", "cpu"],
        monkeypatch,
    )
    assert code == 0
    assert (out_dir / "manifest.json").is_file()
