"""Console progress must stay visible when stderr is not a TTY."""

from __future__ import annotations

from raise_core import console


def test_plain_progress_used_when_non_interactive(monkeypatch, capsys):
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RAISE_FORCE_TQDM", raising=False)
    console.configure_run_logging()
    bar = console.progress(total=100, desc="PPO cand")
    assert isinstance(bar, console._PlainProgress)
    # Large SB3-style deltas must still emit bucket logs.
    bar.update(40)
    bar.update(40)
    bar.update(20)
    bar.close()
    err = capsys.readouterr().err
    assert "PPO cand (0/100)" in err
    assert "40/100" in err or "80/100" in err
    assert "100/100" in err


def test_tqdm_aware_handler_writes_status(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    console.configure_run_logging()
    console.status("stage visible", stage="Stage II")
    err = capsys.readouterr().err
    assert "stage visible" in err
    assert "[Stage II]" in err
