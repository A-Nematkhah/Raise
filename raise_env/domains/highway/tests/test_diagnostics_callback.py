"""Unit tests for observational highway diagnostics callbacks (no GPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from domains.highway.diagnostics_callback import (
    GroundTruthCheckpointCallback,
    RolloutDiagnosticsCallback,
    _logger_get,
    combine_train_callbacks,
)


def test_logger_get_aliases():
    d = {"rollout/ep_rew_mean": 1.5, "train/approx_kl": 0.01}
    assert _logger_get(d, "ep_rew_mean") == 1.5
    assert _logger_get(d, "approx_kl") == 0.01
    assert _logger_get(d, "clip_fraction") is None
    assert _logger_get(None, "ep_rew_mean") is None


def test_rollout_callback_writes_jsonl(tmp_path):
    path = str(tmp_path / "diagnostics_rollout.jsonl")
    cb = RolloutDiagnosticsCallback(candidate_id="c0", log_path=path)
    cb.model = SimpleNamespace(
        logger=SimpleNamespace(
            name_to_value={
                "rollout/ep_rew_mean": 2.0,
                "train/explained_variance": 0.5,
                "time/fps": 100.0,
            }
        )
    )
    cb.num_timesteps = 256
    cb._on_rollout_end()
    assert cb._on_step() is True
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "c0"
    assert rows[0]["steps"] == 256
    assert rows[0]["ep_rew_mean"] == 2.0
    assert rows[0]["explained_variance"] == 0.5
    assert rows[0]["approx_kl"] is None
    assert rows[0]["fps"] == 100.0


def test_rollout_callback_swallows_io_errors(tmp_path, monkeypatch):
    path = str(tmp_path / "diagnostics_rollout.jsonl")
    cb = RolloutDiagnosticsCallback(candidate_id="c0", log_path=path)
    cb.model = SimpleNamespace(logger=SimpleNamespace(name_to_value={}))
    cb.num_timesteps = 1

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "domains.highway.diagnostics_callback._append_jsonl", _boom
    )
    cb._on_rollout_end()
    assert cb._on_step() is True


def test_groundtruth_thresholds():
    cb = GroundTruthCheckpointCallback(
        candidate_id="c0",
        log_path="/tmp/x.jsonl",
        train_env_steps=10_000,
        reward_fn=lambda *_a, **_k: 0.0,
        seed=0,
    )
    assert cb._thresholds == [2000, 4000, 6000, 8000, 10000]


def test_combine_callbacks():
    a = RolloutDiagnosticsCallback(
        candidate_id="a", log_path="/tmp/a.jsonl"
    )
    b = RolloutDiagnosticsCallback(
        candidate_id="b", log_path="/tmp/b.jsonl"
    )
    combo = combine_train_callbacks([a, b])
    from stable_baselines3.common.callbacks import CallbackList

    assert isinstance(combo, CallbackList)
