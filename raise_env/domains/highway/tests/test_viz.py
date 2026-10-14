"""Highway viz: checkpoint lookup and holdout-traffic rollouts (fake env, no PPO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from domains.highway import env_wrapper
from domains.highway import viz


class _FakeEnv:
    created = []

    def __init__(self, reward_fn, *, seed=None, config=None, render_mode=None):
        self.seed = seed
        self.config = config
        self.reset_seed = None
        self.steps = 0
        _FakeEnv.created.append(self)

    def reset(self, seed=None):
        self.reset_seed = seed
        return np.zeros(3), {}

    def render(self):
        return np.zeros((4, 6, 3), dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        info = {"raise_progress": 10.0, "raise_speed": 25.0}
        if self.steps == 3:
            info["raise_collision"] = True
        return np.zeros(3), 0.0, False, False, info

    def close(self):
        pass


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 1, None


def test_rollout_uses_holdout_traffic_by_default(monkeypatch):
    _FakeEnv.created.clear()
    monkeypatch.setattr(env_wrapper, "RewardInjectedHighwayEnv", _FakeEnv)
    frames, summary = viz.record_policy_episode(
        reward_fn=None, model=_FakeModel(), seed=20_000_425
    )
    env = _FakeEnv.created[-1]
    assert env.config == env_wrapper.holdout_env_config()
    assert env.reset_seed == 20_000_425
    assert summary["traffic"] == "holdout"
    assert summary["outcome"] == "collision"
    assert summary["progress_m"] == 30.0
    assert len(frames) == 4  # three steps + final frame


def test_rollout_can_use_train_traffic(monkeypatch):
    _FakeEnv.created.clear()
    monkeypatch.setattr(env_wrapper, "RewardInjectedHighwayEnv", _FakeEnv)
    _, summary = viz.record_policy_episode(
        reward_fn=None, model=_FakeModel(), seed=1, traffic="train"
    )
    assert _FakeEnv.created[-1].config == env_wrapper.training_env_config()
    assert summary["traffic"] == "train"


def test_resolve_checkpoint_paths(tmp_path: Path):
    model = tmp_path / "stage3_train" / "r00_cro_0014" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"zip")
    run = str(tmp_path)

    absolute = {"metadata": {"checkpoint_path": str(model)}}
    assert viz._resolve_checkpoint(run, absolute) == str(model.resolve())
    relative = {"metadata": {"checkpoint_path": "stage3_train/r00_cro_0014/model"}}
    assert viz._resolve_checkpoint(run, relative) == str(model.resolve())
    moved = {"candidate_id": "cro_0014", "metadata": {"checkpoint_path": "/gone/model.zip"}}
    assert Path(viz._resolve_checkpoint(run, moved)) == model
    assert viz._resolve_checkpoint(run, {"candidate_id": "nope"}) is None


def test_visualize_reports_missing_inputs(tmp_path: Path):
    assert viz.visualize_best_policy(str(tmp_path))["reason"] == "no best_stage*.json"
    (tmp_path / "best_stage3.json").write_text(
        json.dumps({"candidate_id": "x", "code": "", "metadata": {}}), encoding="utf-8"
    )
    out = viz.visualize_best_policy(str(tmp_path))
    assert out["ok"] is False and out["reason"] == "no model.zip"
