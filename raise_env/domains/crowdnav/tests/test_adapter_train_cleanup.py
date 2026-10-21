"""P0-6/P0-7: train env closed on exceptions; checkpoint paths stored as POSIX."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from domains.crowdnav import adapter as adapter_mod
from domains.crowdnav.adapter import CrowdNavPPOTrainer, _normalize_stored_path
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate


class _CloseTrackingVec:
    def __init__(self):
        self.closed = False
        self.observation_space = SimpleNamespace()
        self.action_space = SimpleNamespace()
        self.num_envs = 1

    def close(self):
        self.closed = True

    def reset(self):
        return None

    def step(self, *_a, **_k):
        raise RuntimeError("unused")


def test_normalize_stored_path_uses_posix():
    raw = "results\\crowdnav_run\\closed_loop\\stage2_train\\r00_x\\model.zip"
    assert "\\" not in _normalize_stored_path(raw)
    assert _normalize_stored_path(raw).endswith("model.zip")


def test_train_env_closed_when_learn_raises(monkeypatch, tmp_path):
    env = _CloseTrackingVec()
    monkeypatch.setattr(adapter_mod, "_make_train_vec_env", lambda *a, **k: env)

    class _BoomPPO:
        def __init__(self, *a, **k):
            pass

        def learn(self, *a, **k):
            raise RuntimeError("boom during learn")

        def save(self, *a, **k):
            raise AssertionError("save should not run")

    monkeypatch.setattr(adapter_mod, "PPO", _BoomPPO)
    fn = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    cand = RewardCandidate(
        candidate_id="c0", code=D5_SEED_FUNCTION, reward_fn=fn, valid=True
    )
    cfg = SimpleNamespace(
        train_env_steps=8,
        eval_episodes=1,
        seed=1,
        output_root=str(tmp_path / "train"),
        n_envs=1,
        warm_start=False,
        ppo_n_steps=16,
        ppo_n_epochs=1,
        ppo_ent_coef=0.0,
        record_training_dynamics=False,
        device="cpu",
        eval_mode="holdout_only",
        eval_split="val",
        eval_deterministic=True,
        action_mode="discrete",
        robot_kinematics="holonomic",
        action_speed_samples=5,
        action_rotation_samples=16,
        env_config_path=None,
    )
    trainer = CrowdNavPPOTrainer(stage="stage2")
    with pytest.raises(RuntimeError, match="boom"):
        trainer._train_and_eval(cand, round_index=0, config=cfg)
    assert env.closed is True


def test_checkpoint_path_stored_as_posix(monkeypatch, tmp_path):
    env = _CloseTrackingVec()
    monkeypatch.setattr(adapter_mod, "_make_train_vec_env", lambda *a, **k: env)

    class _OkPPO:
        def __init__(self, *a, **k):
            self.num_timesteps = 8
            self.policy = SimpleNamespace(state_dict=lambda: {})

        def learn(self, *a, **k):
            return self

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"zip")

        def set_random_seed(self, *_a, **_k):
            return None

        def predict(self, obs, deterministic=True):
            return 0, None

    monkeypatch.setattr(adapter_mod, "PPO", _OkPPO)

    holdout = SimpleNamespace(sr=0.0, cr=0.0, tr=1.0, nt=1.0, pl=0.0, itr=0.0, sd=0.0)

    def _fake_evaluate(self, model, candidate, *, config, seed, settings, stage_tag):
        return holdout, {"fitness": 0.0, "holdout": {"fitness": 0.0, "SR": 0.0}}

    monkeypatch.setattr(CrowdNavPPOTrainer, "_evaluate", _fake_evaluate)
    fn = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    cand = RewardCandidate(
        candidate_id="c0", code=D5_SEED_FUNCTION, reward_fn=fn, valid=True
    )
    out_root = tmp_path / "stage2_train"
    cfg = SimpleNamespace(
        train_env_steps=8,
        eval_episodes=1,
        seed=1,
        output_root=str(out_root),
        n_envs=1,
        warm_start=False,
        ppo_n_steps=16,
        ppo_n_epochs=1,
        ppo_ent_coef=0.0,
        record_training_dynamics=False,
        device="cpu",
        eval_mode="holdout_only",
        eval_split="val",
        eval_deterministic=True,
        action_mode="discrete",
        robot_kinematics="holonomic",
        action_speed_samples=5,
        action_rotation_samples=16,
        env_config_path=None,
    )
    CrowdNavPPOTrainer(stage="stage2")._train_and_eval(cand, round_index=0, config=cfg)
    ckpt = cand.metadata["checkpoint_path"]
    assert "\\" not in ckpt
    assert ckpt.endswith("model.zip")
    assert os.path.isfile(ckpt) or os.path.isfile(ckpt.replace("/", os.sep))
    assert env.closed is True
