"""Smoke tests for highway VecEnv builder knobs."""

from __future__ import annotations

from types import SimpleNamespace

from domains.highway.adapter import (
    _highway_eval_mode,
    _highway_n_envs,
    _highway_warm_start_enabled,
    _make_train_vec_env,
)
from domains.highway.prompts import D5_SEED_FUNCTION
from raise_core.domains import (
    load_domain,
    make_stage2_trainer_for_domain,
    make_validator_for_domain,
)
from raise_core.explore import RewardCandidate
from raise_core.refine import Stage2Config


def test_n_envs_from_config():
    assert _highway_n_envs(SimpleNamespace(highway_n_envs=4)) == 4
    assert _highway_n_envs(SimpleNamespace(highway_n_envs=0)) == 1


def test_eval_mode_and_warm_start():
    assert _highway_eval_mode(SimpleNamespace(highway_eval_mode="holdout_only")) == (
        "holdout_only"
    )
    assert _highway_eval_mode(SimpleNamespace(highway_eval_mode="nope")) == "both"
    assert _highway_warm_start_enabled(SimpleNamespace(highway_warm_start=False)) is False
    assert _highway_warm_start_enabled(SimpleNamespace(highway_warm_start=True)) is True


def test_warm_start_none_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("RAISE_HIGHWAY_WARM_START", "0")
    assert _highway_warm_start_enabled(SimpleNamespace(highway_warm_start=None)) is False
    monkeypatch.setenv("RAISE_HIGHWAY_WARM_START", "1")
    assert _highway_warm_start_enabled(SimpleNamespace(highway_warm_start=None)) is True


def test_stage2_label_defaults_short_rollout_stochastic_eval():
    cfg = Stage2Config()
    assert cfg.highway_n_steps == 64
    assert cfg.highway_eval_deterministic is False


def test_stochastic_eval_label_reproducible(tmp_path):
    pack = load_domain("highway")
    fn = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    trainer = make_stage2_trainer_for_domain(pack, use_stub=False)
    labels = []
    for run in ("a", "b"):
        cand = RewardCandidate(
            candidate_id="seed", code=D5_SEED_FUNCTION, reward_fn=fn, valid=True
        )
        cfg = Stage2Config(
            train_env_steps=64,
            k2_unit="env_steps",
            eval_episodes=2,
            seed=3,
            device="cpu",
            output_root=str(tmp_path / run),
            highway_warm_start=False,
            highway_eval_mode="holdout_only",
        )
        trainer.train_and_eval(cand, round_index=0, config=cfg)
        md = cand.metadata
        assert md["ppo_n_steps"] == 64
        assert md["eval_deterministic"] is False
        lm = md["last_metrics"]
        labels.append((lm["SR"], lm["CR"], lm["mean_speed"], lm["mean_progress"]))
    assert labels[0] == labels[1]


def test_train_vec_env_wraps_monitor():
    """SB3 needs Monitor so rollout/ep_rew_mean is logged (diagnostics Part 1)."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.highway.env_wrapper import RewardInjectedHighwayEnv
    from domains.highway.prompts import D5_SEED_FUNCTION
    from raise_core.domains import load_domain, make_validator_for_domain

    pack = load_domain("highway")
    fn = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    venv = _make_train_vec_env(fn, seed=0, n_envs=1)
    assert isinstance(venv, DummyVecEnv)
    inner = venv.envs[0]
    assert isinstance(inner, Monitor)
    # No CSV filename → Monitor must not open a results file.
    assert getattr(inner, "results_writer", None) is None
    assert isinstance(inner.env, RewardInjectedHighwayEnv)
    venv.close()
