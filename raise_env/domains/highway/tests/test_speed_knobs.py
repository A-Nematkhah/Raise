"""Smoke tests for highway VecEnv builder knobs."""

from __future__ import annotations

from types import SimpleNamespace

from domains.highway.adapter import (
    _highway_eval_mode,
    _highway_n_envs,
    _highway_warm_start_enabled,
    _make_train_vec_env,
)


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
