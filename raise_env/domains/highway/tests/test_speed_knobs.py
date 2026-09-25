"""Smoke tests for highway VecEnv builder knobs."""

from __future__ import annotations

from types import SimpleNamespace

from domains.highway.adapter import (
    _highway_eval_mode,
    _highway_n_envs,
    _highway_warm_start_enabled,
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
