"""Score1 spawn must not pickle upstream Config (Windows argv crash)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from domains.crowdnav.score1_worker import settings_for_score1_child
from domains.crowdnav.settings import DEFAULT_ENV_CONFIG_PATH, CrowdNavSettings


def test_settings_for_score1_child_strips_env_config_obj():
    fake = SimpleNamespace(sim=SimpleNamespace(circle_radius=1.0))
    settings = CrowdNavSettings(env_config_obj=fake, kinematics="holonomic")
    child = settings_for_score1_child(settings)
    assert child.env_config_obj is None
    assert child.env_config_path == DEFAULT_ENV_CONFIG_PATH
    assert child.kinematics == "holonomic"


def test_settings_for_score1_child_preserves_path():
    fake = SimpleNamespace()
    # Bypass validate_settings (forbids path+obj together) the way a live
    # process may hold an in-memory Config plus an optional path.
    raw = replace(
        CrowdNavSettings(kinematics="unicycle"),
        env_config_obj=fake,
        env_config_path="/tmp/custom_config.py",
    )
    child = settings_for_score1_child(raw)
    assert child.env_config_obj is None
    assert child.env_config_path == "/tmp/custom_config.py"


def test_settings_for_score1_child_noop_without_obj():
    settings = CrowdNavSettings(kinematics="holonomic")
    assert settings_for_score1_child(settings) is settings
