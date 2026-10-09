"""Skip highway tests cleanly when the env / PPO dependencies are not installed."""

from __future__ import annotations

import importlib.util

_HIGHWAY_DEPS = ("gymnasium", "highway_env", "stable_baselines3")
_MISSING = [name for name in _HIGHWAY_DEPS if importlib.util.find_spec(name) is None]

collect_ignore_glob = ["test_*.py"] if _MISSING else []
