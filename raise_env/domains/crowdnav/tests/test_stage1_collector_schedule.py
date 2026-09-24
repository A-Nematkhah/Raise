"""Stage I collector schedule diversity (no env required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_collect_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "collect_stage1_dataset.py"
    spec = importlib.util.spec_from_file_location("collect_stage1_dataset", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_schedule_has_unique_non_random_slots():
    mod = _load_collect_module()
    schedule = mod.build_stage1_traj_schedule(10, (0.25, 0.55))
    assert len(schedule) == 10
    non_random = [(p, n) for p, n, rnd in schedule if not rnd]
    # Clean ORCA/SF + distinct noise levels must not collapse to duplicates.
    assert len(non_random) == len(set(non_random))
    assert sum(1 for *_, rnd in schedule if rnd) == 2
    assert ("orca", 0.0) in non_random
    assert ("social_force", 0.0) in non_random


def test_traj_seeds_are_unique_per_slot():
    """Per-trajectory seeds must differ so post-reset RNG can diversify randoms."""
    base = 425
    seeds = {base + j * 10007 + t * 97 for j in range(3) for t in range(10)}
    assert len(seeds) == 30


def test_truncated_schedule_covers_clean_and_random():
    mod = _load_collect_module()
    schedule = mod.build_stage1_traj_schedule(4, (0.25, 0.55))
    policies = {p for p, _n, _r in schedule}
    assert "orca" in policies
    assert "social_force" in policies
    assert any(rnd for *_p, rnd in schedule)
