"""New comparability-affecting fields must be in resume drift keys (P3)."""

from __future__ import annotations

from raise_core.pipeline import RaiseRunConfig
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import _RESUME_CONFIG_KEYS
from raise_core.validate import Stage3Config


def test_stage2_eval_split_in_resume_keys():
    assert "stage2_eval_split" in _RESUME_CONFIG_KEYS
    assert ClosedLoopConfig().stage2_eval_split == "val"
    assert RaiseRunConfig().stage2_eval_split == "val"


def test_stage3_collapse_defaults_in_snapshot():
    cfg = Stage3Config()
    assert cfg.skip_collapsed is True
    assert cfg.collapse_cr == 0.8
    # Snapshot used for resume drift warnings.
    from raise_core.validate import Stage3Runner, StubPolicyTrainer
    from domains.crowdnav.reward_checks import make_crowdnav_validator

    snap = Stage3Runner(
        StubPolicyTrainer(),
        validator=make_crowdnav_validator(),
        config=cfg,
    )._config_snapshot()
    assert snap["skip_collapsed"] is True
    assert snap["collapse_cr"] == 0.8
