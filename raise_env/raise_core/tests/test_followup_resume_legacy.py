"""Pre-patch checkpoint resume uses legacy values for new keys (no drift-abort)."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import (
    _LEGACY_ABSENT_CONFIG_DEFAULTS,
    _check_config_drift,
    _normalize_resume_config,
)


def test_normalize_resume_fills_legacy_and_warns(monkeypatch):
    warnings: list[str] = []
    from raise_core.raise_loop import runner as runner_mod

    monkeypatch.setattr(
        runner_mod.console, "warn", lambda msg, **_kw: warnings.append(str(msg))
    )
    saved = {"seed": 425, "stage2_train_steps": 8, "population_size": 2}
    out = _normalize_resume_config(saved)
    assert out["stage2_paired_seeds"] is False
    assert out["stage2_seeds"] == 1
    assert out["ppo_normalize_reward"] is False
    assert any("pre-patch checkpoint missing keys" in w for w in warnings)
    for key, legacy in _LEGACY_ABSENT_CONFIG_DEFAULTS.items():
        assert out[key] == legacy


def test_pre_patch_missing_keys_do_not_drift_abort_against_new_defaults(monkeypatch):
    """Simulate runner resume: apply legacy to live cfg, then drift-check."""
    warnings: list[str] = []
    from raise_core.raise_loop import runner as runner_mod

    monkeypatch.setattr(
        runner_mod.console, "warn", lambda msg, **_kw: warnings.append(str(msg))
    )
    raw_saved = {
        "domain": "crowdnav",
        "population_size": 2,
        "generations": 1,
        "stage2_train_steps": 8,
        "seed": 425,
        "eval_episodes": 8,
        "action_mode": "continuous",
        "robot_kinematics": "unicycle",
        "stage2_eval_split": "val",
        "stage2_successive_halving": False,
        "eval_mode": "holdout_only",
        "collapse_cr": 0.8,
        "env_config_path": None,
        # Intentionally omit stage2_paired_seeds / stage2_seeds / ppo_normalize_reward
    }
    cfg = ClosedLoopConfig(
        population_size=2,
        generations=1,
        stage2_train_steps=8,
        seed=425,
        eval_episodes=8,
        stage2_paired_seeds=True,  # post-patch default
        stage2_seeds=1,
        ppo_normalize_reward=False,
    )
    cfg_dict = asdict(cfg)
    saved_cfg = _normalize_resume_config(raw_saved)
    for key, legacy in _LEGACY_ABSENT_CONFIG_DEFAULTS.items():
        if key not in raw_saved:
            setattr(cfg, key, legacy)
            cfg_dict[key] = legacy
    # Must not raise: live cfg now matches legacy fill-ins.
    assert (
        _check_config_drift(
            saved_cfg,
            cfg_dict,
            allow_config_drift=False,
            already_normalized=True,
        )
        is None
    )
    assert cfg.stage2_paired_seeds is False
    assert any("pre-patch checkpoint missing keys" in w for w in warnings)


def test_explicit_new_key_mismatch_still_aborts():
    saved = {
        "stage2_train_steps": 8,
        "seed": 425,
        "population_size": 2,
        "stage2_paired_seeds": False,
        "stage2_seeds": 1,
        "ppo_normalize_reward": False,
        "eval_episodes": 8,
        "action_mode": "continuous",
        "robot_kinematics": "unicycle",
        "stage2_eval_split": "val",
        "stage2_successive_halving": False,
        "collapse_cr": 0.8,
        "env_config_path": None,
    }
    current = {**saved, "stage2_paired_seeds": True}
    with pytest.raises(RuntimeError, match="config drift"):
        _check_config_drift(saved, current, allow_config_drift=False)


def test_legacy_keys_omitted_from_persisted_checkpoint(tmp_path):
    """Pre-patch empty config stays key-absent across resume cycles."""
    from raise_core.explore import RewardCandidate
    from raise_core.raise_loop.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
    from raise_core.raise_loop.config import ClosedLoopConfig
    from raise_core.raise_loop.runner import ClosedLoopRunner
    from domains.crowdnav.reward_checks import make_crowdnav_validator

    out = str(tmp_path / "run")
    code = "def compute_reward(state, memory):\n    return float(state.progress)\n"
    cand = RewardCandidate(
        candidate_id="c0",
        code=code,
        reward_fn=make_crowdnav_validator().validate_code(code),
        valid=True,
        origin="test",
        score=0.5,
        metadata={},
    )
    save_checkpoint(
        out,
        build_checkpoint(
            status="running",
            phase="labeling",
            epoch=0,
            next_epoch=0,
            population=[cand],
            to_label=[cand],
            labeled_ids_this_epoch=[],
            history=[],
            config={},  # pre-patch schema
        ),
    )
    ClosedLoopRunner(
        ClosedLoopConfig(
            population_size=2,
            generations=1,
            stage2_train_steps=8,
            use_stub=True,
            llm_provider="seed",
            output_dir=out,
            resume=True,
            device="cpu",
            stage2_paired_seeds=True,
        )
    ).run()
    saved = load_checkpoint(out)["config"]
    for key in _LEGACY_ABSENT_CONFIG_DEFAULTS:
        assert key not in saved, key
