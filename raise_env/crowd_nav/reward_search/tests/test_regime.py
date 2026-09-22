"""Tests for EVOLUTION_RANDOMIZATION_REGIME + GST model_dir assertions."""

from __future__ import annotations

import logging

import pytest

from crowd_nav.reward_search.regime import (
    EVOLUTION_RANDOMIZATION_REGIME,
    GST_MODEL_DIR_WITH_RANDOM,
    GST_MODEL_DIR_WITHOUT_RANDOM,
    assert_gst_matches_regime,
    gst_model_dir_for_regime,
    gst_path_looks_randomized,
    parse_regime,
    randomization_flags,
)


def test_default_regime_is_without_random():
    assert EVOLUTION_RANDOMIZATION_REGIME == "without_random"
    assert parse_regime(None) == "without_random"


def test_randomization_flags_without_random():
    attrs, goals = randomization_flags("without_random")
    assert attrs is False
    assert goals is False


def test_randomization_flags_with_random():
    attrs, goals = randomization_flags("with_random")
    assert attrs is True
    assert goals is True


def test_both_regime_rejected_for_single_pass():
    with pytest.raises(ValueError, match="two separate"):
        randomization_flags("both")
    with pytest.raises(ValueError, match="both"):
        gst_model_dir_for_regime("both")


def test_gst_dirs_match_regime_suffix():
    assert not gst_path_looks_randomized(GST_MODEL_DIR_WITHOUT_RANDOM)
    assert gst_path_looks_randomized(GST_MODEL_DIR_WITH_RANDOM)
    assert gst_model_dir_for_regime("without_random") == GST_MODEL_DIR_WITHOUT_RANDOM
    assert gst_model_dir_for_regime("with_random") == GST_MODEL_DIR_WITH_RANDOM


def test_assert_gst_passes_for_matching_without_random():
    assert_gst_matches_regime(
        GST_MODEL_DIR_WITHOUT_RANDOM,
        "without_random",
        predict_method="inferred",
        entry_point="test",
    )


def test_assert_gst_fails_on_rand_mismatch(caplog):
    with pytest.raises(AssertionError, match="does not match"):
        assert_gst_matches_regime(
            GST_MODEL_DIR_WITH_RANDOM,
            "without_random",
            predict_method="inferred",
            entry_point="test",
        )


def test_assert_gst_skipped_for_predict_none(caplog):
    with caplog.at_level(logging.INFO):
        assert_gst_matches_regime(
            GST_MODEL_DIR_WITH_RANDOM,
            "without_random",
            predict_method="none",
            entry_point="collect_stage1_dataset",
        )
    assert any("N/A" in r.message for r in caplog.records)


def test_stage2_defaults_use_inferred_and_without_random():
    from crowd_nav.reward_search.stage2 import Stage2Config

    cfg = Stage2Config()
    assert cfg.randomization_regime == "without_random"
    assert cfg.predict_method == "inferred"
    assert cfg.env_name == "CrowdSimPredRealGST-v0"


def test_stage3_defaults_use_inferred_and_without_random():
    from crowd_nav.reward_search.stage3 import Stage3Config

    cfg = Stage3Config()
    assert cfg.randomization_regime == "without_random"
    assert cfg.predict_method == "inferred"
    assert cfg.env_name == "CrowdSimPredRealGST-v0"
