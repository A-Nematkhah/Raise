"""Phase 2: highway multi-target surrogate + gate quality."""

from __future__ import annotations

import math

import pytest

from raise_core.surrogate.model import SurrogateModel
from raise_core.surrogate.targets import (
    HIGHWAY_TARGET_KEYS,
    normalize_label_row,
    quality_from_y_hat,
    target_keys_for_domain,
)


def test_target_keys_domain_guard():
    assert target_keys_for_domain("crowdnav") == ("SR", "CR", "TR")
    assert target_keys_for_domain("highway") == HIGHWAY_TARGET_KEYS
    assert "mean_speed" in target_keys_for_domain("highway")


def test_quality_prefers_highway_scalar_when_rich():
    crawl = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 8.0,
        "mean_progress": 200.0,
        "soft_success": 0.0,
    }
    cruise = {
        "SR": 1.0,
        "CR": 0.0,
        "TR": 0.0,
        "mean_speed": 24.0,
        "mean_progress": 900.0,
        "soft_success": 0.9,
    }
    assert quality_from_y_hat(cruise) > quality_from_y_hat(crawl)
    # Classic SR/CR/TR only → same quality for both survivors.
    classic_q = quality_from_y_hat({"SR": 1.0, "CR": 0.0, "TR": 0.0})
    assert math.isfinite(classic_q)


def test_normalize_label_aliases():
    row = normalize_label_row(
        {"SR": 0.5, "CR": 0.1, "TR": 0.0, "PL": 400.0, "ITR": 18.0},
        target_keys=HIGHWAY_TARGET_KEYS,
    )
    assert row["mean_progress"] == 400.0
    assert row["mean_speed"] == 18.0
    assert row["soft_success"] == 0.0


def test_highway_multi_target_fit_predict(tmp_path):
    pytest.importorskip("sklearn")
    feats = []
    labs = []
    for i in range(8):
        feats.append(
            {
                "code_hash": f"{i:064d}"[:64],
                "code_len": 100 + i,
                "score1": 0.1 * i,
                "score1_raw": 0.1 * i,
                "score1_holdout": 0.05 * i,
                "score1_train": 0.1 * i,
                "score1_degen": 0.0,
                "score1_rejected": False,
                "score1_worst_mean": 0.2,
                "behavior_fingerprint": [float(i), float(i) * 0.5, 1.0, -1.0],
            }
        )
        labs.append(
            {
                "SR": 0.3 + 0.08 * i,
                "CR": max(0.0, 0.5 - 0.05 * i),
                "TR": 0.05,
                "mean_speed": 8.0 + 2.0 * i,
                "mean_progress": 100.0 + 80.0 * i,
                "soft_success": min(1.0, 0.1 * i),
            }
        )
    model = SurrogateModel(n_bags=3, rf_estimators=8, random_seed=11)
    metrics = model.fit(feats, labs, target_keys=HIGHWAY_TARGET_KEYS)
    assert set(metrics["per_target"]) >= set(HIGHWAY_TARGET_KEYS)
    pred = model.predict(feats[-1])
    assert set(pred.y_hat) >= set(HIGHWAY_TARGET_KEYS)
    assert quality_from_y_hat(pred.y_hat) > quality_from_y_hat(
        {**pred.y_hat, "mean_speed": 5.0, "soft_success": 0.0, "mean_progress": 50.0}
    )
    out = tmp_path / "model"
    model.save(str(out))
    loaded = SurrogateModel.load(str(out))
    assert tuple(loaded.target_keys) == HIGHWAY_TARGET_KEYS


def test_fingerprint_states_include_crawl_and_cruise():
    from domains.highway.state import default_smoke_states, fingerprint_smoke_states

    smoke = default_smoke_states()
    assert len(smoke) >= 8
    speeds = [float(s.speed) for s in smoke]
    assert min(speeds) <= 6.5
    assert max(speeds) >= 28.0
    fp = fingerprint_smoke_states()
    assert len(fp) >= len(smoke)
