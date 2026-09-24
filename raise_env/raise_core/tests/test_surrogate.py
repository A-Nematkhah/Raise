"""Surrogate v1 unit tests (PLAN.md §9)."""

from __future__ import annotations

import json
import math
import os

import pytest

from raise_core.explore import RewardCandidate
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from raise_core.sandbox.validator import RewardValidator
from raise_core.scoring import Score1Result
from raise_core.surrogate.dataset_io import (
    append_example,
    load_table,
    read_manifest,
    write_manifest,
)
from raise_core.surrogate.features import extract_candidate_features
from raise_core.surrogate.model import SurrogateModel


def _valid_candidate(candidate_id: str = "c0") -> RewardCandidate:
    reward_fn = RewardValidator().validate_code(D5_SEED_FUNCTION)
    return RewardCandidate(
        candidate_id=candidate_id,
        code=D5_SEED_FUNCTION,
        reward_fn=reward_fn,
        valid=True,
        origin="seed",
    )


def test_feature_schema_serializable():
    cand = _valid_candidate()
    score = Score1Result(
        score=0.5,
        degenerate_fraction=0.0,
        raw_score=0.55,
        train_score=0.6,
        holdout_score=0.4,
        scenario_scores={"s0": 0.1, "s1": 0.9, "s2": 0.2},
    )
    feats = extract_candidate_features(cand, score1_result=score)
    roundtrip = json.loads(json.dumps(feats))
    assert roundtrip["schema_version"] == "1"
    assert roundtrip["code_hash"]
    assert isinstance(roundtrip["behavior_fingerprint"], list)
    assert len(roundtrip["behavior_fingerprint"]) >= 1
    assert roundtrip["score1_worst_k"] is not None
    assert len(roundtrip["score1_worst_k"]) == 3


def test_dataset_io_roundtrip(tmp_path):
    root = str(tmp_path / "surrogate_dataset")
    append_example(
        root,
        features={"schema_version": "1", "code_hash": "a" * 64, "code_len": 10},
        labels={"SR": 0.5, "CR": 0.1, "TR": 0.2, "scalar": 0.3},
        example_id="aaaaaaaaaaaa_stage2_short",
    )
    append_example(
        root,
        features={"schema_version": "1", "code_hash": "b" * 64, "code_len": 12},
        labels={"SR": 0.7, "CR": 0.0, "TR": 0.1, "scalar": 0.65},
        example_id="bbbbbbbbbbbb_stage2_short",
    )
    feats, labs = load_table(root)
    assert len(feats) == 2
    assert len(labs) == 2
    assert feats[0]["example_id"] == labs[0]["example_id"]
    write_manifest(root, {"n": 2, "feature_schema_version": "1"})
    assert read_manifest(root)["n"] == 2


def test_predict_shape(tmp_path):
    pytest.importorskip("sklearn")
    feats = []
    labs = []
    for i in range(6):
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
                "behavior_fingerprint": [float(i), float(i) * 0.5, 1.0, -1.0, 0.0, 0.1, 0.2, 0.3],
            }
        )
        labs.append(
            {
                "SR": 0.2 + 0.1 * i,
                "CR": max(0.0, 0.4 - 0.05 * i),
                "TR": 0.1,
                "scalar": 0.0,
            }
        )
    model = SurrogateModel(n_bags=3, rf_estimators=8, random_seed=7)
    metrics = model.fit(feats, labs)
    assert "per_target" in metrics
    assert set(metrics["per_target"]) >= {"SR", "CR", "TR"}

    out = tmp_path / "model"
    model.save(str(out))
    loaded = SurrogateModel.load(str(out))
    pred = loaded.predict(feats[0])
    assert set(pred.y_hat) >= {"SR", "CR", "TR"}
    assert all(math.isfinite(v) for v in pred.y_hat.values())
    assert pred.uncertainty >= 0.0


def test_bootstrap_fast_stub(tmp_path):
    pytest.importorskip("sklearn")
    from raise_core.surrogate.bootstrap import run_bootstrap

    out_dir = str(tmp_path / "data")
    model_dir = str(tmp_path / "artifacts")
    result = run_bootstrap(
        out_dir=out_dir,
        model_dir=model_dir,
        n_candidates=4,
        stage2_train_steps=32,
        use_stub=True,
        force=True,
        seed=425,
        llm_provider="seed",
    )
    assert result["status"] == "ok"
    assert result["n"] >= 1
    assert os.path.isfile(os.path.join(model_dir, "model.joblib"))
    assert os.path.isfile(os.path.join(model_dir, "metrics.json"))
    feats, labs = load_table(out_dir)
    assert len(feats) == len(labs) == result["n"]
    loaded = SurrogateModel.load(model_dir)
    pred = loaded.predict(feats[0])
    assert math.isfinite(pred.y_hat["SR"])
    assert pred.uncertainty >= 0.0


def test_bootstrap_population_uses_stage1_gen0(monkeypatch):
    """Bootstrap must call StageIEvolver.initialize_population (same as main run)."""
    from raise_core.explore import StageIEvolver
    from raise_core.llm import SeedVariantLLMClient
    from domains.crowdnav.prompts import D1_SYSTEM_PROMPT, D5_SEED_FUNCTION
    from raise_core.sandbox.validator import RewardValidator
    from raise_core.surrogate import bootstrap as boot_mod

    seen: list[str] = []
    called = {"init": 0}

    class RecordingSeed(SeedVariantLLMClient):
        def complete(self, prompt: str, *, max_tokens=None) -> str:
            seen.append(prompt)
            return super().complete(prompt, max_tokens=max_tokens)

    real_init = StageIEvolver.initialize_population

    def _wrap(self):
        called["init"] += 1
        return real_init(self)

    monkeypatch.setattr(StageIEvolver, "initialize_population", _wrap)
    monkeypatch.setattr(
        "raise_core.llm.make_llm_client",
        lambda *a, **k: RecordingSeed(),
    )

    population = boot_mod._build_population(
        n_candidates=3,
        llm_provider="seed",
        seed=7,
    )
    assert called["init"] == 1
    assert len(population) >= 1
    assert all(c.valid for c in population)
    assert seen, "expected Gen0 LLM prompts"
    joined = "\n".join(seen)
    assert "generate a reward variant" not in joined
    assert D1_SYSTEM_PROMPT[:40] in joined
    assert "Seed function" in joined
    # Sanity: seed-only path still validates like Stage I
    RewardValidator().validate_code(D5_SEED_FUNCTION)
