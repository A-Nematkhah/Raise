"""Strict-JSON sanitization for Score1 hard-rejects (P0-3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate
from raise_core.json_io import atomic_write_json, sanitize_nonfinite
from raise_core.population_io import candidate_to_dict, load_candidate_dict
from raise_core.raise_loop.evolve_rank import _score1


def test_sanitize_nonfinite_replaces_nan_and_inf():
    payload = {
        "score": float("-inf"),
        "nested": {"a": float("nan"), "b": float("inf"), "ok": 1.5},
        "list": [0.0, float("-inf")],
    }
    clean = sanitize_nonfinite(payload)
    assert clean["score"] is None
    assert clean["nested"]["a"] is None
    assert clean["nested"]["b"] is None
    assert clean["nested"]["ok"] == 1.5
    assert clean["list"] == [0.0, None]


def test_atomic_write_roundtrip_rejected_score(tmp_path: Path):
    path = tmp_path / "cand.json"
    validator = make_crowdnav_validator()
    code = "def compute_reward(state, memory):\n    return 0.0, {}\n"
    reward_fn, err = validator.try_validate(code)
    assert reward_fn is not None, err
    cand = RewardCandidate(
        candidate_id="mut_rejected",
        code=code,
        reward_fn=reward_fn,
        score=float("-inf"),
        valid=True,
        metadata={"score1_rejected": True, "score1_reject_reason": "score1_error"},
    )
    atomic_write_json(path, candidate_to_dict(cand))
    text = path.read_text(encoding="utf-8")
    # Strict parsers must accept the file (no Infinity / NaN tokens).
    loaded = json.loads(text, parse_constant=lambda _c: (_ for _ in ()).throw(ValueError(_c)))
    assert loaded["score"] is None
    restored = load_candidate_dict(loaded, validator=validator)
    assert restored.score == float("-inf")
    assert math.isinf(restored.score) and restored.score < 0
    assert _score1(restored) == float("-inf")


def test_legacy_infinity_file_still_loads(tmp_path: Path):
    path = tmp_path / "legacy.json"
    # Non-standard JSON that Python's default loader accepts.
    path.write_text(
        '{"candidate_id": "mut_0013", "code": "def compute_reward(state, memory):\\n'
        '    return 0.0, {}\\n", "score": -Infinity, "valid": true, '
        '"origin": "mutation", "parent_ids": [], "validation_error": null, '
        '"metadata": {"score1_rejected": true}}\n',
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["score"] == float("-inf")
    restored = load_candidate_dict(raw, validator=make_crowdnav_validator())
    assert restored.score == float("-inf")
    assert _score1(restored) == float("-inf")
