"""Diversity reorder demotes identical Stage-II metric fingerprints."""

from __future__ import annotations

from types import SimpleNamespace

from raise_core.raise_loop.diversity import diversify_ranking, metrics_fingerprint


def _cand(cid: str, *, sr: float, speed: float, soft: float = 1.0, lc: float = 0.0):
    return SimpleNamespace(
        candidate_id=cid,
        metadata={
            "last_metrics": {
                "SR": sr,
                "CR": 0.0,
                "soft_success": soft,
                "mean_speed": speed,
                "lane_change_rate": lc,
                "PL": 800.0,
            }
        },
    )


def test_fingerprint_stable_for_same_metrics():
    a = _cand("a", sr=1.0, speed=22.0)
    b = _cand("b", sr=1.0, speed=22.0)
    assert metrics_fingerprint(a) == metrics_fingerprint(b)
    c = _cand("c", sr=0.5, speed=22.0)
    assert metrics_fingerprint(a) != metrics_fingerprint(c)


def test_diversify_keeps_first_demotes_clones():
    ranked = [
        _cand("elite", sr=1.0, speed=22.0),
        _cand("clone1", sr=1.0, speed=22.0),
        _cand("other", sr=0.6, speed=18.0, soft=0.2, lc=0.1),
        _cand("clone2", sr=1.0, speed=22.0),
    ]
    out = diversify_ranking(ranked)
    ids = [c.candidate_id for c in out]
    assert ids[0] == "elite"
    assert ids[1] == "other"
    assert ids[2:] == ["clone1", "clone2"]


def test_unlabeled_stay_unique():
    unlabeled = SimpleNamespace(candidate_id="u", metadata={})
    ranked = [unlabeled, _cand("a", sr=1.0, speed=22.0), unlabeled]
    out = diversify_ranking(ranked)
    assert [c.candidate_id for c in out] == ["u", "a", "u"]
