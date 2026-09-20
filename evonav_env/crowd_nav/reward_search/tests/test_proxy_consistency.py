"""Proxy consistency (§4.3.4) unit tests."""

from __future__ import annotations

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.proxy_consistency import (
    compute_proxy_consistency,
    genome_key,
    scores_from_stage1,
    scores_from_trained,
    spearman_rank_correlation,
    top_k_preservation,
)


def _c(code: str, *, score=None, sr=None) -> RewardCandidate:
    md = {}
    if sr is not None:
        md["last_metrics"] = {
            "SR": float(sr),
            "CR": 0.0,
            "TR": 0.0,
            "NT": 1.0,
            "PL": 1.0,
            "ITR": 0.0,
            "SD": 0.5,
        }
    return RewardCandidate(
        candidate_id=f"id_{genome_key(code)[:6]}",
        code=code,
        score=score,
        metadata=md,
    )


def test_genome_key_stable():
    assert genome_key("  abc\n") == genome_key("abc")


def test_perfect_spearman_when_order_preserved():
    codes = [
        "def compute_reward(state, memory):\n    return 1.0\n",
        "def compute_reward(state, memory):\n    return 2.0\n",
        "def compute_reward(state, memory):\n    return 3.0\n",
    ]
    s1 = scores_from_stage1(
        [_c(codes[0], score=0.1), _c(codes[1], score=0.5), _c(codes[2], score=0.9)]
    )
    s2 = scores_from_trained(
        [_c(codes[0], sr=0.2), _c(codes[1], sr=0.5), _c(codes[2], sr=0.8)]
    )
    result = spearman_rank_correlation(s1, s2)
    assert result["usable"] is True
    assert result["n_overlap"] == 3
    assert abs(float(result["rho"]) - 1.0) < 1e-6


def test_top_k_preservation():
    codes = [f"def compute_reward(state, memory):\n    return {i}.0\n" for i in range(5)]
    s1 = {genome_key(c): float(i) for i, c in enumerate(codes)}
    # Stage II reverses order → top-3 of A and B share only the middle genome.
    s2 = {genome_key(c): float(4 - i) for i, c in enumerate(codes)}
    out = top_k_preservation(s1, s2, k=3)
    assert out["kept"] == 1
    assert abs(float(out["preservation"]) - 1.0 / 3.0) < 1e-9
    # Same order → full preservation
    out2 = top_k_preservation(s1, s1, k=3)
    assert out2["preservation"] == 1.0


def test_compute_proxy_consistency_report_shape():
    codes = [
        "def compute_reward(state, memory):\n    return 0.0\n",
        "def compute_reward(state, memory):\n    return 1.0\n",
        "def compute_reward(state, memory):\n    return 2.0\n",
    ]
    s1 = [_c(codes[i], score=0.1 * (i + 1)) for i in range(3)]
    s2 = [_c(codes[i], sr=0.2 * (i + 1)) for i in range(3)]
    s3 = [_c(codes[i], sr=0.15 * (i + 1)) for i in range(3)]
    report = compute_proxy_consistency(stage1=s1, stage2=s2, stage3=s3, top_k=2)
    assert "stage1_vs_stage2" in report
    assert "stage2_vs_stage3" in report
    assert "stage2_vs_stage3_lineage" in report
    assert report["stage1_vs_stage2"]["spearman"]["usable"] is True


def test_lineage_consistency_when_refine_changes_code():
    """Exact fingerprint overlap can be empty after refine; lineage should work."""
    parent_codes = [
        "def compute_reward(state, memory):\n    return 0.0\n",
        "def compute_reward(state, memory):\n    return 1.0\n",
        "def compute_reward(state, memory):\n    return 2.0\n",
    ]
    child_codes = [
        "def compute_reward(state, memory):\n    return 10.0\n",
        "def compute_reward(state, memory):\n    return 11.0\n",
        "def compute_reward(state, memory):\n    return 12.0\n",
    ]
    s2 = [_c(parent_codes[i], sr=0.2 * (i + 1)) for i in range(3)]
    s3 = []
    for i in range(3):
        c = _c(child_codes[i], sr=0.3 * (i + 1))
        c.metadata["parent_genome_key"] = genome_key(parent_codes[i])
        s3.append(c)
    report = compute_proxy_consistency(stage1=s2, stage2=s2, stage3=s3, top_k=2)
    exact = report["stage2_vs_stage3"]["spearman"]
    lineage = report["stage2_vs_stage3_lineage"]["spearman"]
    assert exact["usable"] is False or exact["n_overlap"] == 0
    assert lineage["usable"] is True
    assert report["stage2_vs_stage3_preferred"] == "lineage"
