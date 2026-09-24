"""
Proxy consistency metrics (RAISE §4.3.4 / Appendix C.8).

Compare stage-wise rankings of reward *genomes* (code fingerprints) using
Spearman ρ and top-k preservation. Candidate ids change after D.3 refine;
we key on normalized reward source code.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from raise_core.explore import RewardCandidate
from raise_core.rules import spearman_correlation
from raise_core.selection import candidate_nav_scalar


def genome_key(code: str) -> str:
    """Stable fingerprint for a reward source string."""
    normalized = str(code).strip().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()[:16]


def scores_from_stage1(candidates: Sequence[RewardCandidate]) -> Dict[str, float]:
    """Map genome → Score1 (higher better)."""
    out: Dict[str, float] = {}
    for c in candidates:
        key = genome_key(c.code)
        sc = float(c.score) if c.score is not None else float("-inf")
        # Keep best Score1 if duplicate codes appear.
        if key not in out or sc > out[key]:
            out[key] = sc
    return out


def scores_from_trained(
    candidates: Sequence[RewardCandidate],
) -> Dict[str, float]:
    """Map genome → navigation scalar from last_metrics (higher better)."""
    out: Dict[str, float] = {}
    for c in candidates:
        key = genome_key(c.code)
        sc = candidate_nav_scalar(c)
        if key not in out or sc > out[key]:
            out[key] = sc
    return out


def scores_from_trained_lineage(
    stage3: Sequence[RewardCandidate],
    stage2_scores: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Align Stage III scores to Stage II parent genomes when refine changed the code.

    Returns ``(s2_aligned, s3_aligned)`` keyed by parent genome (or self genome
    when ``parent_genome_key`` is absent). This keeps ρ(II,III) defined after
    successful D.3 refine instead of requiring exact code fingerprint overlap.
    """
    s2_out: Dict[str, float] = {}
    s3_out: Dict[str, float] = {}
    for c in stage3:
        parent_key = str((c.metadata or {}).get("parent_genome_key") or "").strip()
        self_key = genome_key(c.code)
        key = parent_key if parent_key and parent_key in stage2_scores else self_key
        if key not in stage2_scores and key != self_key:
            # Parent unknown — fall back to self if Stage II also has it.
            if self_key not in stage2_scores:
                continue
            key = self_key
        if key not in stage2_scores:
            continue
        sc3 = candidate_nav_scalar(c)
        if key not in s3_out or sc3 > s3_out[key]:
            s3_out[key] = sc3
            s2_out[key] = float(stage2_scores[key])
    return s2_out, s3_out


def _aligned_score_vectors(
    a: Mapping[str, float],
    b: Mapping[str, float],
) -> Tuple[List[str], List[float], List[float]]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    va = [float(a[k]) for k in keys]
    vb = [float(b[k]) for k in keys]
    return keys, va, vb


def spearman_rank_correlation(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Spearman ρ between two genome→score maps on their intersection.

    Returns dict with rho, n_overlap, and whether the estimate is usable.
    """
    keys, va, vb = _aligned_score_vectors(scores_a, scores_b)
    n = len(keys)
    if n < 2:
        return {
            "rho": None,
            "n_overlap": n,
            "usable": False,
            "reason": "need >= 2 overlapping genomes",
        }
    # Constant vectors → undefined Spearman
    if len(set(va)) < 2 or len(set(vb)) < 2:
        return {
            "rho": None,
            "n_overlap": n,
            "usable": False,
            "reason": "constant scores on one side",
        }
    rho = spearman_correlation(va, vb)
    usable = bool(np.isfinite(rho))
    return {
        "rho": float(rho) if usable else None,
        "n_overlap": n,
        "usable": usable,
        "reason": None if usable else "non-finite spearman",
        "overlap_keys": keys,
    }


def top_k_preservation(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    k: int = 3,
) -> Dict[str, Any]:
    """Fraction of top-k genomes in A that remain in top-k of B (on intersection)."""
    keys = set(scores_a.keys()) & set(scores_b.keys())
    if not keys:
        return {"k": k, "preservation": None, "n_overlap": 0, "top_a": [], "top_b": []}
    k_eff = max(1, min(int(k), len(keys)))
    top_a = sorted(keys, key=lambda g: float(scores_a[g]), reverse=True)[:k_eff]
    top_b = sorted(keys, key=lambda g: float(scores_b[g]), reverse=True)[:k_eff]
    kept = len(set(top_a) & set(top_b))
    return {
        "k": k_eff,
        "preservation": float(kept) / float(k_eff),
        "n_overlap": len(keys),
        "top_a": top_a,
        "top_b": top_b,
        "kept": kept,
    }


def compute_proxy_consistency(
    *,
    stage1: Sequence[RewardCandidate],
    stage2: Sequence[RewardCandidate],
    stage3: Optional[Sequence[RewardCandidate]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Build the §4.3.4-style report for one Algorithm 1 run.

    ``stage2`` / ``stage3`` should preferably be trained snapshots (with
    ``last_metrics``); final populations are accepted as fallback.
    """
    s1 = scores_from_stage1(stage1)
    s2 = scores_from_trained(stage2)
    report: Dict[str, Any] = {
        "paper_ref": "RAISE §4.3.4 / Appendix C.8",
        "n_stage1_genomes": len(s1),
        "n_stage2_genomes": len(s2),
        "stage1_vs_stage2": {
            "spearman": spearman_rank_correlation(s1, s2),
            "top_k": top_k_preservation(s1, s2, k=top_k),
        },
    }
    if stage3 is not None:
        s3 = scores_from_trained(stage3)
        report["n_stage3_genomes"] = len(s3)
        report["stage2_vs_stage3"] = {
            "spearman": spearman_rank_correlation(s2, s3),
            "top_k": top_k_preservation(s2, s3, k=top_k),
            "note": (
                "Exact code-fingerprint overlap. Often empty when D.3 refine "
                "succeeds (Stage III trains post-refine genomes never scored in II)."
            ),
        }
        s2_lin, s3_lin = scores_from_trained_lineage(stage3, s2)
        report["stage2_vs_stage3_lineage"] = {
            "spearman": spearman_rank_correlation(s2_lin, s3_lin),
            "top_k": top_k_preservation(s2_lin, s3_lin, k=top_k),
            "n_lineage_pairs": len(s2_lin),
            "note": (
                "Stage III score vs Stage II parent-genome score "
                "(uses metadata.parent_genome_key after refine)."
            ),
        }
        report["stage1_vs_stage3"] = {
            "spearman": spearman_rank_correlation(s1, s3),
            "top_k": top_k_preservation(s1, s3, k=top_k),
        }
        # Prefer lineage ρ in the summary when exact overlap is unusable.
        exact = report["stage2_vs_stage3"]["spearman"]
        lineage = report["stage2_vs_stage3_lineage"]["spearman"]
        report["stage2_vs_stage3_preferred"] = (
            "lineage"
            if (not exact.get("usable")) and lineage.get("usable")
            else "exact"
        )
    return report
