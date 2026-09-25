"""Gate readiness + Stage III elite assembly (highway-aware, CrowdNav-safe)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from raise_core.explore import RewardCandidate
from raise_core.surrogate.features import code_sha256


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def mean_val_mae(
    fit_metrics: Optional[Dict[str, Any]],
    *,
    keys: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """Mean per-target validation MAE from SurrogateModel.fit metrics."""
    if not fit_metrics:
        return None
    per = fit_metrics.get("per_target") or {}
    if not isinstance(per, dict) or not per:
        return None
    want = set(keys) if keys is not None else None
    vals: List[float] = []
    for name, row in per.items():
        if want is not None and str(name) not in want:
            continue
        if not isinstance(row, dict):
            continue
        mae = _finite(row.get("mae"), float("nan"))
        if math.isfinite(mae):
            vals.append(mae)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# Rate-like targets for gate MAE (avoid mixing m/s and meters into the mean).
_GATE_MAE_KEYS: Tuple[str, ...] = ("SR", "CR", "TR", "soft_success")


def surrogate_hard_gate_ready(
    *,
    n_labeled: int,
    min_labels: int,
    fit_metrics: Optional[Dict[str, Any]] = None,
    max_val_mae: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Hard gate may run if enough labels **or** validation MAE is already low.

    CrowdNav default: only ``n_labeled >= min_labels`` (``max_val_mae=None``).
    Highway Phase 4: ``min_labels=16`` or mean MAE on rate targets ≤ threshold.

    MAE uses SR/CR/TR/(soft_success) only — mean_speed/progress are different
    units and would dominate an unscaled mean (e.g. MAE≈15–70 forever).
    """
    n = int(n_labeled)
    need = int(min_labels)
    if n >= need:
        return True, f"n_labeled={n}>={need}"
    if max_val_mae is not None:
        mae = mean_val_mae(fit_metrics, keys=_GATE_MAE_KEYS)
        if mae is None:
            # Fall back to whatever targets exist if soft_success absent.
            mae = mean_val_mae(fit_metrics, keys=("SR", "CR", "TR"))
        if mae is not None and mae <= float(max_val_mae):
            return True, f"val_mae={mae:.4f}<={float(max_val_mae):.4f} (n={n})"
        if mae is not None:
            return False, (
                f"n_labeled={n}<{need} and val_mae={mae:.4f}>{float(max_val_mae):.4f}"
            )
    return False, f"n_labeled={n}<{need}"


def _unique_append(
    pool: List[RewardCandidate],
    cand: Optional[RewardCandidate],
    *,
    seen_hash: set[str],
    seen_ids: set[str],
) -> None:
    if cand is None:
        return
    cid = str(cand.candidate_id)
    digest = code_sha256(str(cand.code or ""))
    if cid in seen_ids or digest in seen_hash:
        return
    pool.append(cand)
    seen_ids.add(cid)
    seen_hash.add(digest)


def _metrics_of(cand: RewardCandidate) -> Dict[str, Any]:
    md = cand.metadata or {}
    m = md.get("last_metrics")
    return dict(m) if isinstance(m, dict) else {}


def _has_cached_fitness(cand: RewardCandidate) -> bool:
    md = cand.metadata or {}
    return md.get("fitness") is not None or md.get("selection_scalar") is not None


def _selection_scalar(cand: RewardCandidate) -> float:
    """Official fitness (highway) or nav scalar (CrowdNav)."""
    from raise_core.selection import candidate_fitness

    md = cand.metadata or {}
    for key in ("fitness", "selection_scalar"):
        cached = md.get(key)
        if cached is not None:
            v = _finite(cached, float("nan"))
            if math.isfinite(v):
                return v
    return float(candidate_fitness(cand))


def _soft_success(cand: RewardCandidate) -> float:
    m = _metrics_of(cand)
    if "soft_success" in m:
        return _finite(m.get("soft_success"), 0.0)
    return 0.0


def pick_best_by_scalar(pool: Sequence[RewardCandidate]) -> Optional[RewardCandidate]:
    """
    Best elite for Stage III forcing.

    If every labeled candidate has a fresh ``pareto_rank``, use lowest rank.
    Otherwise fall back to cached ``fitness`` / nav scalar (never let a lone
    stamped genome beat a higher-fitness unstamped one).
    """
    labeled = [
        c
        for c in pool
        if _metrics_of(c) or _has_cached_fitness(c)
    ]
    if not labeled:
        return None
    all_stamped = all((c.metadata or {}).get("pareto_rank") is not None for c in labeled)
    if all_stamped:
        return min(labeled, key=lambda c: int((c.metadata or {}).get("pareto_rank", 10**9)))
    best = None
    best_v = float("-inf")
    for c in labeled:
        v = _selection_scalar(c)
        if v > best_v:
            best_v = v
            best = c
    return best


def score1_elite_ok(
    cand: Optional[RewardCandidate],
    *,
    soft_success_min: float = 0.25,
    scalar_min: Optional[float] = None,
) -> bool:
    """Whether Score1-best is good enough to force into Stage III."""
    if cand is None:
        return False
    if _soft_success(cand) >= float(soft_success_min):
        return True
    if scalar_min is not None and _selection_scalar(cand) >= float(scalar_min):
        return True
    # No Stage II metrics yet — do not force Score1-only crawl survivors.
    if not _metrics_of(cand) and not _has_cached_fitness(cand):
        return False
    return _selection_scalar(cand) >= 0.0


def assemble_stage3_population(
    gated: Sequence[RewardCandidate],
    *,
    full_pool: Sequence[RewardCandidate],
    best_s2: Optional[RewardCandidate] = None,
    best_s1: Optional[RewardCandidate] = None,
    domain: str = "crowdnav",
    soft_success_min: float = 0.25,
) -> Tuple[List[RewardCandidate], Dict[str, Any]]:
    """
    Stage III input = kept ∪ best_s2 ∪ best_fitness; Score1-best only if strong.

    CrowdNav: still unions elites (behavior-preserving additive). Highway uses
    soft_success / fitness to gate Score1-best.
    """
    out: List[RewardCandidate] = []
    seen_h: set[str] = set()
    seen_i: set[str] = set()
    for c in gated:
        _unique_append(out, c, seen_hash=seen_h, seen_ids=seen_i)

    best_scalar = pick_best_by_scalar(full_pool)
    if best_s2 is None:
        best_s2 = best_scalar

    added: List[str] = []
    for label, cand in (("best_s2", best_s2), ("best_fitness", best_scalar)):
        before = len(out)
        _unique_append(out, cand, seen_hash=seen_h, seen_ids=seen_i)
        if len(out) > before and cand is not None:
            added.append(f"{label}:{cand.candidate_id}")

    s1_added = False
    domain_key = str(domain or "crowdnav").strip().lower()
    if domain_key == "highway":
        scalars = [
            _selection_scalar(c)
            for c in full_pool
            if _metrics_of(c) or _has_cached_fitness(c)
        ]
        scalar_floor = (
            float(sorted(scalars)[len(scalars) // 2]) if scalars else 0.0
        )
        if score1_elite_ok(
            best_s1,
            soft_success_min=soft_success_min,
            scalar_min=scalar_floor,
        ):
            before = len(out)
            _unique_append(out, best_s1, seen_hash=seen_h, seen_ids=seen_i)
            s1_added = len(out) > before
    else:
        # CrowdNav: keep historical behavior — Score1-best may already be in gated;
        # still ensure it is present (additive, never drops).
        before = len(out)
        _unique_append(out, best_s1, seen_hash=seen_h, seen_ids=seen_i)
        s1_added = len(out) > before

    report = {
        "n_gated": len(list(gated)),
        "n_stage3": len(out),
        "forced_elites": added,
        "score1_best_added": bool(s1_added),
        "score1_best_id": None if best_s1 is None else str(best_s1.candidate_id),
        "domain": domain_key,
    }
    return out, report


def epoch_population_stats(population: Sequence[Any]) -> Dict[str, Any]:
    """Score1 spread + soft_success / fitness summary for logs."""
    scores: List[float] = []
    softs: List[float] = []
    scalars: List[float] = []
    for c in population:
        if getattr(c, "score", None) is not None:
            s = _finite(c.score, float("nan"))
            if math.isfinite(s):
                scores.append(s)
        m: Dict[str, Any] = {}
        if isinstance(c, RewardCandidate):
            m = _metrics_of(c)
        elif isinstance(getattr(c, "metadata", None), dict):
            raw = (c.metadata or {}).get("last_metrics")
            m = dict(raw) if isinstance(raw, dict) else {}
        if "soft_success" in m:
            softs.append(_finite(m.get("soft_success"), 0.0))
        has_scalar = bool(m) or (
            isinstance(c, RewardCandidate) and _has_cached_fitness(c)
        )
        if has_scalar and isinstance(c, RewardCandidate):
            scalars.append(_selection_scalar(c))
    return {
        "n": len(list(population)),
        "score1_mean": float(sum(scores) / len(scores)) if scores else None,
        "score1_spread": (
            float(max(scores) - min(scores))
            if len(scores) >= 2
            else (0.0 if scores else None)
        ),
        "soft_success_mean": float(sum(softs) / len(softs)) if softs else None,
        "soft_success_max": float(max(softs)) if softs else None,
        # fitness_* is canonical; scalar_* kept for older plot scripts.
        "fitness_mean": float(sum(scalars) / len(scalars)) if scalars else None,
        "fitness_max": float(max(scalars)) if scalars else None,
        "scalar_mean": float(sum(scalars) / len(scalars)) if scalars else None,
        "scalar_max": float(max(scalars)) if scalars else None,
        "n_with_proxy": len(scalars),
    }
