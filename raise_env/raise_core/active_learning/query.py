"""
Query scoring: which candidates / scenarios to acquire next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple



@dataclass
class QueryItem:
    """One unit of work for the active-learning worker."""

    kind: str  # "stage1_scenario" | "stage2_label" | "stage3_label"
    priority: float
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "priority": float(self.priority),
            "payload": dict(self.payload),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryItem":
        return cls(
            kind=str(data.get("kind") or "stage2_label"),
            priority=float(data.get("priority") or 0.0),
            payload=dict(data.get("payload") or {}),
            reason=str(data.get("reason") or ""),
        )


DEFAULT_WEIGHTS = {
    "u": 0.45,
    "disagree": 0.35,
    "borderline": 0.15,
    "diversity": 0.15,
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _rank_percentiles(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank / max(n - 1, 1)
    return ranks


def _minmax_norm(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _quality_from_pred(pred: Any) -> float:
    if pred is None:
        return float("-inf")
    y_hat = getattr(pred, "y_hat", None)
    if isinstance(pred, dict):
        y_hat = pred.get("y_hat", pred)
    if not isinstance(y_hat, dict):
        return float("-inf")
    from raise_core.surrogate.targets import quality_from_y_hat

    return float(quality_from_y_hat(y_hat))


def _uncertainty_from_pred(pred: Any) -> float:
    if pred is None:
        return 0.0
    if hasattr(pred, "uncertainty"):
        return max(0.0, _finite(pred.uncertainty, 0.0))
    if isinstance(pred, dict):
        return max(0.0, _finite(pred.get("uncertainty"), 0.0))
    return 0.0


def _score1_value(raw: Any, candidate: Any) -> float:
    if raw is not None:
        if hasattr(raw, "train_score") and raw.train_score is not None:
            return _finite(raw.train_score, float("nan"))
        if hasattr(raw, "score"):
            return _finite(raw.score, float("nan"))
        return _finite(raw, float("nan"))
    score = getattr(candidate, "score", None)
    return _finite(score, float("nan"))


def _fingerprint(candidate: Any, payload_extra: Optional[Dict[str, Any]] = None) -> List[float]:
    if payload_extra and isinstance(payload_extra.get("behavior_fingerprint"), list):
        out: List[float] = []
        for x in payload_extra["behavior_fingerprint"]:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    md = getattr(candidate, "metadata", None) or {}
    fp = md.get("behavior_fingerprint")
    if isinstance(fp, list):
        out = []
        for x in fp:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    return []


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        xa = float(a[i]) if i < len(a) else 0.0
        xb = float(b[i]) if i < len(b) else 0.0
        d = xa - xb
        s += d * d
    return math.sqrt(s)


def _redundancy(
    fp: Sequence[float],
    reference_fps: Sequence[Sequence[float]],
) -> float:
    if not fp or not reference_fps:
        return 0.0
    dists = [_l2(fp, ref) for ref in reference_fps if ref]
    if not dists:
        return 0.0
    # Closer ⇒ more redundant. Map min distance to [0,1] via 1/(1+d).
    return 1.0 / (1.0 + min(dists))


def _weak_scenario_cluster(
    score1_results: Optional[Sequence[Any]],
    uncertainties: Sequence[float],
    u_thresh: float = 0.6,
) -> List[str]:
    if not score1_results:
        return []
    counts: Dict[str, int] = {}
    for raw, u in zip(score1_results, uncertainties):
        if u < u_thresh or raw is None:
            continue
        scenarios = getattr(raw, "scenario_scores", None) or {}
        if not isinstance(scenarios, dict) or not scenarios:
            continue
        worst = sorted(scenarios.items(), key=lambda kv: float(kv[1]))[:2]
        for sid, _rho in worst:
            counts[str(sid)] = counts.get(str(sid), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [sid for sid, cnt in ranked if cnt >= 2][:5]


def score_queries(
    candidates: List[Any],
    *,
    surrogate_preds: List[Any],
    score1_results: Optional[List[Any]] = None,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    promote_threshold: float = 0.0,
    reference_fingerprints: Optional[Sequence[Sequence[float]]] = None,
    max_stage2_labels: Optional[int] = None,
    max_stage1_scenarios: int = 1,
    already_labeled_ids: Optional[Sequence[str]] = None,
) -> List[QueryItem]:
    """
    Rank acquisition targets by disagreement / uncertainty.

    See PLAN.md §3–4. Prefer ``stage2_label``; optionally emit ``stage1_scenario``.
    """
    if len(candidates) != len(surrogate_preds):
        raise ValueError("candidates and surrogate_preds length mismatch")
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return []

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    s1_vals = []
    for i, cand in enumerate(candidates):
        raw = None
        if score1_results is not None and i < len(score1_results):
            raw = score1_results[i]
        val = _score1_value(raw, cand)
        if not math.isfinite(val):
            val = 0.0
        s1_vals.append(val)

    qualities = [_quality_from_pred(p) for p in surrogate_preds]
    for i, q in enumerate(qualities):
        if not math.isfinite(q):
            qualities[i] = 0.0
    uncertainties = [_uncertainty_from_pred(p) for p in surrogate_preds]
    u_norm = _minmax_norm(uncertainties)
    s1_pct = _rank_percentiles(s1_vals)
    q_pct = _rank_percentiles(qualities)

    labeled = {str(x) for x in (already_labeled_ids or [])}
    refs = list(reference_fingerprints or [])

    scored: List[Tuple[float, QueryItem]] = []
    for i, cand in enumerate(candidates):
        cid = str(getattr(cand, "candidate_id", f"cand_{i}"))
        code = str(getattr(cand, "code", "") or "")
        if not code:
            continue
        # Skip if payload says already labeled via code hash short id — optional.
        d_agree = abs(s1_pct[i] - q_pct[i])
        border = 1.0 / (1.0 + abs(qualities[i] - float(promote_threshold)))
        fp = _fingerprint(cand)
        red = _redundancy(fp, refs)
        priority = (
            w["u"] * u_norm[i]
            + w["disagree"] * d_agree
            + w["borderline"] * border
            - w["diversity"] * red
        )
        reason = (
            f"uncertainty={u_norm[i]:.2f}; disagree_s1_surr={d_agree:.2f}; "
            f"borderline={border:.2f}; redundancy={red:.2f}"
        )
        payload = {
            "candidate_id": cid,
            "code": code,
            "score1": s1_vals[i],
            "y_hat": getattr(surrogate_preds[i], "y_hat", None)
            if not isinstance(surrogate_preds[i], dict)
            else (surrogate_preds[i].get("y_hat") or surrogate_preds[i]),
            "uncertainty": uncertainties[i],
            "behavior_fingerprint": fp,
        }
        if cid in labeled:
            # Still allow but down-weight heavily.
            priority -= 1.0
            reason += "; already_labeled"
        scored.append(
            (
                priority,
                QueryItem(
                    kind="stage2_label",
                    priority=float(priority),
                    payload=payload,
                    reason=reason,
                ),
            )
        )

    scored.sort(key=lambda t: t[0], reverse=True)
    limit_s2 = int(max_stage2_labels) if max_stage2_labels is not None else int(top_k)
    items = [item for _, item in scored[: max(0, limit_s2)]]

    # Optional stage1_scenario from weak-sid cluster among high-uncertainty.
    weak_sids = _weak_scenario_cluster(score1_results, u_norm)
    if weak_sids and max_stage1_scenarios > 0 and len(items) < top_k:
        mean_u = sum(u_norm) / max(len(u_norm), 1)
        items.append(
            QueryItem(
                kind="stage1_scenario",
                priority=float(0.5 * mean_u),
                payload={"scenario_ids": weak_sids},
                reason=f"weak_sid_cluster={','.join(weak_sids)}",
            )
        )

    items.sort(key=lambda q: q.priority, reverse=True)
    return items[: int(top_k)]
