"""
Highway diagnostic feedback card for LLM reward-evolution prompts.

Sections (each prints "unavailable: <reason>" when its data was not recorded):

  A current performance      B training dynamics     C reward components
  D parent comparison        E lineage history       F representative episodes
  G diagnostic summary       H preference check      P population context

Every number comes from data the loop already stored: Stage II holdout
metrics, per-episode holdout records and training-rollout windows written by
``domains.highway.adapter``, Stage I ``score1_scenario_scores``, and the
candidate archive. Nothing is re-evaluated here.

``truncation_ladder`` lists progressively smaller renderings; the evolver
walks it until the full LLM request fits the token budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from domains.highway.metrics import LANE_LOCK_RATE, fitness_components
from domains.highway.objective_constants import PPO_GAMMA, SOFT_PROGRESS_M
from raise_core.feedback import (
    ArchiveRecord,
    CandidateArchive,
    FeedbackCardConfig,
    diff_code,
    summarize_code,
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    # +1 higher is better, -1 lower is better, 0 no preferred direction.
    direction: int
    fmt: str
    rate: bool = False


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("SR", "SR", "", +1, ".2f", rate=True),
    MetricSpec("CR", "CR", "", -1, ".2f", rate=True),
    MetricSpec("TR", "TR", "", -1, ".2f", rate=True),
    MetricSpec("mean_speed", "mean_speed", " m/s", +1, ".1f"),
    MetricSpec("speed_p10", "speed_p10", " m/s", +1, ".1f"),
    MetricSpec("mean_progress", "progress", " m", +1, ".0f"),
    MetricSpec("lane_change_rate", "lane_changes", "/step", 0, ".3f"),
    MetricSpec("overtakes_per_km", "overtakes", "/km", 0, ".2f"),
)
FITNESS = MetricSpec("fitness", "fitness", "", +1, ".3f")
_COMPACT_KEYS = ("SR", "CR", "TR", "mean_speed", "mean_progress", "fitness")
_ALIASES = {"mean_speed": ("ITR",), "mean_progress": ("PL",)}
_SPECS = {s.key: s for s in METRICS + (FITNESS,)}

_FLAG_LABELS = {
    "crawling": "crawling / too slow",
    "speed_with_collisions": "speed with collisions",
    "insufficient_progress": "insufficient progress",
    "unstable_training": "unstable training",
    "reward_hacking": "possible reward hacking",
    "insufficient_evidence": "insufficient evidence",
}

_LEGEND = (
    "Numbers and 'Observation' lines are measured; 'Hypothesis' lines are "
    "unverified interpretations. 'unavailable' means the data was not recorded "
    "- do not guess it."
)
_GUIDANCE = (
    "How to use this card: base each code change on a cited Observation and "
    "treat Hypotheses as unverified. Make one or two targeted, explainable "
    "edits; in your diagnosis name the observation each edit addresses and the "
    "behaviour you expect it to change."
)


# ------------------------------------------------------------------ metric access


def holdout_metrics(record: ArchiveRecord) -> Mapping[str, Any]:
    m = record.metrics
    h = m.get("holdout")
    return h if isinstance(h, Mapping) and h else m


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metric_value(record: ArchiveRecord, key: str) -> Optional[float]:
    if not record.evaluated:
        return None
    if key == "fitness":
        return _finite(record.metadata.get("fitness"))
    src = holdout_metrics(record)
    for k in (key, *_ALIASES.get(key, ())):
        v = _finite(src.get(k))
        if v is not None:
            return v
    return None


def _n_episodes(record: ArchiveRecord) -> Optional[int]:
    v = _finite(holdout_metrics(record).get("n_eval_episodes"))
    return int(v) if v is not None else None


def _standard_error(record: ArchiveRecord, spec: MetricSpec) -> Optional[float]:
    if spec.key == "fitness":
        return _finite(holdout_metrics(record).get("fitness_se"))
    if not spec.rate:
        return None
    p, n = metric_value(record, spec.key), _n_episodes(record)
    if p is None or not n:
        return None
    # Agresti-Coull style: rates of exactly 0 or 1 still carry sampling noise.
    adj = (p * n + 1.0) / (n + 2.0)
    return math.sqrt(adj * (1.0 - adj) / n)


def _fmt(spec: MetricSpec, value: Optional[float]) -> str:
    return "n/a" if value is None else format(value, spec.fmt)


def behaviour(record: ArchiveRecord) -> str:
    """Short behaviour characterization from holdout metrics and the speed gate."""
    if not record.evaluated:
        return "not evaluated"
    comps = fitness_components(record.metrics)
    gate = float(comps.get("gate", 0.0))
    sr = metric_value(record, "SR") or 0.0
    cr = metric_value(record, "CR") or 0.0
    tr = metric_value(record, "TR") or 0.0
    tags = []
    if comps.get("disqualified"):
        tags.append("near-stationary (fitness disqualified)")
    elif gate < 0.3:
        tags.append("crawling")
    elif gate < 0.7:
        tags.append("slow")
    else:
        tags.append("near free-flow speed")
    if cr >= 0.5:
        tags.append("crash-prone")
    elif cr >= 0.2:
        tags.append("some collisions")
    elif sr >= 0.9:
        tags.append("survives reliably")
    if tr >= 0.2:
        tags.append("leaves the road")
    lc = metric_value(record, "lane_change_rate")
    if lc is not None and lc < LANE_LOCK_RATE:
        tags.append("lane-locked")
    ofrac = _finite(holdout_metrics(record).get("overtake_episode_frac"))
    if ofrac is not None and ofrac >= 0.5:
        tags.append("overtakes in most episodes")
    return ", ".join(tags)


def _delta_text(spec: MetricSpec, child: ArchiveRecord, parent: ArchiveRecord) -> str:
    a, b = metric_value(parent, spec.key), metric_value(child, spec.key)
    if a is None or b is None:
        return f"{spec.label} n/a"
    d = b - a
    if abs(d) < 1e-12:
        return f"{spec.label} {_fmt(spec, a)} (same)"
    tags = []
    if spec.direction != 0:
        tags.append("better" if d * spec.direction > 0 else "worse")
    se_a, se_b = _standard_error(parent, spec), _standard_error(child, spec)
    if se_a is not None and se_b is not None and abs(d) <= 2.0 * math.hypot(se_a, se_b):
        tags.append("within noise")
    tag = (" " + ", ".join(tags)) if tags else ""
    return f"{spec.label} {_fmt(spec, a)}->{_fmt(spec, b)} ({d:+{spec.fmt}}{tag})"


def _metrics_inline(record: ArchiveRecord) -> str:
    if not record.evaluated:
        return "not evaluated (Stage II failed or skipped)"
    parts = [f"{_SPECS[k].label} {_fmt(_SPECS[k], metric_value(record, k))}" for k in _COMPACT_KEYS]
    return " ".join(parts)


def _delta_inline(child: ArchiveRecord, parent: ArchiveRecord) -> str:
    parts = []
    for key in _COMPACT_KEYS:
        spec = _SPECS[key]
        a, b = metric_value(parent, key), metric_value(child, key)
        if a is not None and b is not None:
            parts.append(f"{spec.label} {b - a:+{spec.fmt}}")
    return " ".join(parts) if parts else "no comparable metrics"


# ------------------------------------------------------------------ episode data


def _episodes(record: ArchiveRecord) -> List[Dict[str, Any]]:
    eps = record.metrics.get("holdout_episodes")
    return [e for e in eps if isinstance(e, dict)] if isinstance(eps, list) else []


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _return_key(eps: Sequence[Mapping[str, Any]]) -> str:
    return "disc_return" if eps and all("disc_return" in e for e in eps) else "return"


def preference_check(record: ArchiveRecord) -> Tuple[Optional[Dict[str, Any]], str]:
    """Mann-Whitney AUC: P(return of a successful episode > a failed one)."""
    eps = _episodes(record)
    if not eps:
        return None, "no per-episode holdout records for this candidate"
    key = _return_key(eps)
    pos = [float(e[key]) for e in eps if e.get("outcome") == "success" and key in e]
    neg = [float(e[key]) for e in eps if e.get("outcome") != "success" and key in e]
    if not pos or not neg:
        return None, (
            f"needs both outcomes; holdout had {len(pos)} successful and "
            f"{len(neg)} failed episodes"
        )
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return {"auc": wins / (len(pos) * len(neg)), "n_pos": len(pos), "n_neg": len(neg), "key": key}, ""


def _component_table(
    record: ArchiveRecord,
) -> Tuple[Optional[List[Tuple[str, float, Optional[float], Optional[float]]]], str, Optional[float]]:
    """Rows (key, mean all, mean success, mean failed) + coverage of the mean return."""
    eps = _episodes(record)
    if not eps:
        return None, "no per-episode holdout records for this candidate", None
    with_comps = [e for e in eps if isinstance(e.get("components"), dict)]
    if not with_comps:
        return None, "component sums were not recorded for this candidate", None
    keys: List[str] = []
    for e in with_comps:
        for k in e["components"]:
            if k not in keys:
                keys.append(k)
    if not keys:
        return None, "the reward returned no components (empty dict)", None
    succ = [e for e in with_comps if e.get("outcome") == "success"]
    fail = [e for e in with_comps if e.get("outcome") != "success"]
    rows = []
    for k in keys:
        rows.append(
            (
                k,
                _mean([float(e["components"].get(k, 0.0)) for e in with_comps]) or 0.0,
                _mean([float(e["components"].get(k, 0.0)) for e in succ]),
                _mean([float(e["components"].get(k, 0.0)) for e in fail]),
            )
        )
    rows.sort(key=lambda r: -abs(r[1]))
    mean_return = _mean([float(e.get("return", 0.0)) for e in with_comps]) or 0.0
    coverage = sum(r[1] for r in rows) / mean_return if abs(mean_return) > 1e-9 else None
    return rows, "", coverage


def _representatives(record: ArchiveRecord) -> List[Tuple[str, Dict[str, Any]]]:
    eps = _episodes(record)
    succ = sorted((e for e in eps if e.get("outcome") == "success"), key=lambda e: e.get("progress", 0.0))
    coll = sorted((e for e in eps if e.get("outcome") == "collision"), key=lambda e: e.get("steps", 0))
    off = sorted((e for e in eps if e.get("outcome") == "off_road"), key=lambda e: e.get("steps", 0))
    picks: List[Tuple[str, Dict[str, Any]]] = []
    typical = succ[len(succ) // 2] if succ else None
    if typical is not None:
        picks.append(("typical success", typical))
    if coll:
        picks.append(("typical collision", coll[len(coll) // 2]))
    if off:
        picks.append(("typical off-road", off[len(off) // 2]))
    elif len(succ) >= 2:
        slowest = min(succ, key=lambda e: e.get("mean_speed", 0.0))
        if slowest is not typical:
            picks.append(("slowest survivor", slowest))
    return picks


# ------------------------------------------------------------------ rendering


@dataclass(frozen=True)
class CardTrim:
    """One rendering level of the card (``name`` = truncation step that produced it)."""

    name: str
    trajectories: int
    history: int
    history_code: bool = True
    population: bool = True
    compact_instructions: bool = False
    detail: bool = True
    optional: bool = True
    diagnostics: bool = True


def truncation_ladder(config: FeedbackCardConfig) -> List[CardTrim]:
    """
    Cumulative truncation steps, in order: trajectory details, older code
    summaries, population context and history entries, prompt instructions,
    then B/C detail, all optional sections and finally the diagnostic summary.
    Current code, action constraints, A (current results), D (parent
    comparison) and the output schema are never cut.
    """
    trim = CardTrim("full", trajectories=int(config.max_trajectories), history=int(config.history_entries))
    ladder = [trim]

    def step(name: str, **changes: Any) -> None:
        nonlocal trim
        trim = replace(trim, name=name, **changes)
        ladder.append(trim)

    for k in range(trim.trajectories - 1, -1, -1):
        step(f"trajectories->{k}", trajectories=k)
    if trim.history > 0:
        step("history code summaries dropped", history_code=False)
    step("population context dropped", population=False)
    for k in range(trim.history - 1, -1, -1):
        step(f"history entries->{k}", history=k)
    step("instructions compacted", compact_instructions=True)
    step("training dynamics/components condensed", detail=False)
    step("optional sections dropped (B,C,E,F,H)", optional=False)
    step("diagnostic summary dropped", diagnostics=False)
    return ladder


@dataclass
class RenderedCard:
    text: str
    sections: List[str]
    history_ids: List[str]
    unavailable: Dict[str, str] = field(default_factory=dict)


class FeedbackCard:
    """
    Card for one LLM call.

    ``call_type``: ``mutation`` (one subject: the parent being mutated),
    ``crossover`` (two subjects: parents A and B) or ``reference`` (fresh
    proposals; the subject is the best evaluated candidate so far).
    """

    def __init__(
        self,
        *,
        call_type: str,
        subjects: Sequence[ArchiveRecord],
        archive: CandidateArchive,
        config: FeedbackCardConfig,
        population_note: str = "",
    ) -> None:
        if not subjects:
            raise ValueError("FeedbackCard needs at least one subject record")
        self.call_type = str(call_type)
        self.subjects = list(subjects)
        self.archive = archive
        self.config = config
        self.population_note = str(population_note or "").strip()
        shown = {s.candidate_id for s in self.subjects}
        if self.call_type != "crossover":
            shown |= set(self.subjects[0].parent_ids)
        self.history = archive.select_history(
            [s.candidate_id for s in self.subjects],
            int(config.history_entries),
            exclude=shown,
        )

    # ---------------------------------------------------------------- sections

    def _a(self, rec: ArchiveRecord, role: str) -> List[str]:
        lineage = (
            f"{rec.origin} of {', '.join(rec.parent_ids)}" if rec.parent_ids else rec.origin
        )
        head = f"[A] Current performance - {role} {rec.candidate_id} (gen {rec.generation}, {lineage})"
        if not rec.evaluated:
            return [head, "unavailable: no Stage II evaluation (training failed or was skipped)"]
        ctx = rec.metrics.get("eval_context") if isinstance(rec.metrics.get("eval_context"), dict) else {}
        steps = rec.metadata.get("train_steps")
        bits = [f"Stage II PPO {int(steps):,} steps" if steps else "Stage II"]
        if ctx.get("vehicles_count") is not None:
            bits.append(
                f"{ctx.get('traffic', 'holdout')} traffic ({ctx['vehicles_count']} vehicles, "
                f"density {float(ctx.get('vehicles_density') or 1.0):g})"
            )
        else:
            bits.append(f"{rec.metrics.get('eval_profile', 'holdout')} traffic")
        n = _n_episodes(rec)
        bits.append(f"{n} episodes" if n else "episode count n/a")
        det = ctx.get("deterministic", rec.metadata.get("eval_deterministic"))
        if det is not None:
            bits.append("deterministic policy" if det else "stochastic policy")
        values = " | ".join(
            f"{s.label} {_fmt(s, metric_value(rec, s.key))}{s.unit if metric_value(rec, s.key) is not None else ''}"
            for s in METRICS
        )
        comps = fitness_components(rec.metrics)
        se = _standard_error(rec, FITNESS)
        fit = metric_value(rec, "fitness")
        fit_txt = _fmt(FITNESS, fit) + (f" +/- {se:.3f}" if se is not None else "")
        return [
            head,
            "Eval: " + "; ".join(bits) + ".",
            values,
            f"fitness {fit_txt} (selection objective; speed gate {float(comps.get('gate', 0.0)):.2f} of 1)",
            f"Behaviour: {behaviour(rec)}",
        ]

    def _b(self, rec: ArchiveRecord, detail: bool) -> Tuple[List[str], Optional[str]]:
        head = f"[B] Training dynamics - {rec.candidate_id}"
        if not self.config.training_dynamics:
            return [f"{head}: unavailable (disabled in config)"], "disabled in config"
        td = rec.metadata.get("training_dynamics")
        windows = [w for w in (td or {}).get("windows", []) if isinstance(w, dict)]
        if not windows:
            reason = "no training-rollout statistics recorded (no checkpoint evaluations are run)"
            return [f"{head}: unavailable ({reason})"], reason
        rows = []
        shown = windows if detail else [windows[0], windows[-1]]
        for w in shown:
            lo, hi = (w.get("steps") or [0, 0])[:2]
            if not w.get("n_episodes"):
                rows.append(f"steps {lo}-{hi}: 0 episodes finished")
                continue
            ret = _finite(w.get("mean_return"))
            rows.append(
                f"steps {lo}-{hi}: {w['n_episodes']} ep SR {w.get('SR', 0):.2f} CR {w.get('CR', 0):.2f} "
                f"TR {w.get('TR', 0):.2f} return {'n/a' if ret is None else f'{ret:+.1f}'} "
                f"speed {float(w.get('mean_speed', 0.0)):.1f} m/s"
            )
        return [
            f"{head} (outcomes of stochastic training rollouts in training traffic; not separate evaluations)",
            " | ".join(rows),
        ], None

    def _c(self, rec: ArchiveRecord, detail: bool) -> Tuple[List[str], Optional[str]]:
        head = f"[C] Reward components - {rec.candidate_id}"
        if not self.config.component_instrumentation:
            return [f"{head}: unavailable (disabled in config)"], "disabled in config"
        rows, reason, coverage = _component_table(rec)
        if rows is None:
            return [f"{head}: unavailable ({reason})"], reason
        shown = rows if detail else rows[:3]

        def _v(x: Optional[float]) -> str:
            return "n/a" if x is None else f"{x:+.1f}"

        lines = [f"{head} (mean per-episode sum on holdout: all | successful | failed)"]
        lines.append(" ; ".join(f"{k} {_v(a)} | {_v(s)} | {_v(f)}" for k, a, s, f in shown))
        if len(shown) < len(rows):
            lines[-1] += f" ; +{len(rows) - len(shown)} smaller terms"
        if coverage is not None:
            lines.append(f"Components sum to {coverage:.0%} of the mean episode return.")
        return lines, None

    def _d(self) -> List[str]:
        legend = (
            "better/worse by metric direction (lower CR/TR, higher SR/speed/progress/fitness); "
            "'within noise' = |delta| <= 2 SE"
        )
        if self.call_type == "crossover" and len(self.subjects) >= 2:
            a, b = self.subjects[0], self.subjects[1]
            if not (a.evaluated and b.evaluated):
                return [f"[D] Parent A {a.candidate_id} vs parent B {b.candidate_id}: unavailable (one is not evaluated)"]
            return [
                f"[D] Parent A {a.candidate_id} vs parent B {b.candidate_id} (B - A; {legend})",
                " | ".join(_delta_text(s, b, a) for s in METRICS + (FITNESS,)),
                f"Behaviour: A {behaviour(a)}; B {behaviour(b)}",
                f"Code difference A->B: {diff_code(a.code, b.code)}",
            ]
        child = self.subjects[0]
        if not child.parent_ids:
            return [f"[D] Parent comparison: unavailable ({child.candidate_id} is a fresh {child.origin} proposal with no parent)"]
        lines: List[str] = []
        for pid in child.parent_ids[:2]:
            parent = self.archive.get(pid)
            if parent is None or not parent.evaluated or not child.evaluated:
                why = "not archived" if parent is None else "missing evaluation"
                lines.append(f"[D] {child.candidate_id} vs parent {pid}: unavailable ({why})")
                continue
            lines += [
                f"[D] {child.candidate_id} vs parent {pid} (child - parent; {legend})",
                " | ".join(_delta_text(s, child, parent) for s in METRICS + (FITNESS,)),
                f"Behaviour: {behaviour(parent)} -> {behaviour(child)}",
                f"Code change: {diff_code(parent.code, child.code)}",
            ]
        return lines

    def _e(self, n: int, with_code: bool) -> Tuple[List[str], List[str]]:
        entries = self.history[:n]
        shown = ", ".join(s.candidate_id for s in self.subjects)
        if not entries:
            return [f"[E] History: unavailable (no other evaluated relatives of {shown} in the archive)"], []
        lines = [f"[E] History - up to {n} relatives ({shown} and the compared parents are above, not repeated)"]
        for rec, relation in entries:
            lines.append(f"- {rec.candidate_id} | {relation} | gen {rec.generation} {rec.origin}")
            lines.append(f"  {_metrics_inline(rec)} | {behaviour(rec)}")
            if rec.parent_ids:
                parent = self.archive.get(rec.parent_ids[0])
                if parent is not None and parent.evaluated and rec.evaluated:
                    lines.append(
                        f"  vs its parent {parent.candidate_id}: {_delta_inline(rec, parent)}; "
                        f"code: {diff_code(parent.code, rec.code)}"
                    )
                else:
                    lines.append(f"  vs its parent {rec.parent_ids[0]}: unavailable")
            if with_code:
                lines.append(f"  code: {summarize_code(rec.code)}")
        return lines, [rec.candidate_id for rec, _ in entries]

    def _f(self, n: int) -> Tuple[List[str], Optional[str]]:
        picks: List[Tuple[ArchiveRecord, str, Dict[str, Any]]] = []
        per_subject = [[(s, label, ep) for label, ep in _representatives(s)] for s in self.subjects]
        while len(picks) < n and any(per_subject):
            for queue in per_subject:
                if queue and len(picks) < n:
                    picks.append(queue.pop(0))
        if not picks:
            reason = "no per-episode holdout records"
            return [f"[F] Representative episodes: unavailable ({reason})"], reason
        lines = [f"[F] Representative holdout episodes (at most {n})"]
        for rec, label, e in picks:
            ctx = rec.metrics.get("eval_context") if isinstance(rec.metrics.get("eval_context"), dict) else {}
            freq = _finite(ctx.get("policy_frequency"))
            secs = f" ({e.get('steps', 0) / freq:.1f} s)" if freq else ""
            line = (
                f"- {rec.candidate_id} {label}: {e.get('steps', 0)} steps{secs}, "
                f"{float(e.get('mean_speed', 0.0)):.1f} m/s, {float(e.get('progress', 0.0)):.0f} m, "
                f"{e.get('lane_changes', 0)} lane changes, {e.get('overtakes', 0)} overtakes; "
                f"return {float(e.get('return', 0.0)):+.1f}"
            )
            if "disc_return" in e:
                line += f" (discounted {float(e['disc_return']):+.1f})"
            comps = e.get("components")
            if isinstance(comps, dict) and comps:
                top = sorted(comps.items(), key=lambda kv: -abs(float(kv[1])))[:2]
                line += "; top terms " + ", ".join(f"{k} {float(v):+.1f}" for k, v in top)
            lines.append(line)
        return lines, None

    def _g_one(self, rec: ArchiveRecord) -> Tuple[List[str], List[str], List[str]]:
        flags: List[str] = []
        obs: List[str] = []
        hyp: List[str] = []
        cid = rec.candidate_id
        if not rec.evaluated:
            return ["insufficient_evidence"], [f"{cid}: no Stage II evaluation"], []
        comps = fitness_components(rec.metrics)
        gate = float(comps.get("gate", 0.0))
        sr = metric_value(rec, "SR") or 0.0
        cr = metric_value(rec, "CR") or 0.0
        speed = metric_value(rec, "mean_speed")
        p10 = metric_value(rec, "speed_p10")
        prog = metric_value(rec, "mean_progress")
        n = _n_episodes(rec)

        if comps.get("disqualified") or gate < 0.5:
            flags.append("crawling")
            obs.append(
                f"{cid}: speed gate {gate:.2f}, so fitness keeps {gate:.0%} of the survival credit "
                f"(speed_p10 {_fmt(_SPECS['speed_p10'], p10)} m/s, mean_speed {_fmt(_SPECS['mean_speed'], speed)} m/s)"
            )
            hyp.append(
                f"{cid}: slow driving costs too little relative to its safety benefit under this reward "
                "(compare speed/progress terms with the penalties in [C])"
            )
        if not comps.get("disqualified") and gate >= 0.5 and cr >= 0.25:
            flags.append("speed_with_collisions")
            obs.append(f"{cid}: CR {cr:.2f} at speed gate {gate:.2f} (mean_speed {_fmt(_SPECS['mean_speed'], speed)} m/s)")
            hyp.append(
                f"{cid}: collision avoidance is under-weighted relative to speed/progress reward "
                "(e.g. a small collision cost or no headway/gap term)"
            )
        if sr >= 0.5 and prog is not None and prog < SOFT_PROGRESS_M:
            flags.append("insufficient_progress")
            obs.append(f"{cid}: progress {prog:.0f} m/episode with SR {sr:.2f}")
            hyp.append(f"{cid}: the reward pays for staying alive more than for distance covered")

        eps = _episodes(rec)
        if eps:
            key = _return_key(eps)
            s_ret = _mean([float(e[key]) for e in eps if e.get("outcome") == "success" and key in e])
            f_ret = _mean([float(e[key]) for e in eps if e.get("outcome") != "success" and key in e])
            if s_ret is not None and f_ret is not None:
                obs.append(
                    f"{cid}: mean {'discounted ' if key == 'disc_return' else ''}episode return "
                    f"successful {s_ret:+.1f} vs failed {f_ret:+.1f}"
                )

        windows = [
            w
            for w in ((rec.metadata.get("training_dynamics") or {}).get("windows") or [])
            if isinstance(w, dict) and int(w.get("n_episodes") or 0) >= 3
        ]
        if len(windows) >= 2:
            srs = [float(w.get("SR", 0.0)) for w in windows]
            crs = [float(w.get("CR", 0.0)) for w in windows]
            if max(srs[:-1]) - srs[-1] >= 0.25:
                flags.append("unstable_training")
                obs.append(f"{cid}: training-rollout SR peaked at {max(srs[:-1]):.2f} and ended at {srs[-1]:.2f}")
                hyp.append(f"{cid}: the policy found a late exploit or the reward scale destabilises PPO")
            elif crs[-1] - crs[0] >= 0.25:
                flags.append("unstable_training")
                obs.append(f"{cid}: training-rollout CR rose from {crs[0]:.2f} to {crs[-1]:.2f}")
                hyp.append(f"{cid}: speed reward grows faster during training than the collision cost restrains it")
        elif self.config.training_dynamics:
            obs.append(f"{cid}: training stability unknown (fewer than 2 rollout windows with >= 3 episodes)")

        pref, _ = preference_check(rec)
        if pref is not None and pref["auc"] < 0.5:
            flags.append("reward_hacking")
            obs.append(
                f"{cid}: failed episodes out-earn successful ones (preference AUC {pref['auc']:.2f}, "
                f"{pref['n_pos']}x{pref['n_neg']} pairs)"
            )
            hyp.append(
                f"{cid}: some term pays more on the way to failure than survival does "
                "(crash bonus, large speed reward, or a negative per-step reward that makes ending early pay)"
            )
        lc = metric_value(rec, "lane_change_rate")
        if lc is not None and lc < LANE_LOCK_RATE and sr >= 0.9:
            flags.append("reward_hacking")
            obs.append(f"{cid}: lane_change_rate {lc:.3f} with SR {sr:.2f}; fitness applies a lane-lock penalty")
            hyp.append(f"{cid}: lane changes never pay off under this reward, so the policy idles in one lane")
        _, _, coverage = _component_table(rec)
        if coverage is not None and abs(coverage - 1.0) > 0.2:
            obs.append(f"{cid}: reward components explain only {coverage:.0%} of the total return")

        if n is not None and n < 10:
            flags.append("insufficient_evidence")
            obs.append(f"{cid}: only {n} evaluation episodes")
        if not flags:
            obs.append(f"{cid}: no failure pattern matched (crawling, speed with collisions, insufficient progress, unstable training, reward hacking)")
        return flags, obs, hyp

    def _g(self) -> List[str]:
        flags: List[str] = []
        obs: List[str] = []
        hyp: List[str] = []
        for rec in self.subjects:
            f, o, h = self._g_one(rec)
            flags += [x for x in f if x not in flags]
            obs += o
            hyp += h
        lines = ["[G] Diagnostic summary"]
        lines.append("Flags: " + (", ".join(_FLAG_LABELS[f] for f in flags) if flags else "none"))
        lines += [f"Observation: {o}" for o in obs]
        lines += [f"Hypothesis: {h}" for h in hyp]
        return lines

    def _h(self, rec: ArchiveRecord) -> Tuple[List[str], Optional[str]]:
        head = f"[H] Preference check - {rec.candidate_id}"
        pref, reason = preference_check(rec)
        lines: List[str] = []
        if pref is None:
            lines.append(f"{head}: holdout episodes unavailable ({reason})")
        else:
            disc = f"discounted, gamma {PPO_GAMMA:g}" if pref["key"] == "disc_return" else "undiscounted"
            lines.append(
                f"{head}: P(return of a successful episode > a failed one) = {pref['auc']:.2f} "
                f"(Mann-Whitney AUC over {pref['n_pos']}x{pref['n_neg']} holdout episode pairs, {disc}; "
                "survivors run longer, so duration contributes)"
            )
        safety = _finite((rec.metadata.get("score1_scenario_scores") or {}).get("safety"))
        if safety is not None:
            n_traj = rec.metadata.get("score1_n_pairs")
            lines.append(
                f"Fixed Stage I dataset: outcome-ordering AUC {(safety + 1.0) / 2.0:.2f} "
                f"(success > timeout > collision on discounted returns"
                + (f", {int(n_traj)} trajectories" if n_traj else "")
                + ")"
            )
        if pref is None and safety is None:
            return lines, reason
        return lines, None

    # ---------------------------------------------------------------- render

    def render(self, trim: CardTrim) -> RenderedCard:
        out: List[str] = [f"=== DIAGNOSTIC FEEDBACK CARD ({self.call_type}) ===", _LEGEND]
        sections: List[str] = []
        unavailable: Dict[str, str] = {}
        roles = {
            "mutation": ["parent to mutate"],
            "crossover": ["parent A", "parent B"],
            "reference": ["reference (best fitness so far)"],
        }.get(self.call_type, ["candidate"] * len(self.subjects))

        for i, rec in enumerate(self.subjects):
            out += self._a(rec, roles[min(i, len(roles) - 1)])
        sections.append("A")

        if trim.optional:
            for letter, render_one in (("B", self._b), ("C", self._c)):
                reasons = []
                for rec in self.subjects:
                    lines, reason = render_one(rec, trim.detail)
                    out += lines
                    if reason:
                        reasons.append(f"{rec.candidate_id}: {reason}")
                sections.append(letter)
                if len(reasons) == len(self.subjects):
                    unavailable[letter] = "; ".join(reasons)

        out += self._d()
        sections.append("D")

        history_ids: List[str] = []
        if trim.optional and trim.history > 0:
            lines, history_ids = self._e(trim.history, trim.history_code)
            out += lines
            sections.append("E")
            if not history_ids:
                unavailable["E"] = "no other evaluated relatives"

        if trim.optional and trim.trajectories > 0:
            lines, reason = self._f(trim.trajectories)
            out += lines
            sections.append("F")
            if reason:
                unavailable["F"] = reason

        if trim.diagnostics:
            out += self._g()
            sections.append("G")

        if trim.optional:
            reasons = []
            for rec in self.subjects:
                lines, reason = self._h(rec)
                out += lines
                if reason:
                    reasons.append(f"{rec.candidate_id}: {reason}")
            sections.append("H")
            if len(reasons) == len(self.subjects):
                unavailable["H"] = "; ".join(reasons)

        if trim.population and self.population_note:
            out.append(f"[P] Population context (latest generation): {self.population_note}")
            sections.append("P")

        out.append(_GUIDANCE)
        return RenderedCard(
            text="\n".join(out),
            sections=sections,
            history_ids=history_ids,
            unavailable=unavailable,
        )


__all__ = [
    "CardTrim",
    "FeedbackCard",
    "METRICS",
    "RenderedCard",
    "behaviour",
    "holdout_metrics",
    "metric_value",
    "preference_check",
    "truncation_ladder",
]
