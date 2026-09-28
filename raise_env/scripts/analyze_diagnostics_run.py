#!/usr/bin/env python
"""Analyze a highway RAISE run that wrote diagnostics_*.jsonl files.

Purely descriptive — plateau thresholds here are exploratory analysis
aids, not proposed production early-stop rules.

From raise_env/:

  python scripts/analyze_diagnostics_run.py --run-dir results/highway_4h_...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Loose exploratory detector only — NOT a production early-stop threshold.
PLATEAU_SLOPE_THRESHOLD = 0.35
PLATEAU_WINDOW = 3
JUMP_AFTER_PLATEAU_SIGMAS = 1.0
MAX_CANDIDATES_IN_FULL_TABLES = 24


@dataclass
class CandLogs:
    candidate_id: str
    out_dir: str
    rollout: List[Dict[str, Any]]
    groundtruth: List[Dict[str, Any]]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _discover_candidates(run_dir: str) -> List[CandLogs]:
    found: List[CandLogs] = []
    for root, _dirs, files in os.walk(run_dir):
        if "diagnostics_rollout.jsonl" not in files:
            continue
        if "diagnostics_groundtruth.jsonl" not in files:
            continue
        rollout = _read_jsonl(os.path.join(root, "diagnostics_rollout.jsonl"))
        gt = _read_jsonl(os.path.join(root, "diagnostics_groundtruth.jsonl"))
        if not rollout or not gt:
            continue
        cid = (
            str(gt[0].get("candidate_id") or rollout[0].get("candidate_id") or "")
            or os.path.basename(root)
        )
        found.append(
            CandLogs(
                candidate_id=cid,
                out_dir=root,
                rollout=sorted(rollout, key=lambda r: int(r.get("steps") or 0)),
                groundtruth=sorted(gt, key=lambda r: int(r.get("steps") or 0)),
            )
        )
    found.sort(key=lambda c: c.candidate_id)
    return found


def _nearest_rollout(rollout: Sequence[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    best = rollout[0]
    best_d = abs(int(best.get("steps") or 0) - steps)
    for row in rollout[1:]:
        d = abs(int(row.get("steps") or 0) - steps)
        if d < best_d:
            best, best_d = row, d
    return best


def _series(rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            out.append((int(r.get("steps") or 0), float(v)))
        except (TypeError, ValueError):
            continue
    return out


def _trailing_normalized_slope(
    series: Sequence[Tuple[int, float]],
    *,
    end_idx: int,
    window: int,
) -> Optional[float]:
    """Slope over trailing window, divided by window std (exploratory)."""
    if end_idx < 0 or end_idx >= len(series):
        return None
    start = max(0, end_idx - window + 1)
    chunk = series[start : end_idx + 1]
    if len(chunk) < 2:
        return None
    xs = [float(p[0]) for p in chunk]
    ys = [float(p[1]) for p in chunk]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = cov / var_x
    # std of y in window
    var_y = sum((y - y_mean) ** 2 for y in ys) / max(1, len(ys) - 1)
    std = math.sqrt(max(var_y, 0.0))
    # Normalize by std and by typical step span so units are comparable.
    span = max(xs) - min(xs)
    if span <= 0:
        return 0.0
    # dimensionless-ish: (Δy per step) * span / std ≈ Δy/std over window
    if std < 1e-12:
        return 0.0 if abs(slope) < 1e-12 else float("inf")
    return abs(slope * span / std)


def _plateau_step(
    series: Sequence[Tuple[int, float]],
    *,
    threshold: float = PLATEAU_SLOPE_THRESHOLD,
    window: int = PLATEAU_WINDOW,
) -> Optional[int]:
    """First step where trailing normalized slope drops below threshold."""
    if len(series) < window:
        return None
    for i in range(window - 1, len(series)):
        ns = _trailing_normalized_slope(series, end_idx=i, window=window)
        if ns is not None and ns < threshold:
            return int(series[i][0])
    return None


def _std(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _jump_after_plateau(
    series: Sequence[Tuple[int, float]],
    plateau_step: Optional[int],
    *,
    n_sigma: float = JUMP_AFTER_PLATEAU_SIGMAS,
) -> Optional[Dict[str, Any]]:
    if plateau_step is None or len(series) < 3:
        return None
    pre = [v for s, v in series if s <= plateau_step]
    post = [(s, v) for s, v in series if s > plateau_step]
    if len(pre) < 2 or not post:
        return None
    hist_std = _std(pre)
    if hist_std < 1e-12:
        # flat history: any non-trivial absolute jump counts
        base = pre[-1]
        for s, v in post:
            if abs(v - base) > 1e-3:
                return {
                    "plateau_step": plateau_step,
                    "jump_step": s,
                    "jump_value": v,
                    "pre_std": hist_std,
                    "delta": v - base,
                }
        return None
    base = pre[-1]
    for s, v in post:
        if abs(v - base) > n_sigma * hist_std:
            return {
                "plateau_step": plateau_step,
                "jump_step": s,
                "jump_value": v,
                "pre_std": hist_std,
                "delta": v - base,
            }
    return None


def _divergence(
    reward_plateau: Optional[int],
    sr_plateau: Optional[int],
    reward_series: Sequence[Tuple[int, float]],
    sr_series: Sequence[Tuple[int, float]],
) -> Optional[str]:
    """Flag when one curve plateaus while the other is still moving."""
    if reward_plateau is None and sr_plateau is None:
        return None
    if reward_plateau is not None and sr_plateau is None:
        # reward plateaued; SR never plateaued — check SR still moves after
        after = [v for s, v in sr_series if s > reward_plateau]
        if len(after) >= 2 and _std(after) > 1e-6:
            return "reward_plateau_sr_still_moving"
        return None
    if sr_plateau is not None and reward_plateau is None:
        after = [v for s, v in reward_series if s > sr_plateau]
        if len(after) >= 2 and _std(after) > 1e-6:
            return "sr_plateau_reward_still_moving"
        return None
    assert reward_plateau is not None and sr_plateau is not None
    # Both plateaued but at very different steps ( > 30% of max step span)
    max_step = max(
        (reward_series[-1][0] if reward_series else 0),
        (sr_series[-1][0] if sr_series else 0),
        1,
    )
    gap = abs(reward_plateau - sr_plateau)
    if gap > 0.3 * max_step:
        if reward_plateau < sr_plateau:
            return "reward_plateaus_earlier_than_sr"
        return "sr_plateaus_earlier_than_reward"
    return None


def _fmt_table_row(cols: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cols) + " |"


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def analyze(run_dir: str) -> str:
    cands = _discover_candidates(run_dir)
    lines: List[str] = []
    lines.append("# DIAGNOSTICS_RUN_REPORT")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- candidates_with_both_logs: **{len(cands)}**")
    lines.append(
        f"- plateau detector (exploratory only, NOT a production threshold): "
        f"trailing window={PLATEAU_WINDOW}, "
        f"normalized_slope < {PLATEAU_SLOPE_THRESHOLD}"
    )
    lines.append(
        f"- jump-after-plateau flag: > {JUMP_AFTER_PLATEAU_SIGMAS}σ of "
        f"that candidate's own pre-plateau history"
    )
    lines.append("")

    if not cands:
        lines.append("No candidates with both `diagnostics_rollout.jsonl` and "
                      "`diagnostics_groundtruth.jsonl` were found.")
        lines.append("")
        lines.append("## Open questions")
        lines.append("")
        lines.append(
            "(a) Insufficient data — no paired diagnostic logs in this run."
        )
        lines.append(
            "(b) Insufficient data — no paired diagnostic logs in this run."
        )
        return "\n".join(lines) + "\n"

    diffs: List[float] = []
    jump_cases: List[Tuple[CandLogs, str, Dict[str, Any]]] = []
    diverge_cases: List[Tuple[CandLogs, str]] = []
    per_cand_rows: List[str] = []

    show = cands
    truncated = False
    if len(cands) > MAX_CANDIDATES_IN_FULL_TABLES:
        show = cands[:MAX_CANDIDATES_IN_FULL_TABLES]
        truncated = True

    lines.append("## Per-candidate side-by-side (reward proxy vs ground-truth)")
    lines.append("")
    if truncated:
        lines.append(
            f"Showing {len(show)} of {len(cands)} candidates "
            f"(cap={MAX_CANDIDATES_IN_FULL_TABLES})."
        )
        lines.append("")

    for cand in cands:
        rew = _series(cand.rollout, "ep_rew_mean")
        # Prefer groundtruth SR series for SR plateau; also keep mean_speed.
        sr = _series(cand.groundtruth, "SR")
        r_plat = _plateau_step(rew)
        s_plat = _plateau_step(sr)
        if r_plat is not None and s_plat is not None:
            diffs.append(float(abs(r_plat - s_plat)))
        elif r_plat is not None or s_plat is not None:
            # one-sided: use max step as proxy distance for summary? skip abs diff
            pass

        for label, series, plat in (
            ("reward", rew, r_plat),
            ("SR", sr, s_plat),
        ):
            jump = _jump_after_plateau(series, plat)
            if jump:
                jump_cases.append((cand, label, jump))

        div = _divergence(r_plat, s_plat, rew, sr)
        if div:
            diverge_cases.append((cand, div))

        if cand in show:
            per_cand_rows.append(f"### `{_md_escape(cand.candidate_id)}`")
            per_cand_rows.append("")
            per_cand_rows.append(
                f"- reward_plateau_step: `{r_plat}`"
            )
            per_cand_rows.append(f"- sr_plateau_step: `{s_plat}`")
            if r_plat is not None and s_plat is not None:
                per_cand_rows.append(
                    f"- |reward_plateau_step − sr_plateau_step|: "
                    f"**{abs(r_plat - s_plat)}**"
                )
            per_cand_rows.append("")
            per_cand_rows.append(
                _fmt_table_row(
                    [
                        "steps",
                        "ep_rew_mean",
                        "explained_variance",
                        "SR",
                        "CR",
                        "mean_speed",
                    ]
                )
            )
            per_cand_rows.append(
                _fmt_table_row(["---", "---", "---", "---", "---", "---"])
            )
            for gt in cand.groundtruth:
                steps = int(gt.get("steps") or 0)
                near = _nearest_rollout(cand.rollout, steps)
                per_cand_rows.append(
                    _fmt_table_row(
                        [
                            steps,
                            near.get("ep_rew_mean"),
                            near.get("explained_variance"),
                            gt.get("SR"),
                            gt.get("CR"),
                            gt.get("mean_speed"),
                        ]
                    )
                )
            per_cand_rows.append("")
            # Full sequences for inspection
            per_cand_rows.append(
                "<details><summary>full rollout ep_rew_mean sequence</summary>"
            )
            per_cand_rows.append("")
            per_cand_rows.append("```")
            for s, v in rew:
                per_cand_rows.append(f"  steps={s}  ep_rew_mean={v}")
            per_cand_rows.append("```")
            per_cand_rows.append("")
            per_cand_rows.append("</details>")
            per_cand_rows.append("")
            per_cand_rows.append(
                "<details><summary>full groundtruth SR sequence</summary>"
            )
            per_cand_rows.append("")
            per_cand_rows.append("```")
            for s, v in sr:
                per_cand_rows.append(f"  steps={s}  SR={v}")
            per_cand_rows.append("```")
            per_cand_rows.append("")
            per_cand_rows.append("</details>")
            per_cand_rows.append("")

    # Summary stats
    lines.append("## Summary: plateau step gap |reward − SR|")
    lines.append("")
    if diffs:
        diffs_sorted = sorted(diffs)
        mean_d = sum(diffs) / len(diffs)
        mid = diffs_sorted[len(diffs_sorted) // 2]
        lines.append(f"- n_paired_plateaus: {len(diffs)} / {len(cands)}")
        lines.append(f"- mean |Δsteps|: **{mean_d:.1f}**")
        lines.append(f"- median |Δsteps|: **{mid:.1f}**")
        lines.append(f"- max |Δsteps|: **{max(diffs):.1f}**")
    else:
        lines.append(
            "- No candidate had *both* reward and SR plateau detections "
            "(loose exploratory detector); gap distribution unavailable."
        )
    lines.append("")

    lines.append("## Jump after apparent plateau")
    lines.append("")
    uniq_jump_cands = {c.candidate_id for c, _, _ in jump_cases}
    lines.append(
        f"- candidates showing jump-after-plateau: "
        f"**{len(uniq_jump_cands)} / {len(cands)}**"
    )
    lines.append("")
    if jump_cases:
        for cand, label, jump in jump_cases:
            lines.append(
                f"- `{cand.candidate_id}` ({label}): plateau@{jump['plateau_step']}, "
                f"jump@{jump['jump_step']} value={jump['jump_value']:.4f} "
                f"Δ={jump['delta']:.4f} (pre_std={jump['pre_std']:.4f})"
            )
            series = _series(
                cand.rollout if label == "reward" else cand.groundtruth,
                "ep_rew_mean" if label == "reward" else "SR",
            )
            lines.append("  sequence:")
            for s, v in series:
                lines.append(f"    steps={s}  {label}={v}")
        lines.append("")
    else:
        lines.append("- None observed under the loose detector.")
        lines.append("")

    lines.append("## Reward ↔ SR divergence")
    lines.append("")
    lines.append(
        f"- candidates with divergence: **{len(diverge_cases)} / {len(cands)}**"
    )
    lines.append("")
    if diverge_cases:
        for cand, kind in diverge_cases:
            lines.append(f"- `{cand.candidate_id}`: {kind}")
            rew = _series(cand.rollout, "ep_rew_mean")
            sr = _series(cand.groundtruth, "SR")
            lines.append("  reward sequence:")
            for s, v in rew:
                lines.append(f"    steps={s}  ep_rew_mean={v}")
            lines.append("  SR sequence:")
            for s, v in sr:
                lines.append(f"    steps={s}  SR={v}")
        lines.append("")
    else:
        lines.append("- None observed under the loose detector.")
        lines.append("")

    lines.extend(per_cand_rows)

    # Open questions — stick to numbers
    lines.append("## Open questions (data only)")
    lines.append("")
    n = len(cands)
    n_div = len(diverge_cases)
    n_jump = len(uniq_jump_cands)
    lines.append(
        "(a) **Does reward-curve plateau reliably coincide with SR plateau, "
        "or do they diverge meaningfully?**"
    )
    if n < 5:
        lines.append(
            f"    Not enough candidates (n={n}) for a confident answer. "
            f"Divergence count under the exploratory detector: "
            f"{n_div}/{n}."
        )
    elif diffs:
        mean_d = sum(diffs) / len(diffs)
        lines.append(
            f"    Among {len(diffs)} candidates with both plateaus detected, "
            f"mean |Δsteps|={mean_d:.1f}, median={sorted(diffs)[len(diffs)//2]:.1f}, "
            f"max={max(diffs):.1f}. Divergence flags: {n_div}/{n}."
        )
        if n_div == 0 and mean_d < (0.15 * max(1, max(diffs) if diffs else 1)):
            lines.append(
                "    Observed numbers are consistent with coincidence in this run; "
                "they do not by themselves prove reliability beyond this sample."
            )
        elif n_div / max(1, n) >= 0.25 or (diffs and max(diffs) > 0):
            lines.append(
                "    Observed divergence/gap counts indicate the two signals "
                "are not identical in this run; see per-candidate tables."
            )
    else:
        lines.append(
            f"    Plateau detector did not fire on both curves for enough "
            f"candidates (paired={len(diffs)}, n={n}); divergence flags={n_div}/{n}. "
            f"Cannot confidently answer from this run alone."
        )
    lines.append("")
    lines.append(
        "(b) **Does 'jump after plateau' happen often enough in highway-fast-v0 "
        "to be a real design concern, or is it rare/absent?**"
    )
    if n < 5:
        lines.append(
            f"    Not enough candidates (n={n}) for a confident answer. "
            f"Jump-after-plateau count: {n_jump}/{n}."
        )
    else:
        rate = n_jump / n
        lines.append(
            f"    Observed jump-after-plateau: {n_jump}/{n} "
            f"({100.0 * rate:.1f}%) under the exploratory detector."
        )
        if n_jump == 0:
            lines.append(
                "    Absent in this run under the stated detector; that does not "
                "rule it out at larger sample sizes."
            )
        elif rate < 0.15:
            lines.append(
                "    Rare in this sample; whether that is 'often enough' for "
                "design concern is outside what these counts alone decide."
            )
        else:
            lines.append(
                "    Present at a non-trivial rate in this sample; see flagged "
                "sequences above."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="results/highway_4h_* directory from a diagnostics-enabled run",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Markdown report path (default: <run-dir>/DIAGNOSTICS_RUN_REPORT.md)",
    )
    args = parser.parse_args()
    run_dir = os.path.abspath(str(args.run_dir))
    if not os.path.isdir(run_dir):
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 2
    report = analyze(run_dir)
    out = str(args.out).strip() or os.path.join(run_dir, "DIAGNOSTICS_RUN_REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"\nWrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
