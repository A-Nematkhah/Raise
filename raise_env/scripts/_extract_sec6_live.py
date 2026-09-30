"""Extract Section 6 evidence from a live highway closed-loop checkpoint."""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "_reward_components_live_verify" / "closed_loop"
OUT = ROOT / "results" / "_reward_components_verification" / "section6_live_extract.txt"

PRINCIPLE_PATTERNS = {
    "plateau/flat (principle 2)": re.compile(
        r"(nearly identical|flat|constant|plateau|not able to optimi[sz]e|"
        r"barely (changes|varies)|same value|no variation|unchanged)",
        re.I,
    ),
    "magnitude/rescale (principle 3)": re.compile(
        r"(rescale|dominat|magnitude|too large|overwhelm|comparable range|scale down|scale up)",
        re.I,
    ),
    "rewrite entire (principle 1)": re.compile(
        r"(rewrite the entire|rewrite entire|success rate .* near zero|survival .* near zero)",
        re.I,
    ),
}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_candidate(out: Dict[str, Dict[str, Any]], c: Dict[str, Any]) -> None:
    cid = c.get("candidate_id")
    if not cid or not c.get("code"):
        return
    prev = out.get(cid, {})
    merged = dict(prev)
    merged.update({k: v for k, v in c.items() if v not in (None, "", [], {})})
    md = dict(prev.get("metadata") or {})
    md.update(c.get("metadata") or {})
    merged["metadata"] = md
    out[cid] = merged


def _walk_candidates(obj: Any, out: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "candidate_id" in obj and "code" in obj:
            _merge_candidate(out, obj)
        for v in obj.values():
            _walk_candidates(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_candidates(v, out)


def _all_candidates(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if path.is_file():
            _walk_candidates(_load(path), out)
    return out


def _trends_from_diagnostics(run_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Per-candidate component rollout means from diagnostics_rollout.jsonl."""
    trends: Dict[str, Dict[str, List[float]]] = {}
    train = run_dir / "stage2_train"
    if not train.is_dir():
        return trends
    for diag in sorted(train.glob("*/diagnostics_rollout.jsonl")):
        for raw in diag.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            cid = row.get("candidate_id")
            means = row.get("reward_component_means") or {}
            if not cid or not isinstance(means, dict):
                continue
            per = trends.setdefault(cid, {})
            for name, value in means.items():
                per.setdefault(name, []).append(float(value))
    return trends


def _component_names_from_code(code: str) -> List[str]:
    match = re.search(r"return\s+[^,\n]+,\s*\{(.*?)\}", code, re.S)
    if not match:
        return []
    return re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:", match.group(1))


def main() -> int:
    ck_path = RUN / "checkpoint.json"
    if not ck_path.is_file():
        print(f"missing {ck_path}", file=sys.stderr)
        return 1
    ck = _load(ck_path)
    snapshots = [ck_path, RUN.parent / "checkpoint.json"]
    snapshots.extend(sorted((RUN.parent / "_sec6_snapshots").glob("*.json")))
    cands = _all_candidates(snapshots)
    diag_trends = _trends_from_diagnostics(RUN)
    lines: List[str] = []
    lines.append(f"checkpoint epoch={ck.get('epoch')} next_epoch={ck.get('next_epoch')} status={ck.get('status')}")
    lines.append(f"n_candidates_seen={len(cands)}")
    lines.append("")

    lines.append("=== 6.1 component names per candidate ===")
    for cid in sorted(cands):
        c = cands[cid]
        code = c.get("code") or ""
        names_code = _component_names_from_code(code)
        per = diag_trends.get(cid) or {}
        lines.append(
            f"{cid} parents={c.get('parent_ids')} code_components={names_code} "
            f"trained_component_keys={sorted(per.keys())}"
        )
        for name in sorted(per):
            vals = per[name]
            lines.append(
                f"    {name}: n={len(vals)} first={vals[0]:.3f} last={vals[-1]:.3f} "
                f"min={min(vals):.3f} max={max(vals):.3f}"
            )
    lines.append("")

    lines.append("=== 6.2 diagnoses naming a component + EUREKA principle ===")
    hits = 0
    for cid in sorted(cands):
        c = cands[cid]
        md = c.get("metadata") or {}
        diag = str(md.get("llm_diagnosis") or "").strip()
        if not diag:
            continue
        parent_ids = c.get("parent_ids") or []
        parent_names: List[str] = []
        for pid in parent_ids:
            p = cands.get(pid) or {}
            parent_names.extend(_component_names_from_code(p.get("code") or ""))
            pt = (p.get("metadata") or {}).get("reward_component_trends") or {}
            if isinstance(pt, dict):
                parent_names.extend(pt.keys())
        own = _component_names_from_code(c.get("code") or "")
        vocab = sorted(set(parent_names) | set(own))
        named = [n for n in vocab if re.search(rf"\b{re.escape(n)}\b", diag)]
        # Prose mentions ("collision penalty" for collision_penalty) count separately.
        prose = [
            n
            for n in vocab
            if n not in named
            and re.search(r"\b" + re.escape(n.replace("_", " ")) + r"\b", diag, re.I)
        ]
        principles = [lab for lab, pat in PRINCIPLE_PATTERNS.items() if pat.search(diag)]
        lines.append(
            f"--- {cid} parents={parent_ids} named_identifier={named} "
            f"named_in_prose={prose} principles={principles}"
        )
        lines.append(diag)
        lines.append("")
        if named and principles:
            hits += 1
    lines.append(f"diagnoses_with_named_component_and_principle={hits}")
    lines.append("")

    lines.append("=== 6.3 parent -> child code diffs (mutation/crossover) ===")
    for cid in sorted(cands):
        if not (cid.startswith("mut_") or cid.startswith("cro_")):
            continue
        c = cands[cid]
        for pid in c.get("parent_ids") or []:
            p = cands.get(pid)
            if not p or not p.get("code") or not c.get("code"):
                continue
            diff = difflib.unified_diff(
                p["code"].splitlines(),
                c["code"].splitlines(),
                fromfile=f"{pid}",
                tofile=f"{cid}",
                lineterm="",
            )
            lines.append(f"### {pid} -> {cid}")
            lines.extend(diff)
            lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines); hits={hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
