"""Answers 4–5 evidence: feasibility git proof + full origin census."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RUN = ROOT / "results" / "_reward_components_live_verify"


def feasibility_git_proof() -> None:
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "HEAD",
            "-U30",
            "--",
            "raise_env/raise_core/raise_loop/proxy_feedback.py",
        ],
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
    )
    lines = diff.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "_feasibility_lines" in line and (
            line.startswith(" ")
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith("@@")
        ):
            # Prefer the actual def line
            if "def _feasibility_lines" in line:
                start = i
                break
    print("=== Q4: git diff chunk from def _feasibility_lines onward ===")
    if start is None:
        print("NO_DEF_LINE_IN_DIFF (function not present in diff at all)")
        # Still print whether any feasibility body lines are +/ -
        body_hits = [
            l
            for l in lines
            if ("Feasibility this run" in l or "Reachable action" in l or "pareto_v_floor" in l)
            and (l.startswith("+") or l.startswith("-"))
        ]
        print("changed_feasibility_body_lines=", body_hits)
        return
    chunk = []
    for line in lines[start:]:
        if chunk and line.startswith("@@"):
            break
        if (
            chunk
            and (line.startswith(" def ") or line.startswith("+def ") or line.startswith("-def "))
            and "_feasibility" not in line
        ):
            break
        chunk.append(line)
    for line in chunk:
        print(line)
    changed = [l for l in chunk if l.startswith("+") or l.startswith("-")]
    print("CHANGED_LINES_COUNT=", len(changed))
    for line in changed:
        print("CHANGED:", line)

    # Also: git diff with function context using pickaxe? Show that working tree
    # function bytes match HEAD via git hash-object of extracted ranges is hard;
    # instead use `git diff -U0` and assert zero +/- inside feasibility.
    u0 = subprocess.check_output(
        [
            "git",
            "diff",
            "HEAD",
            "-U0",
            "--",
            "raise_env/raise_core/raise_loop/proxy_feedback.py",
        ],
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
    )
    print("\n=== Q4: all +/- lines in full proxy_feedback.py diff ===")
    for line in u0.splitlines():
        if line.startswith("+") or line.startswith("-"):
            if line.startswith("+++") or line.startswith("---"):
                continue
            print(line)


def ingest(obj: Any, out: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        cid = obj.get("candidate_id")
        if cid:
            cid = str(cid)
            prev = out.get(cid, {})
            merged = dict(prev)
            for k, v in obj.items():
                if v in (None, "", [], {}):
                    continue
                if k == "metadata" and isinstance(v, dict):
                    md = dict(prev.get("metadata") or {})
                    md.update(v)
                    merged["metadata"] = md
                else:
                    merged[k] = v
            out[cid] = merged
        for v in obj.values():
            ingest(v, out)
    elif isinstance(obj, list):
        for v in obj:
            ingest(v, out)


def census() -> None:
    out: Dict[str, Dict[str, Any]] = {}
    for path in RUN.rglob("*"):
        if path.suffix not in (".json", ".jsonl"):
            continue
        text = path.read_text(encoding="utf-8")
        if "candidate_id" not in text:
            continue
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    ingest(json.loads(line), out)
                except json.JSONDecodeError:
                    pass
        else:
            try:
                ingest(json.loads(text), out)
            except json.JSONDecodeError:
                pass

    print("\n=== Q5: every candidate_id with origin ===")
    ctr: Counter[str] = Counter()
    for cid in sorted(out):
        c = out[cid]
        origin = c.get("origin")
        if not origin:
            origin = (c.get("metadata") or {}).get("origin")
        if not origin:
            # id-prefix inference only as last resort, flagged
            if cid.startswith("see_"):
                origin = "(missing_field; id_prefix=see_ → seed_fallback family)"
            elif cid.startswith("ini_"):
                origin = "(missing_field; id_prefix=ini_ → initial)"
            elif cid.startswith("mut_"):
                origin = "(missing_field; id_prefix=mut_ → mutation)"
            elif cid.startswith("cro_"):
                origin = "(missing_field; id_prefix=cro_ → crossover)"
            else:
                origin = "(missing)"
        print(f"{cid}\torigin={origin}")
        bucket = str(origin)
        if "seed" in bucket or bucket.startswith("(missing_field; id_prefix=see_"):
            ctr["seed_fallback_family"] += 1
        elif bucket in ("initial",) or bucket.startswith("(missing_field; id_prefix=ini_"):
            ctr["initial"] += 1
        elif "mutation" in bucket or bucket.startswith("(missing_field; id_prefix=mut_"):
            ctr["mutation"] += 1
        elif "crossover" in bucket or bucket.startswith("(missing_field; id_prefix=cro_"):
            ctr["crossover"] += 1
        elif "d3" in bucket or "refin" in bucket:
            ctr["d3_or_refinement"] += 1
        else:
            ctr[f"other:{bucket}"] += 1
    print("BUCKETS", dict(ctr))
    print("TOTAL_DISTINCT_IDS", len(out))

    # Gen0 specifically from live log
    log = ROOT / "results" / "_reward_components_verification" / "live_run.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        print("\n=== Gen0 log lines (scoring / fallback) ===")
        for line in text.splitlines():
            if any(
                s in line
                for s in (
                    "scoring candidate",
                    "using fallback",
                    "Gen0 batch produced",
                    "regenerating",
                    "slots filled",
                    "sandbox reject phase=gen0",
                )
            ):
                print(line)


if __name__ == "__main__":
    feasibility_git_proof()
    census()
