"""Print the five Claude verification answers from the live run."""

from __future__ import annotations

import ast
import difflib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RUN = ROOT / "results" / "_reward_components_live_verify"
SNAP = RUN / "_sec6_snapshots"


def walk(obj: Any, out: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "candidate_id" in obj and ("code" in obj or "origin" in obj):
            cid = str(obj["candidate_id"])
            prev = out.get(cid, {})
            merged = dict(prev)
            for k, v in obj.items():
                if v in (None, "", [], {}):
                    continue
                merged[k] = v
            md = dict(prev.get("metadata") or {})
            md.update(obj.get("metadata") or {})
            merged["metadata"] = md
            out[cid] = merged
        for v in obj.values():
            walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out)


def load_all() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    paths: List[Path] = []
    paths.extend(sorted(SNAP.glob("*.json")))
    paths.append(RUN / "closed_loop" / "checkpoint.json")
    paths.append(RUN / "checkpoint.json")
    paths.extend(sorted((RUN / "closed_loop").glob("pop_epoch_*.json")))
    for path in paths:
        if path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")), out)
    rej = RUN / "closed_loop" / "stage1_rejections.jsonl"
    if rej.is_file():
        for line in rej.read_text(encoding="utf-8").splitlines():
            if line.strip():
                walk(json.loads(line), out)
    epochs = RUN / "closed_loop" / "epochs.jsonl"
    if epochs.is_file():
        for line in epochs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                walk(json.loads(line), out)
    return out


def classify_origin(c: Dict[str, Any]) -> str:
    md = c.get("metadata") or {}
    for key in ("origin", "operator", "source", "generation_origin"):
        for blob in (c, md):
            if isinstance(blob, dict) and blob.get(key) not in (None, ""):
                return str(blob[key])
    cid = str(c.get("candidate_id") or "")
    if cid.startswith("see_"):
        return "INFERRED_from_id_prefix:seed_like"
    if cid.startswith("ini_"):
        return "INFERRED_from_id_prefix:initial"
    if cid.startswith("mut_"):
        return "INFERRED_from_id_prefix:mutation"
    if cid.startswith("cro_"):
        return "INFERRED_from_id_prefix:crossover"
    return "MISSING"


def extract_fn(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            return seg or ""
    raise SystemExit(f"function {name} not found")


def main() -> None:
    cands = load_all()
    print("=== CANDIDATE ORIGINS (all seen) ===")
    seedish = []
    llmish = []
    unknown = []
    for cid in sorted(cands):
        c = cands[cid]
        origin = classify_origin(c)
        print(f"{cid}\torigin={origin!r}\ttop_level_origin={c.get('origin')!r}")
        o = str(c.get("origin") or "").lower()
        if "seed" in o or "fallback" in o or cid.startswith("see_"):
            seedish.append(cid)
        elif o or cid.startswith(("ini_", "mut_", "cro_")):
            llmish.append(cid)
        else:
            unknown.append(cid)

    print("\n=== Q1 cro_0017 / mut_0020 ===")
    for cid in ("cro_0017", "mut_0020", "cro_0017_d3"):
        c = cands.get(cid)
        if not c:
            print(cid, "NOT_FOUND")
            continue
        md = c.get("metadata") or {}
        print(f"\n--- {cid} ---")
        print("top_level_origin:", repr(c.get("origin")))
        print("metadata.origin:", repr(md.get("origin")))
        interesting = {
            k: md.get(k)
            for k in sorted(md)
            if any(
                s in k.lower()
                for s in ("origin", "seed", "fallback", "operator", "source", "phase", "llm")
            )
        }
        print("metadata_interesting:", json.dumps(interesting, ensure_ascii=False, indent=2)[:2000])
        print("parent_ids:", c.get("parent_ids"))

    print("\n=== Q2 mut_0020 llm_diagnosis RAW ===")
    diag = (cands.get("mut_0020") or {}).get("metadata", {}).get("llm_diagnosis")
    print(diag if diag is not None else "MISSING")

    print("\n=== Q3 cro_0017 code ===")
    print((cands.get("cro_0017") or {}).get("code") or "MISSING")
    print("\n=== Q3 cro_0017_d3 code ===")
    print((cands.get("cro_0017_d3") or {}).get("code") or "MISSING")
    a = ((cands.get("cro_0017") or {}).get("code") or "").splitlines()
    b = ((cands.get("cro_0017_d3") or {}).get("code") or "").splitlines()
    print("\n=== Q3 unified diff ===")
    print(
        "\n".join(
            difflib.unified_diff(a, b, fromfile="cro_0017", tofile="cro_0017_d3", lineterm="")
        )
    )

    print("\n=== Q4 _feasibility_lines identity ===")
    work = (ROOT / "raise_core" / "raise_loop" / "proxy_feedback.py").read_text(encoding="utf-8")
    head = subprocess.check_output(
        ["git", "show", "HEAD:raise_env/raise_core/raise_loop/proxy_feedback.py"],
        cwd=str(REPO),
        text=True,
    )
    fw = extract_fn(work, "_feasibility_lines")
    fh = extract_fn(head, "_feasibility_lines")
    print("IDENTICAL=", fw == fh)
    print("--- function body (work) ---")
    print(fw)
    # Narrow git diff: only lines of the function after the inserted helper
    full = subprocess.check_output(
        ["git", "diff", "HEAD", "-U3", "--", "raise_env/raise_core/raise_loop/proxy_feedback.py"],
        cwd=str(REPO),
        text=True,
    )
    print("--- raw git hunk that ends at _feasibility_lines header (no body lines changed) ---")
    # Show that no '-'/'+' lines alter the function body: body appears only as context
    in_fn = False
    for line in full.splitlines():
        if line.startswith(" def _feasibility_lines") or line.startswith("def _feasibility_lines"):
            in_fn = True
            print(line)
            continue
        if in_fn:
            if line.startswith("diff ") or (line.startswith("@@") and "_feasibility" not in line):
                break
            if line.startswith("@@"):
                print(line)
                continue
            # stop at next top-level def that is not part of this function
            if line.startswith("def ") and "_feasibility" not in line:
                break
            if line.startswith("+def ") or line.startswith("-def "):
                break
            print(line)

    print("\n=== Q5 seed vs LLM counts ===")
    print("seedish_or_see=", seedish)
    print("llmish_or_ini_mut_cro=", llmish)
    print("unknown=", unknown)
    print("n_seedish=", len(seedish), "n_llmish=", len(llmish), "n_unknown=", len(unknown), "n_total=", len(cands))

    # Gen0 from log / epochs if available
    report = RUN / "closed_loop" / "REPORT.txt"
    if report.is_file():
        print("\n--- REPORT.txt (first 80 lines) ---")
        print("\n".join(report.read_text(encoding="utf-8").splitlines()[:80]))


if __name__ == "__main__":
    main()
