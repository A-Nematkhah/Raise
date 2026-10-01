"""Locate cro_0014 source and compare to mut_0020 for faithfulness correction."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "_reward_components_live_verify"
OUT = ROOT / "results" / "_reward_components_verification" / "sec6_cro0014_mut0020.txt"


def walk_find(obj: Any, cid: str) -> Optional[Dict[str, Any]]:
    found = None
    if isinstance(obj, dict):
        if obj.get("candidate_id") == cid and obj.get("code"):
            return dict(obj)
        for v in obj.values():
            hit = walk_find(v, cid)
            if hit:
                found = hit
    elif isinstance(obj, list):
        for v in obj:
            hit = walk_find(v, cid)
            if hit:
                found = hit
    return found


def load_candidate(cid: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_src = ""
    for path in RUN.rglob("*"):
        if path.suffix not in (".json", ".jsonl"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if cid not in text:
            continue
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if cid not in line or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hit = walk_find(row, cid)
                if hit and hit.get("code"):
                    best = hit
                    best_src = str(path)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            hit = walk_find(data, cid)
            if hit and hit.get("code"):
                best = hit
                best_src = str(path)
    if best:
        best["_src"] = best_src
    return best


def component_names(code: str) -> list[str]:
    match = re.search(r"return\s+[^,\n]+,\s*\{(.*?)\}", code, re.S)
    if not match:
        # also catch multi-return sites
        names = re.findall(
            r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:",
            code,
        )
        # filter common non-component locals poorly; prefer dict returns
        return sorted(set(names))
    return re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:", match.group(1))


def all_component_keys_in_returns(code: str) -> list[str]:
    keys = set()
    for m in re.finditer(r"return\s*\([^)]*,\s*\{([^}]*)\}", code, re.S):
        keys.update(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:", m.group(1)))
    # also single-line dict literals after return total,
    for m in re.finditer(r"return\s+[^,\n]+,\s*\{([^}]*)\}", code):
        keys.update(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:", m.group(1)))
    return sorted(keys)


def coef_mentions(code: str, name: str) -> list[str]:
    lines = []
    for line in code.splitlines():
        if name.lower() in line.lower() or "speed" in line.lower() and "coef" in line.lower():
            if any(x in line for x in ("speed", "SPEED", "proximity", "PROXIMITY", "collision", "COLLISION")):
                lines.append(line.strip())
    return lines


def main() -> None:
    parent = load_candidate("cro_0014")
    child = load_candidate("mut_0020")
    lines: list[str] = []
    lines.append("=== parent cro_0014 ===")
    if not parent:
        lines.append("NOT_FOUND in results/_reward_components_live_verify")
    else:
        lines.append(f"src={parent.get('_src')}")
        lines.append(f"origin={parent.get('origin')!r} parent_ids={parent.get('parent_ids')}")
        lines.append(f"component_keys={all_component_keys_in_returns(parent['code'])}")
        lines.append("--- CODE ---")
        lines.append(parent["code"])

    lines.append("")
    lines.append("=== child mut_0020 ===")
    if not child:
        lines.append("NOT_FOUND")
    else:
        lines.append(f"src={child.get('_src')}")
        lines.append(f"origin={child.get('origin')!r} parent_ids={child.get('parent_ids')}")
        md = child.get("metadata") or {}
        lines.append(f"llm_diagnosis={md.get('llm_diagnosis')!r}")
        lines.append(f"component_keys={all_component_keys_in_returns(child['code'])}")
        lines.append("--- CODE ---")
        lines.append(child["code"])

    if parent and child:
        lines.append("")
        lines.append("=== VERIFY parent_ids link ===")
        lines.append(f"mut_0020.parent_ids={child.get('parent_ids')}")
        lines.append(
            f"link_ok={child.get('parent_ids') == ['cro_0014'] or child.get('parent_ids') == ('cro_0014',)}"
        )
        lines.append("")
        lines.append("=== unified diff cro_0014 -> mut_0020 ===")
        diff = difflib.unified_diff(
            parent["code"].splitlines(),
            child["code"].splitlines(),
            fromfile="cro_0014",
            tofile="mut_0020",
            lineterm="",
        )
        lines.extend(diff)

        pkeys = set(all_component_keys_in_returns(parent["code"]))
        ckeys = set(all_component_keys_in_returns(child["code"]))
        lines.append("")
        lines.append("=== faithfulness checks ===")
        lines.append(f"parent_has_proximity_penalty={'proximity_penalty' in pkeys}")
        lines.append(f"parent_keys={sorted(pkeys)}")
        lines.append(f"child_keys={sorted(ckeys)}")
        lines.append(f"new_in_child={sorted(ckeys - pkeys)}")
        lines.append(f"removed_in_child={sorted(pkeys - ckeys)}")
        # speed coef heuristics
        lines.append("parent_speed_lines:")
        lines.extend("  " + x for x in coef_mentions(parent["code"], "speed"))
        lines.append("child_speed_lines:")
        lines.extend("  " + x for x in coef_mentions(child["code"], "speed"))

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(lines[:80]))
    if parent is None:
        print("\nPARENT_MISSING — need alternate recovery")


if __name__ == "__main__":
    main()
