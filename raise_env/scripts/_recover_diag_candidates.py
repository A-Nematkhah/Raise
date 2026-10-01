"""Recover Stage-2 diagnostic candidate codes via seed-variant catalog + known archives."""

from __future__ import annotations

import ast
import json
import os
import re
import sys

from domains.highway.prompts import D5_SEED_FUNCTION
from raise_core.llm import SeedVariantLLMClient, strip_redundant_scalar_subscripts
from raise_core.surrogate.features import code_sha256

WANT = [
    "ini_0000",
    "ini_0001",
    "ini_0002",
    "ini_0003",
    "ini_0004",
    "ini_0005",
    "mut_0009",
    "mut_0014",
    "mut_0018",
    "mut_0019",
    "mut_0020",
    "cro_0006",
    "cro_0007",
    "cro_0011",
    "cro_0012",
]

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract(raw: str) -> str:
    m = FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _canon(code: str) -> str:
    text = strip_redundant_scalar_subscripts(code)
    try:
        return ast.unparse(ast.parse(text))
    except Exception:
        return text


def build_catalog(n_max: int = 5000) -> dict[str, str]:
    client = SeedVariantLLMClient(base_code=D5_SEED_FUNCTION)
    catalog: dict[str, str] = {}
    for src in (D5_SEED_FUNCTION,):
        for variant in (src, _canon(src)):
            catalog[code_sha256(variant)] = variant
    for n in range(1, n_max + 1):
        raw = client._highway_variant(n)
        code = _extract(raw)
        for variant in (code, _canon(code)):
            if variant:
                catalog[code_sha256(variant)] = variant
    return catalog


def load_want_hashes(run_dir: str) -> dict[str, str]:
    path = os.path.join(run_dir, "surrogate_dataset", "features.jsonl")
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row.get("candidate_id")
            if cid in WANT:
                out[str(cid)] = str(row["code_hash"])
    return out


def load_known_codes(run_dir: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for rel in (
        "stage3/checkpoint.json",
        "closed_loop/checkpoint.json",
        "stage2_population.json",
        "stage1_population.json",
    ):
        path = os.path.join(run_dir, rel)
        if not os.path.isfile(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        stack = [data]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                cid = obj.get("candidate_id")
                code = obj.get("code")
                if cid in WANT and isinstance(code, str) and "compute_reward" in code:
                    found[str(cid)] = code
                stack.extend(obj.values())
            elif isinstance(obj, list):
                stack.extend(obj)
    return found


def main() -> int:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "results/highway_4h_20260926_211502"
    )
    want = load_want_hashes(run_dir)
    known = load_known_codes(run_dir)
    print(f"want hashes: {len(want)}; known archive codes: {len(known)}")

    catalog = build_catalog()
    print(f"catalog size: {len(catalog)}")

    # Sanity: known codes must be in catalog OR at least hash-stable.
    for cid, code in known.items():
        h = code_sha256(code)
        print(
            f"  known {cid}: hash_in_want={h == want.get(cid)} "
            f"in_catalog={h in catalog} "
            f"canon_in_catalog={code_sha256(_canon(code)) in catalog}"
        )
        # inject known into catalog always
        catalog[h] = code
        catalog[code_sha256(_canon(code))] = _canon(code)

    matched: dict[str, dict] = {}
    for cid, h in want.items():
        if cid in known:
            matched[cid] = {
                "candidate_id": cid,
                "code": known[cid],
                "code_hash": h,
                "source": "run_archive",
            }
        elif h in catalog:
            matched[cid] = {
                "candidate_id": cid,
                "code": catalog[h],
                "code_hash": h,
                "source": "seed_variant_catalog",
            }

    print(f"matched {len(matched)}/{len(want)}: {sorted(matched)}")
    missing = sorted(set(want) - set(matched))
    print(f"missing: {missing}")

    # Debug: show a few catalog entries near known float styles
    sample_n = 20
    client = SeedVariantLLMClient(base_code=D5_SEED_FUNCTION)
    raw = client._highway_variant(sample_n)
    code = _extract(raw)
    print("--- sample n=20 ---")
    print(repr(code[:120]))
    print("hash raw", code_sha256(code))
    print("hash canon", code_sha256(_canon(code)))
    if "mut_0018" in known:
        print("known mut_0018", repr(known["mut_0018"][:120]))
        print("known hash", code_sha256(known["mut_0018"]))

    out = os.path.join(run_dir, "recovered_stage2_diag_candidates.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(matched, fh, indent=2)
    print("wrote", out)
    return 0 if len(matched) == len(want) else 2


if __name__ == "__main__":
    raise SystemExit(main())
