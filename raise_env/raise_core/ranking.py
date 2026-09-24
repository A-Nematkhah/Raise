"""
Final Stage II/III candidate ranking.

Paper (Alg. 1 lines 20, 30): R2/R3 via LLM evaluation of multi-objective M(r).
Default mode is ``llm`` (seed / parse failure → multiobjective_lex).
``scalar`` (SR−CR−0.5·TR) remains available as an engineering opt-in.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from raise_core.explore import RewardCandidate
from raise_core.selection import candidate_nav_scalar

logger = logging.getLogger(__name__)


def rank_ids_by_scalar(candidates: Sequence[RewardCandidate]) -> List[str]:
    """Best-first by SR-CR-0.5·TR (engineering opt-in; not paper R2/R3)."""
    ordered = sorted(candidates, key=candidate_nav_scalar, reverse=True)
    return [c.candidate_id for c in ordered]


def pick_candidate_by_ranking(
    ranking: Dict[str, Any],
    pool: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """
    Return the first ranked id's match in ``pool`` (best-first).

    If several pool entries share the same ``candidate_id`` (legacy trained
    snapshots before unique snap ids), prefer the highest navigation scalar
    among that id so a later weak retrain cannot overwrite an earlier elite.
    """
    from collections import defaultdict

    from raise_core.selection import candidate_nav_scalar

    by_id: Dict[str, List[RewardCandidate]] = defaultdict(list)
    for c in pool:
        by_id[str(c.candidate_id)].append(c)
    for cid in ranking.get("ranking") or []:
        hits = by_id.get(str(cid)) or []
        if not hits:
            continue
        if len(hits) == 1:
            return hits[0]
        return max(hits, key=candidate_nav_scalar)
    return None


def _metrics_block(candidates: Sequence[RewardCandidate]) -> str:
    lines = []
    for c in candidates:
        md = (c.metadata or {}).get("last_metrics") or {}
        lines.append(
            f"- {c.candidate_id}: "
            f"SR={float(md.get('SR', md.get('sr', 0.0))):.4f}, "
            f"CR={float(md.get('CR', md.get('cr', 0.0))):.4f}, "
            f"TR={float(md.get('TR', md.get('tr', 0.0))):.4f}, "
            f"NT={float(md.get('NT', md.get('nt', 0.0))):.4f}, "
            f"PL={float(md.get('PL', md.get('pl', 0.0))):.4f}, "
            f"ITR={float(md.get('ITR', md.get('itr', 0.0))):.4f}, "
            f"SD={float(md.get('SD', md.get('sd', 0.0))):.4f}"
        )
    return "\n".join(lines)


def format_final_rank_prompt(candidates: Sequence[RewardCandidate]) -> str:
    ids = ", ".join(c.candidate_id for c in candidates)
    return (
        "You are ranking robot-navigation reward candidates after RL training "
        "(RAISE Algorithm 1 final ranking R2/R3).\n"
        "Using the multi-objective metrics below (higher SR/SD better; lower "
        "CR/TR/NT/PL/ITR better), produce a total order from best to worst.\n"
        f"Candidate ids: {ids}\n"
        "Metrics:\n"
        f"{_metrics_block(candidates)}\n"
        "Return ONLY a comma-separated list of all candidate ids best-first, "
        "no other text."
    )


def _parse_rank_response(text: str, valid_ids: Sequence[str]) -> Optional[List[str]]:
    valid = {str(i) for i in valid_ids}
    # Prefer comma / whitespace separated tokens that match ids.
    tokens = re.split(r"[\s,;|>]+", text.strip())
    ordered: List[str] = []
    for tok in tokens:
        t = tok.strip().strip("`\"'")
        if t in valid and t not in ordered:
            ordered.append(t)
    if len(ordered) == len(valid):
        return ordered
    # Fallback: find ids as substrings in order of appearance.
    ordered = []
    for match in re.finditer(
        r"(" + "|".join(re.escape(i) for i in sorted(valid, key=len, reverse=True)) + r")",
        text,
    ):
        tid = match.group(1)
        if tid not in ordered:
            ordered.append(tid)
    if len(ordered) == len(valid):
        return ordered
    return None


def multiobjective_lex_rank(candidates: Sequence[RewardCandidate]) -> List[str]:
    """
    Deterministic multi-metric order (stand-in when LLM rank is unavailable).

    Key: SR ↓CR ↓TR ↓NT ↓PL ↓ITR ↑SD — closer to Alg. 1 M(r) than SR-CR-0.5TR alone.
    """

    def key(c: RewardCandidate) -> tuple:
        md = (c.metadata or {}).get("last_metrics") or {}
        sr = float(md.get("SR", md.get("sr", 0.0)))
        cr = float(md.get("CR", md.get("cr", 0.0)))
        tr = float(md.get("TR", md.get("tr", 0.0)))
        nt = float(md.get("NT", md.get("nt", 0.0)))
        pl = float(md.get("PL", md.get("pl", 0.0)))
        itr = float(md.get("ITR", md.get("itr", 0.0)))
        sd = float(md.get("SD", md.get("sd", 0.0)))
        return (sr, -cr, -tr, -nt, -pl, -itr, sd)

    return [c.candidate_id for c in sorted(candidates, key=key, reverse=True)]


def produce_final_ranking(
    candidates: Sequence[RewardCandidate],
    *,
    mode: str = "llm",
    llm: Any = None,
) -> Dict[str, Any]:
    """
    Produce a best-first id list.

    mode:
      - ``llm``: Alg. 1 R2/R3; on failure or seed-like providers, fall back to
        multiobjective_lex_rank (not scalar) and record the fallback.
      - ``scalar``: SR-CR-0.5·TR engineering opt-in
    """
    mode_key = str(mode).strip().lower()
    ids = [c.candidate_id for c in candidates]
    if not candidates:
        return {"mode": mode_key, "ranking": [], "fallback": None}

    if mode_key == "scalar":
        return {
            "mode": "scalar",
            "ranking": rank_ids_by_scalar(candidates),
            "fallback": None,
        }

    if mode_key != "llm":
        raise ValueError(f"Unknown final_rank mode: {mode!r}")

    provider = getattr(llm, "provider", None) or getattr(llm, "name", "")
    provider_s = str(provider).lower()
    if llm is None or provider_s in ("seed", "scripted", ""):
        ranking = multiobjective_lex_rank(candidates)
        return {
            "mode": "llm",
            "ranking": ranking,
            "fallback": "multiobjective_lex",
            "note": "No LLM / seed provider — used multi-objective lexicographic rank",
        }

    prompt = format_final_rank_prompt(candidates)
    try:
        raw = llm.complete(prompt)
        parsed = _parse_rank_response(str(raw), ids)
        if parsed is not None:
            return {"mode": "llm", "ranking": parsed, "fallback": None, "raw": str(raw)[:2000]}
        logger.warning("LLM final-rank parse failed; falling back to multiobjective_lex")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM final-rank failed (%s); falling back to multiobjective_lex", exc)

    return {
        "mode": "llm",
        "ranking": multiobjective_lex_rank(candidates),
        "fallback": "multiobjective_lex",
    }
