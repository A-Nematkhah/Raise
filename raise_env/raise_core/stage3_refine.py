"""Stage III multi-round LLM reward refinement (EvoNav-style G3).

After a long PPO train, each finalist is mutated from multi-objective holdout
metrics (SR/CR/TR/NT/PL/ITR/SD) and re-trained. ``refine_rounds=1`` is a no-op
(legacy single train); ``refine_rounds=3`` matches paper G3 fidelity rounds.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from raise_core import console
from raise_core.candidate import RewardCandidate
from raise_core.llm import LLMClient, estimate_groq_prompt_tokens, extract_python_code
from raise_core.selection import candidate_fitness

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("SR", "CR", "TR", "NT", "PL", "ITR", "SD", "fitness")


def _fmt_metric(metrics: dict, key: str) -> str:
    raw = metrics.get(key)
    if raw is None and key == "NT":
        raw = metrics.get("nav_time")
    if raw is None and key == "PL":
        raw = metrics.get("path_length")
    if raw is None and key == "ITR":
        raw = metrics.get("danger_freq")
    if raw is None and key == "SD":
        raw = metrics.get("min_dist_in_danger")
    try:
        return f"{float(raw):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def multi_objective_feedback(candidate: RewardCandidate) -> str:
    """Compact multi-objective block for Stage III reward mutation prompts."""
    md = candidate.metadata or {}
    metrics = dict(md.get("last_metrics") or md.get("holdout") or {})
    holdout = metrics.get("holdout")
    if isinstance(holdout, dict) and holdout:
        metrics = {**metrics, **holdout}
    lines = [
        f"Parent {candidate.candidate_id} Stage III holdout (multi-objective):",
        "  "
        + " ".join(f"{k}={_fmt_metric(metrics, k)}" for k in _METRIC_KEYS),
        "Prioritize improving ITR (danger_freq) and SD (min_dist_in_danger) "
        "without sacrificing SR or inflating CR/TR/NT/PL.",
        "Make a minimal, localized edit to compute_reward; return only code.",
    ]
    weaknesses = md.get("selection_weaknesses") or ()
    if weaknesses:
        lines.append("LLM ranking weaknesses: " + "; ".join(str(w) for w in weaknesses))
    return "\n".join(lines)


def _shrink_prompt_to_budget(
    prompt: str,
    *,
    system_prompt: str,
    max_input_tokens: int,
) -> Optional[str]:
    """Drop middle of an oversized refine prompt until it fits the input budget.

    Returns ``None`` when even a heavily truncated prompt still exceeds the
    margined input budget (caller should skip the LLM call).
    """
    text = str(prompt or "")
    limit = max(512, int(max_input_tokens))
    if estimate_groq_prompt_tokens(system_prompt, text) <= limit:
        return text
    # Keep head (instructions) and tail (code); carve the middle.
    while len(text) > 800 and estimate_groq_prompt_tokens(system_prompt, text) > limit:
        cut = max(200, len(text) // 5)
        head = text[: len(text) // 3]
        tail = text[-(len(text) - len(head) - cut) :]
        text = head + "\n\n# ... (truncated for token budget) ...\n\n" + tail
    if estimate_groq_prompt_tokens(system_prompt, text) > limit:
        return None
    return text


def propose_refined_candidates(
    trained: Sequence[RewardCandidate],
    *,
    llm: LLMClient,
    validator: Any,
    prompts: Any,
    refine_round: int,
    func_name: str = "compute_reward",
    max_input_tokens: Optional[int] = None,
) -> List[RewardCandidate]:
    """LLM-mutate each trained finalist; skip parents that fail validation."""
    out: List[RewardCandidate] = []
    system = str(getattr(llm, "system_prompt", "") or "")
    try:
        from raise_config import CONFIG

        input_cap = int(
            max_input_tokens
            if max_input_tokens is not None
            else CONFIG.selection.max_input_tokens
        )
    except Exception:  # noqa: BLE001
        input_cap = int(max_input_tokens or 4500)

    # Cap margined input so prompt + llm.max_tokens stay under Groq soft TPM.
    try:
        from raise_core.llm import groq_margined_input_budget

        out_reserve = int(getattr(llm, "max_tokens", None) or 2000)
        input_cap = min(input_cap, groq_margined_input_budget(out_reserve))
    except Exception:  # noqa: BLE001
        pass

    for parent in trained:
        feedback = multi_objective_feedback(parent)
        prompt = prompts.format_d2_mutation(
            parent.code,
            feedback,
            func_name=func_name,
        )
        prompt = _shrink_prompt_to_budget(
            prompt, system_prompt=system, max_input_tokens=input_cap
        )
        if prompt is None:
            console.warn(
                f"refine round {refine_round}: prompt over TPM budget for "
                f"{parent.candidate_id}; keeping parent",
                stage="Stage III",
            )
            out.append(parent)
            continue
        try:
            raw = llm.complete(prompt)
            code = extract_python_code(raw)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Stage III refine LLM failed for %s (round %d)",
                parent.candidate_id,
                refine_round,
            )
            console.warn(
                f"refine round {refine_round}: LLM failed for "
                f"{parent.candidate_id} ({type(exc).__name__}); keeping parent",
                stage="Stage III",
            )
            out.append(parent)
            continue
        reward_fn, error = validator.try_validate(code)
        new_id = f"{parent.candidate_id}_r{refine_round}"
        if reward_fn is None:
            console.warn(
                f"refine round {refine_round}: invalid code for "
                f"{parent.candidate_id} ({error}); keeping parent",
                stage="Stage III",
            )
            out.append(parent)
            continue
        out.append(
            RewardCandidate(
                candidate_id=new_id,
                code=code,
                reward_fn=reward_fn,
                score=parent.score,
                valid=True,
                origin="stage3_refine",
                parent_ids=(parent.candidate_id,),
                metadata={
                    "stage3_refine_round": int(refine_round),
                    "refined_from": str(parent.candidate_id),
                    "proxy_feedback": feedback,
                },
            )
        )
        console.status(
            f"refine round {refine_round}: {parent.candidate_id} -> {new_id}",
            stage="Stage III",
        )
    return out


def merge_refine_populations(
    previous: Sequence[RewardCandidate],
    refined: Sequence[RewardCandidate],
) -> List[RewardCandidate]:
    """Keep the better of parent vs refined child by Stage III fitness."""
    by_parent = {str(c.candidate_id): c for c in previous}
    kept: List[RewardCandidate] = []
    claimed_parents: set[str] = set()
    for child in refined:
        parents = child.parent_ids or ()
        parent_id = str(parents[0]) if parents else ""
        parent = by_parent.get(parent_id)
        if parent is None:
            kept.append(child)
            continue
        claimed_parents.add(parent_id)
        child_fit = candidate_fitness(child)
        parent_fit = candidate_fitness(parent)
        if child is parent or child_fit < parent_fit:
            kept.append(parent)
        else:
            kept.append(child)
    for cid, parent in by_parent.items():
        if cid not in claimed_parents:
            kept.append(parent)
    return kept


def refine_round_count(raw: Optional[int]) -> int:
    try:
        n = int(raw if raw is not None else 1)
    except (TypeError, ValueError):
        return 1
    return max(1, n)


__all__ = [
    "merge_refine_populations",
    "multi_objective_feedback",
    "propose_refined_candidates",
    "refine_round_count",
]
