"""Compose selection feedback after unchanged diagnostic-card text."""

from __future__ import annotations

import logging
from typing import Any, Sequence
from typing import Iterable, List, Optional, Tuple

from raise_core.llm import estimate_groq_prompt_tokens
from raise_core.token_budget import BudgetFit, PromptBudgetError

logger = logging.getLogger(__name__)

_RANKING_SYSTEM = (
    "You are a multi-objective candidate evaluator. Return strict JSON only."
)

_CARD_BOILERPLATE_PREFIXES = (
    "=== DIAGNOSTIC FEEDBACK CARD",
    "Numbers and 'Observation'",
    "How to use this card:",
)


def strip_card_boilerplate(text: str) -> str:
    """Drop per-card legend/title/guidance duplicated N times in llm_rank."""
    lines = []
    for line in str(text or "").splitlines():
        if any(line.startswith(p) for p in _CARD_BOILERPLATE_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def render_candidate_cards(
    candidates: Sequence[Any],
    *,
    archive: Any,
    card_module: Any,
    config: Any,
    population_note: str = "",
    fit_to_input_budget: bool = True,
    provider: str = "groq",
) -> list[tuple[str, str]]:
    """Render candidates through the domain's card builder.

    When ``fit_to_input_budget`` is True (default), walk the domain truncation
    ladder and, if needed, drop lowest-priority candidates (end of ``candidates``)
    so a full ``llm_rank`` request stays within ``config.target_input_tokens`` /
    ``config.max_input_tokens``. Pass candidates in fitness-descending order so
    drops remove the weakest first.
    """
    if not candidates:
        return []
    ladder = card_module.truncation_ladder(config)
    if not fit_to_input_budget:
        trim = ladder[0]
        return [
            (
                cid,
                strip_card_boilerplate(text),
            )
            for cid, text in (
                _render_one(
                    candidate,
                    archive=archive,
                    card_module=card_module,
                    config=config,
                    population_note=population_note,
                    trim=trim,
                )
                for candidate in candidates
            )
        ]

    from raise_core.evo_selection.llm_ranker import (
        MAX_OUTPUT_TOKENS,
        anonymous_labels,
        ranking_prompt,
    )
    from raise_core.llm import groq_margined_input_budget

    # Margined prompt estimate must leave room for ranking completion under TPM.
    out_reserve = int(MAX_OUTPUT_TOKENS)
    max_in = int(config.max_input_tokens)
    target = int(config.target_input_tokens)
    if str(provider or "").lower() in ("", "groq"):
        max_in = min(max_in, groq_margined_input_budget(out_reserve))
        target = min(target, max_in)
    ordered = list(candidates)

    def _cards_at(trim: Any, subset: Sequence[Any]) -> list[tuple[str, str]]:
        return [
            (cid, strip_card_boilerplate(text))
            for cid, text in (
                _render_one(
                    candidate,
                    archive=archive,
                    card_module=card_module,
                    config=config,
                    population_note=population_note,
                    trim=trim,
                )
                for candidate in subset
            )
        ]

    def _request_tokens(cards: Sequence[tuple[str, str]]) -> int:
        ids = [cid for cid, _ in cards]
        labels = anonymous_labels(ids)
        labeled = [(labels[cid], text) for cid, text in cards]
        prompt = ranking_prompt(labeled)
        # Margined estimate so ranking prompts leave room under Groq TPM.
        return estimate_groq_prompt_tokens(_RANKING_SYSTEM, prompt)

    best_cards: Optional[list[tuple[str, str]]] = None
    best_tokens = 0
    for trim in ladder:
        cards = _cards_at(trim, ordered)
        tokens = _request_tokens(cards)
        best_cards, best_tokens = cards, tokens
        if tokens <= target:
            if trim.name != ladder[0].name:
                logger.info(
                    "ranking cards truncated to %r (%d tokens <= target %d)",
                    trim.name,
                    tokens,
                    target,
                )
            return cards

    # Still over target at max truncation: drop weakest until under max (keep ≥2).
    subset = list(ordered)
    trim = ladder[-1]
    while True:
        cards = _cards_at(trim, subset)
        tokens = _request_tokens(cards)
        best_cards, best_tokens = cards, tokens
        if tokens <= target or (tokens <= max_in and len(subset) <= 2):
            if len(subset) < len(ordered):
                logger.warning(
                    "ranking pool reduced %d → %d to fit input budget "
                    "(%d tokens, max=%d)",
                    len(ordered),
                    len(subset),
                    tokens,
                    max_in,
                )
            if tokens <= max_in:
                return cards
            break
        if len(subset) <= 2:
            break
        subset = subset[:-1]

    assert best_cards is not None
    if best_tokens > max_in:
        raise PromptBudgetError(
            f"llm_rank request needs ~{best_tokens} input tokens "
            f"(margined) after truncating cards and shrinking the pool "
            f"to {len(best_cards)}; hard limit is {max_in}",
            BudgetFit(
                messages=[],
                tokens=best_tokens,
                level=len(ladder) - 1,
                steps=["ranking_card_truncation", "ranking_pool_shrink"],
                info={"n_cards": len(best_cards)},
                within_target=False,
            ),
        )
    return best_cards


def _render_one(
    candidate: Any,
    *,
    archive: Any,
    card_module: Any,
    config: Any,
    population_note: str,
    trim: Any,
) -> tuple[str, str]:
    record = archive.get(candidate.candidate_id)
    if record is None:
        raise KeyError(f"No archived card record for {candidate.candidate_id}")
    card = card_module.FeedbackCard(
        call_type="candidate",
        subjects=[record],
        archive=archive,
        config=config,
        population_note=population_note,
    )
    return (str(candidate.candidate_id), card.render(trim).text)


def build_refinement_feedback(
    card_text: str,
    *,
    weaknesses: Iterable[str] = (),
    reflection_notes: Iterable[str] = (),
) -> str:
    """Append localized refinement guidance without editing the diagnostic card."""
    lines = [str(card_text)]
    weakness_items = [str(item).strip() for item in weaknesses if str(item).strip()]
    note_items = [str(item).strip() for item in reflection_notes if str(item).strip()]
    if weakness_items or note_items:
        lines.append("")
        lines.append("LLM selection feedback (separate from the diagnostic card):")
        if weakness_items:
            lines.append("Weaknesses to address: " + "; ".join(weakness_items))
        if note_items:
            lines.append("Accumulated reflection notes: " + " | ".join(note_items))
        lines.append(
            "Make minimal, localized edits; preserve working terms and the "
            "compute_reward(state, memory) signature. Return only code."
        )
    return "\n".join(lines)
