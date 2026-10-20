"""Compose selection feedback after unchanged diagnostic-card text."""

from __future__ import annotations

from typing import Any, Sequence
from typing import Iterable


def render_candidate_cards(
    candidates: Sequence[Any],
    *,
    archive: Any,
    card_module: Any,
    config: Any,
    population_note: str = "",
) -> list[tuple[str, str]]:
    """Render candidates through the domain's existing card builder unchanged."""
    trim = card_module.truncation_ladder(config)[0]
    cards = []
    for candidate in candidates:
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
        cards.append((str(candidate.candidate_id), card.render(trim).text))
    return cards


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
