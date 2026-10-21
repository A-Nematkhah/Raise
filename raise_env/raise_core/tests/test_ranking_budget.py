"""llm_rank card rendering must fit the configured input-token budget."""

from __future__ import annotations

from raise_core.candidate import RewardCandidate
from raise_core.evo_selection.feedback import render_candidate_cards
from raise_core.evo_selection.llm_ranker import (
    MAX_OUTPUT_TOKENS,
    anonymous_labels,
    ranking_prompt,
)
from raise_core.feedback import CandidateArchive, FeedbackCardConfig
from raise_core.llm import (
    _GROQ_ORG_TOKEN_LIMIT,
    _GROQ_REQUEST_TOKEN_CAP,
    estimate_groq_prompt_tokens,
)

import domains.crowdnav.feedback_card as card_module


def _cand(i: int, *, fitness: float) -> RewardCandidate:
    code = "def compute_reward(state, memory):\n    return 0.0\n"
    return RewardCandidate(
        candidate_id=f"c{i}",
        code=code,
        valid=True,
        origin="mutation",
        metadata={
            "fitness": fitness,
            "last_metrics": {
                "SR": 0.7,
                "CR": 0.1,
                "TR": 0.2,
                "NT": 12.0,
                "PL": 9.0,
                "ITR": 0.05,
                "SD": 0.12,
                "fitness": fitness,
                "n_eval_episodes": 10,
            },
        },
    )


def test_ranking_cards_fit_under_tight_budget():
    pop = [_cand(i, fitness=1.0 - 0.05 * i) for i in range(8)]
    archive = CandidateArchive()
    archive.add_population(pop, generation=0)
    cfg = FeedbackCardConfig(
        enabled=True,
        max_input_tokens=3500,
        target_input_tokens=3000,
        history_entries=0,
        max_trajectories=0,
        training_dynamics=False,
        component_instrumentation=False,
    )
    cards = render_candidate_cards(
        pop,
        archive=archive,
        card_module=card_module,
        config=cfg,
        fit_to_input_budget=True,
        provider="groq",
    )
    assert 2 <= len(cards) <= 8
    labels = anonymous_labels([cid for cid, _ in cards])
    labeled = [(labels[cid], text) for cid, text in cards]
    prompt = ranking_prompt(labeled)
    system = "You are a multi-objective candidate evaluator. Return strict JSON only."
    tokens = estimate_groq_prompt_tokens(system, prompt)
    assert tokens <= cfg.max_input_tokens
    # Full request (margined prompt + ranking completion) must stay under soft/org caps.
    assert tokens + int(MAX_OUTPUT_TOKENS) <= int(_GROQ_REQUEST_TOKEN_CAP)
    assert tokens + int(MAX_OUTPUT_TOKENS) <= int(_GROQ_ORG_TOKEN_LIMIT)
