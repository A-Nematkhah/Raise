"""Strict-JSON, multi-vote LLM ranking over unchanged diagnostic cards."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from raise_core.evo_selection.config import SelectionConfig

logger = logging.getLogger(__name__)
MAX_INVALID_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 1800


def anonymous_labels(candidate_ids: Sequence[str]) -> dict[str, str]:
    def label_for(index: int) -> str:
        result = ""
        value = int(index)
        while True:
            value, remainder = divmod(value, 26)
            result = chr(ord("A") + remainder) + result
            if value == 0:
                return result
            value -= 1

    return {str(candidate_id): label_for(index) for index, candidate_id in enumerate(candidate_ids)}


def anonymize_card_text(card_text: str, candidate_id: str, label: str) -> str:
    candidate_id = str(candidate_id)
    if len(candidate_id) == 1:
        pattern = re.compile(
            rf"(\b(?:candidate[\s_-]*)?id\s*[:=]\s*){re.escape(candidate_id)}(?![\w])",
            flags=re.IGNORECASE | re.UNICODE,
        )
        return pattern.sub(rf"\1Candidate {label}", str(card_text))
    pattern = re.compile(
        rf"(?<![\w]){re.escape(candidate_id)}(?![\w])",
        flags=re.UNICODE,
    )
    return pattern.sub(f"Candidate {label}", str(card_text))


def ranking_prompt(cards: Sequence[tuple[str, str]], *, flags: Mapping[str, Sequence[str]] | None = None) -> str:
    sections = []
    for label, card_text in cards:
        text = str(card_text)
        candidate_flags = list((flags or {}).get(label, ()))
        if candidate_flags:
            text += "\nSanity flags: " + "; ".join(candidate_flags)
        sections.append(f"===== Candidate {label} =====\n{text}")
    labels = [label for label, _ in cards]
    return (
        "Rank these reward-function candidates using the diagnostic cards below. "
        "Treat each card as the sole source of candidate evidence. Use a "
        "multi-objective judgment of task success, safety, timeouts, navigation "
        "quality, and evidence reliability; do not infer unreported metrics. "
        "For every candidate, provide concrete weaknesses and one-sentence "
        "rationale. Rank best to worst. Return STRICT JSON only, with exactly "
        'these keys and no markdown: {"ranking":["A"],"scores":{"A":0.5},'
        '"weaknesses":{"A":["..."]},"rationale":{"A":"..."}}. '
        "Include every supplied label exactly once in ranking and exactly once "
        "as a key in scores, weaknesses, and rationale. Scores must be JSON "
        "numbers in [0,1]. The supplied labels are: "
        + ", ".join(labels)
        + ".\n\n"
        + "\n\n".join(sections)
    )


def _validate_response(raw: str, labels: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(raw)
    expected = set(labels)
    if not isinstance(payload, dict) or set(payload) != {
        "ranking",
        "scores",
        "weaknesses",
        "rationale",
    }:
        raise ValueError("response must be an object with exactly the required keys")
    ranking = payload["ranking"]
    if (
        not isinstance(ranking, list)
        or any(not isinstance(label, str) for label in ranking)
        or len(ranking) != len(labels)
        or set(ranking) != expected
        or len(set(ranking)) != len(ranking)
    ):
        raise ValueError("ranking must contain every label exactly once")
    for key in ("scores", "weaknesses", "rationale"):
        value = payload[key]
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"{key} must contain exactly every supplied label")
    for label, value in payload["scores"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"score for {label} must be a JSON number")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"score for {label} must be finite and in [0, 1]")
    for label, value in payload["weaknesses"].items():
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"weaknesses for {label} must be a list of strings")
    for label, value in payload["rationale"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rationale for {label} must be a non-empty string")
    return payload


@dataclass
class RankResult:
    ranking: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    weaknesses: dict[str, list[str]] = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)
    label_to_id: dict[str, str] = field(default_factory=dict)
    fallback: bool = False
    fallback_reason: str | None = None
    votes: list[dict[str, Any]] = field(default_factory=list)


class LLMRanker:
    def __init__(
        self,
        llm: Any,
        config: SelectionConfig,
        *,
        log_path: str | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.log_path = log_path

    def _log(self, payload: dict[str, Any]) -> None:
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def rank(
        self,
        cards: Sequence[tuple[str, str]],
        *,
        fallback_order: Sequence[str],
        seed: int | None = None,
        stage: str = "stage2",
        generation: int = 0,
        flags: Mapping[str, Sequence[str]] | None = None,
    ) -> RankResult:
        ids = [str(candidate_id) for candidate_id, _ in cards]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("ranker requires a non-empty list of unique candidate ids")
        if set(ids) != set(str(candidate_id) for candidate_id in fallback_order):
            raise ValueError("fallback_order must contain exactly the card candidate ids")
        id_to_label = anonymous_labels(ids)
        label_to_id = {label: candidate_id for candidate_id, label in id_to_label.items()}
        anonymized = [
            (
                id_to_label[candidate_id],
                anonymize_card_text(card_text, candidate_id, id_to_label[candidate_id]),
            )
            for candidate_id, card_text in cards
        ]
        label_flags = {
            id_to_label[candidate_id]: list(values)
            for candidate_id, values in (flags or {}).items()
            if candidate_id in id_to_label
        }
        rng = random.Random(self.config.seed if seed is None else int(seed))
        labels = [label for label, _ in anonymized]
        valid_votes: list[dict[str, Any]] = []

        original_temperature = getattr(self.llm, "temperature", None)
        has_temperature = hasattr(self.llm, "temperature")
        original_system_prompt = getattr(self.llm, "system_prompt", None)
        has_system_prompt = hasattr(self.llm, "system_prompt")
        ranking_system_prompt = (
            "You are a multi-objective candidate evaluator. Return strict JSON only."
        )
        if has_temperature:
            self.llm.temperature = float(self.config.temperature)
        if has_system_prompt:
            self.llm.system_prompt = ranking_system_prompt
        try:
            for vote_index in range(self.config.n_votes):
                parsed_vote = None
                for attempt in range(1, MAX_INVALID_ATTEMPTS + 1):
                    shuffled = list(anonymized)
                    rng.shuffle(shuffled)
                    prompt = ranking_prompt(shuffled, flags=label_flags)
                    try:
                        raw = str(
                            self.llm.complete(prompt, max_tokens=MAX_OUTPUT_TOKENS)
                        )
                    except Exception as exc:
                        self._log(
                            {
                                "event": "ranking_vote",
                                "stage": stage,
                                "generation": int(generation),
                                "vote": vote_index + 1,
                                "attempt": attempt,
                                "system_prompt": ranking_system_prompt,
                                "prompt": prompt,
                                "raw_response": None,
                                "parsed_result": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        raise
                    try:
                        parsed_vote = _validate_response(raw, labels)
                        validation_error = None
                    except (json.JSONDecodeError, ValueError) as exc:
                        validation_error = f"{type(exc).__name__}: {exc}"
                    self._log(
                        {
                            "event": "ranking_vote",
                            "stage": stage,
                            "generation": int(generation),
                            "vote": vote_index + 1,
                            "attempt": attempt,
                            "system_prompt": ranking_system_prompt,
                            "prompt": prompt,
                            "raw_response": raw,
                            "parsed_result": parsed_vote,
                            "validation_error": validation_error,
                        }
                    )
                    if parsed_vote is not None:
                        valid_votes.append(parsed_vote)
                        break
                if parsed_vote is None:
                    reason = (
                        f"vote {vote_index + 1} remained invalid after "
                        f"{MAX_INVALID_ATTEMPTS} attempts"
                    )
                    self._log(
                        {
                            "event": "ranking_fallback",
                            "stage": stage,
                            "generation": int(generation),
                            "reason": reason,
                            "ranking": list(fallback_order),
                        }
                    )
                    return RankResult(
                        ranking=list(fallback_order),
                        label_to_id=label_to_id,
                        fallback=True,
                        fallback_reason=reason,
                        votes=valid_votes,
                    )
        finally:
            if has_temperature:
                self.llm.temperature = original_temperature
            if has_system_prompt:
                self.llm.system_prompt = original_system_prompt

        borda = {label: 0 for label in labels}
        score_values = {label: [] for label in labels}
        weakness_values = {label: [] for label in labels}
        rationale_values = {label: [] for label in labels}
        for vote in valid_votes:
            for position, label in enumerate(vote["ranking"]):
                borda[label] += len(labels) - position - 1
            for label in labels:
                score_values[label].append(float(vote["scores"][label]))
                for weakness in vote["weaknesses"][label]:
                    if weakness not in weakness_values[label]:
                        weakness_values[label].append(weakness)
                rationale_values[label].append(vote["rationale"][label])

        fallback_position = {candidate_id: index for index, candidate_id in enumerate(fallback_order)}
        ranking_labels = sorted(
            labels,
            key=lambda label: (
                -borda[label],
                fallback_position[label_to_id[label]],
            ),
        )
        result = RankResult(
            ranking=[label_to_id[label] for label in ranking_labels],
            scores={
                label_to_id[label]: sum(score_values[label]) / len(score_values[label])
                for label in labels
            },
            weaknesses={
                label_to_id[label]: weakness_values[label] for label in labels
            },
            rationale={
                label_to_id[label]: rationale_values[label][0] for label in labels
            },
            label_to_id=label_to_id,
            votes=valid_votes,
        )
        self._log(
            {
                "event": "ranking_aggregate",
                "stage": stage,
                "generation": int(generation),
                "label_to_id": label_to_id,
                "ranking": result.ranking,
                "scores": result.scores,
                "weaknesses": result.weaknesses,
                "rationale": result.rationale,
                "borda": {label_to_id[label]: borda[label] for label in labels},
            }
        )
        return result
