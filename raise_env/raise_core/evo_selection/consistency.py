"""Rank-consistency metrics and CSV persistence."""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Sequence

logger = logging.getLogger(__name__)


def compare_rankings(
    previous: Sequence[str], current: Sequence[str], *, top_k: int
) -> dict[str, float | int | None]:
    common = set(previous).intersection(current)
    previous_common = [candidate for candidate in previous if candidate in common]
    current_common = [candidate for candidate in current if candidate in common]
    correlation: float | None = None
    if len(common) >= 2:
        old_rank = {candidate: index for index, candidate in enumerate(previous_common)}
        new_rank = {candidate: index for index, candidate in enumerate(current_common)}
        old_values = [float(old_rank[candidate]) for candidate in previous_common]
        new_values = [float(new_rank[candidate]) for candidate in previous_common]
        old_mean = sum(old_values) / len(old_values)
        new_mean = sum(new_values) / len(new_values)
        numerator = sum(
            (old - old_mean) * (new - new_mean)
            for old, new in zip(old_values, new_values)
        )
        old_norm = math.sqrt(sum((old - old_mean) ** 2 for old in old_values))
        new_norm = math.sqrt(sum((new - new_mean) ** 2 for new in new_values))
        if old_norm > 0.0 and new_norm > 0.0:
            correlation = numerator / (old_norm * new_norm)

    k = max(1, int(top_k))
    old_top = set(previous[:k])
    new_top = set(current[:k])
    denominator = min(k, len(old_top), len(new_top))
    preservation = (
        len(old_top.intersection(new_top)) / denominator if denominator else 0.0
    )
    return {
        "spearman": correlation,
        "top_k_preservation": preservation,
        "intersection_n": len(common),
    }


def append_consistency_csv(
    path: str,
    *,
    stage: str,
    generation: int,
    previous: Sequence[str],
    current: Sequence[str],
    top_k: int,
) -> dict[str, float | int | None]:
    metrics = compare_rankings(previous, current, top_k=top_k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "generation",
                "spearman",
                "top_k",
                "top_k_preservation",
                "intersection_n",
            ),
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "stage": stage,
                "generation": int(generation),
                "spearman": metrics["spearman"],
                "top_k": int(top_k),
                "top_k_preservation": metrics["top_k_preservation"],
                "intersection_n": metrics["intersection_n"],
            }
        )
    logger.info(
        "Selection consistency stage=%s generation=%d spearman=%s top-%d preservation=%.3f common=%d",
        stage,
        generation,
        "n/a" if metrics["spearman"] is None else f"{metrics['spearman']:.3f}",
        int(top_k),
        float(metrics["top_k_preservation"] or 0.0),
        int(metrics["intersection_n"]),
    )
    return metrics
