"""Seeded elitist offspring plans for LLM-ranked populations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from raise_core.candidate import RewardCandidate, genome_fingerprint


@dataclass(frozen=True)
class Offspring:
    operator: str
    parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvolutionPlan:
    elites: tuple[RewardCandidate, ...]
    offspring: tuple[Offspring, ...]


def make_evolution_plan(
    ranked: Sequence[RewardCandidate],
    *,
    population_size: int,
    top_k: int,
    p_restart: float,
    seed: int,
    generation: int,
) -> EvolutionPlan:
    if len(ranked) < 2:
        raise ValueError("LLM-ranked evolution requires at least two candidates")
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    elite_count = min(max(1, int(top_k)), int(population_size) - 1, len(ranked))
    elites = tuple(ranked[:elite_count])
    parents = elites
    parent_a = parents[0]
    parent_a_fp = genome_fingerprint(parent_a.code)
    parent_b = next(
        (item for item in ranked[1:] if genome_fingerprint(item.code) != parent_a_fp),
        ranked[1],
    )
    rng = random.Random(f"{int(seed)}:{int(generation)}:" + ",".join(
        str(candidate.candidate_id) for candidate in ranked
    ))
    offspring: list[Offspring] = []
    for index in range(int(population_size) - elite_count):
        if rng.random() < float(p_restart):
            offspring.append(Offspring("random", ()))
        elif index % 2 == 0:
            parent = rng.choice(parents)
            offspring.append(Offspring("mutation", (str(parent.candidate_id),)))
        else:
            offspring.append(
                Offspring(
                    "crossover",
                    (str(parent_a.candidate_id), str(parent_b.candidate_id)),
                )
            )
    return EvolutionPlan(elites=elites, offspring=tuple(offspring))
