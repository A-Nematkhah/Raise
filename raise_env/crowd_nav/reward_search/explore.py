"""
RAISE Stage I evolutionary loop (analytical Score1).

Population N=8, G1=10 generations. Each generation:
  1. Score candidates via ``score1_for_dataset`` (injected / Phase 2).
  2. Rank by score.
  3. Build next population: mutation (lower performers + weakness feedback),
     crossover (top-2), occasional random restart (D.1) — **no** novelty
     filter / archive (those are AMFRS-only).
  4. Accumulate a short reflective note for the next generation's prompts.

Invalid sandbox candidates are replaced by regenerating (Gen0: up to N
extra draws, same spirit as AMFRS "generate 8 extra to replace invalid").
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from crowd_nav.reward_search.llm import (
    LLMClient,
    extract_python_code,
    normalize_to_compute_reward,
    split_reward_function_sources,
)
from crowd_nav.reward_search.scoring import Score1Result, format_score1_failure_hints
from crowd_nav.reward_search import console
from crowd_nav.reward_search.prompts import (
    D1_SYSTEM_PROMPT,
    D5_SEED_FUNCTION,
    format_d1_initial,
    format_d1_initial_batch,
    format_d2_crossover,
    format_d2_mutation,
)
from crowd_nav.reward_search.rejection_log import categorize_validation_error
from crowd_nav.reward_search.sandbox import RewardSandboxError, RewardValidator
from crowd_nav.reward_search.sandbox.runtime import SandboxedReward
from crowd_nav.reward_search.state import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class RewardCandidate:
    """One evolvable reward individual."""

    candidate_id: str
    code: str
    reward_fn: Optional[RewardFunction] = None
    score: Optional[float] = None
    valid: bool = False
    origin: str = "unknown"  # initial | mutation | crossover | random
    parent_ids: tuple = ()
    validation_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.valid and self.reward_fn is not None

    def as_reward_function(self) -> RewardFunction:
        """Return the validated reward callable (Stage I Score1 entry point)."""
        if self.reward_fn is None:
            raise ValueError(f"Candidate {self.candidate_id} has no reward_fn")
        return self.reward_fn


@dataclass
class StageIConfig:
    population_size: int = 8
    generations: int = 10
    n_crossover: int = 2
    n_mutation: int = 4
    n_random: int = 2
    # Extra LLM draws allowed when replacing invalids (Gen0 + each gen).
    max_invalid_replacements: int = 16
    func_name: str = "compute_reward"
    include_external_knowledge: bool = True
    # When True: carry top-1 into next gen and steal one slot from random/mut/xover
    # (engineering elitism). Paper-faithful default is False → exact 2/4/2.
    keep_runtime_elite: bool = False


@dataclass
class GenerationRecord:
    generation: int
    population: List[RewardCandidate]
    ranking: List[str]
    reflection: str
    best_score: Optional[float]
    best_id: Optional[str]


class StageIEvolver:
    """
    Stage I: evolve N reward candidates for G1 generations under Score1.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        score_fn: Optional[Callable[..., float]] = None,
        validator: Optional[RewardValidator] = None,
        config: Optional[StageIConfig] = None,
        system_prompt: str = D1_SYSTEM_PROMPT,
        rejection_log_path: Optional[str] = None,
    ) -> None:
        self.llm = llm
        if score_fn is None:
            raise ValueError(
                "StageIEvolver requires score_fn=. Pass make_score1_fn(load_stage1_dataset(...)) "
                "for paper Score1, or make_smoke_score_fn() only for --fast / --score1 smoke tests."
            )
        self.score_fn = score_fn
        self.validator = validator or RewardValidator()
        self.config = config or StageIConfig()
        self.system_prompt = system_prompt
        self.rejection_log_path = rejection_log_path
        self._counter = 0
        self.reflection: str = ""
        self.history: List[GenerationRecord] = []
        self.global_best: Optional[RewardCandidate] = None

        n = self.config.population_size
        if self.config.n_crossover + self.config.n_mutation + self.config.n_random != n:
            raise ValueError(
                "n_crossover + n_mutation + n_random must equal population_size "
                f"({self.config.n_crossover}+{self.config.n_mutation}+"
                f"{self.config.n_random} != {n})"
            )

    # ------------------------------------------------------------------ helpers

    def _next_id(self, prefix: str) -> str:
        cid = f"{prefix}_{self._counter:04d}"
        self._counter += 1
        return cid

    def _completion_to_code(self, raw: str) -> str:
        code = extract_python_code(raw)
        return normalize_to_compute_reward(code, self.config.func_name)

    @staticmethod
    def _batch_max_tokens(population_size: int) -> int:
        return max(4000, int(population_size) * 500)

    def _log_rejection(
        self,
        *,
        phase: str,
        attempt: int,
        origin: str,
        raw_completion: str,
        extracted_code: str,
        validation_error: Optional[str],
        batch_id: Optional[str] = None,
        accepted: bool = False,
    ) -> None:
        if not self.rejection_log_path:
            return
        record: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "attempt": attempt,
            "origin": origin,
            "batch_id": batch_id,
            "accepted": accepted,
            "raw_preview": (raw_completion or "")[:200],
            "extracted_preview": (extracted_code or "")[:200],
            "validation_error": validation_error,
            "rejection_category": (
                "accepted" if accepted else categorize_validation_error(validation_error)
            ),
        }
        os.makedirs(os.path.dirname(self.rejection_log_path) or ".", exist_ok=True)
        with open(self.rejection_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not accepted:
            # One short WARNING line; raw preview only at DEBUG to avoid flood.
            logger.warning(
                "[Stage I] sandbox reject phase=%s attempt=%s category=%s: %s",
                phase,
                attempt,
                record["rejection_category"],
                (validation_error or "")[:160],
            )
            logger.debug(
                "Gen0 rejection raw_preview=%r extracted_preview=%r",
                record["raw_preview"],
                record["extracted_preview"],
            )

    def _llm_raw(self, user_prompt: str, *, max_tokens: Optional[int] = None) -> str:
        sys_txt = self.system_prompt
        full = f"{sys_txt}\n\n{user_prompt}"
        raw = self.llm.complete(full, max_tokens=max_tokens)
        if not raw or not str(raw).strip():
            raise RuntimeError("LLM returned empty completion (retry layer should raise).")
        effective_max_tokens = max_tokens or getattr(self.llm, "max_tokens", None)
        extract_python_code(str(raw), max_tokens=effective_max_tokens)
        return str(raw)

    def _validate(self, code: str) -> tuple[Optional[SandboxedReward], Optional[str]]:
        return self.validator.try_validate(code)

    def _make_candidate(
        self,
        code: str,
        *,
        origin: str,
        parent_ids: tuple = (),
        metadata: Optional[dict] = None,
    ) -> RewardCandidate:
        cid = self._next_id(origin[:3])
        reward_fn, err = self._validate(code)
        return RewardCandidate(
            candidate_id=cid,
            code=code,
            reward_fn=reward_fn,
            valid=reward_fn is not None,
            origin=origin,
            parent_ids=parent_ids,
            validation_error=err,
            metadata=dict(metadata or {}),
        )

    def _llm_code(self, user_prompt: str, *, system: Optional[str] = None) -> str:
        sys_txt = system if system is not None else self.system_prompt
        full = f"{sys_txt}\n\n{user_prompt}"
        raw = self.llm.complete(full)
        return self._completion_to_code(raw)

    def _attempt_candidate(
        self,
        code: str,
        *,
        origin: str,
        phase: str,
        attempt: int,
        raw_completion: str = "",
        batch_id: Optional[str] = None,
        parent_ids: tuple = (),
        metadata: Optional[dict] = None,
    ) -> RewardCandidate:
        cand = self._make_candidate(
            code,
            origin=origin,
            parent_ids=parent_ids,
            metadata=metadata,
        )
        self._log_rejection(
            phase=phase,
            attempt=attempt,
            origin=origin,
            raw_completion=raw_completion,
            extracted_code=code,
            validation_error=cand.validation_error,
            batch_id=batch_id,
            accepted=cand.valid,
        )
        return cand

    def _fill_valid(
        self,
        needed: int,
        factory: Callable[[int], RewardCandidate],
        *,
        max_extra: int,
        phase: str = "fill",
        fallback: Optional[Callable[[], RewardCandidate]] = None,
    ) -> List[RewardCandidate]:
        """Collect ``needed`` valid candidates; regenerate up to max_extra on failure."""
        out: List[RewardCandidate] = []
        attempts = 0
        budget = needed + max_extra
        while len(out) < needed and attempts < budget:
            cand = factory(attempts + 1)
            attempts += 1
            if cand.valid:
                out.append(cand)
        if len(out) < needed and fallback is not None:
            fb = fallback()
            if fb.valid:
                logger.warning(
                    "Phase %s: using fallback parent code after %d failed LLM attempt(s)",
                    phase,
                    attempts,
                )
                self._log_rejection(
                    phase=phase,
                    attempt=attempts + 1,
                    origin=str(fb.origin),
                    raw_completion="",
                    extracted_code=fb.code,
                    validation_error=None,
                    batch_id=None,
                    accepted=True,
                )
                out.append(fb)
        if len(out) < needed:
            raise RuntimeError(
                f"Could not obtain {needed} valid candidates after {attempts} attempts "
                f"({len(out)} valid)."
            )
        return out

    def _clone_valid(
        self,
        code: str,
        *,
        origin: str,
        parent_ids: tuple = (),
        metadata: Optional[dict] = None,
    ) -> RewardCandidate:
        """Reuse known-valid source when LLM regeneration exhausts its budget."""
        return self._make_candidate(
            code,
            origin=origin,
            parent_ids=parent_ids,
            metadata={**(metadata or {}), "llm_fallback_clone": True},
        )

    # ------------------------------------------------------------------ Gen0

    def initialize_population(self) -> List[RewardCandidate]:
        """
        Gen0: one batch LLM call (Appendix D.1) then per-slot regeneration for
        any candidates that fail sandbox validation.
        """
        cfg = self.config
        needed = cfg.population_size
        out: List[RewardCandidate] = []
        batch_id = f"gen0_batch_{self._counter:04d}"
        console.status(
            f"Gen 0/{cfg.generations} - generating initial population (N={needed})",
            stage="Stage I",
        )
        with console.timed("Gen0 LLM batch", stage="Stage I"):
            batch_prompt = format_d1_initial_batch(
                needed,
                func_name=cfg.func_name,
                include_seed=True,
                include_external_knowledge=cfg.include_external_knowledge,
                reflection=self.reflection,
            )
            raw_batch = self._llm_raw(
                batch_prompt,
                max_tokens=self._batch_max_tokens(needed),
            )
        try:
            codes = split_reward_function_sources(
                raw_batch,
                func_name=cfg.func_name,
                max_tokens=self._batch_max_tokens(needed),
            )
        except Exception as exc:
            console.fail(f"Gen0 batch parse failed: {exc}", stage="Stage I")
            self._log_rejection(
                phase="gen0_batch",
                attempt=1,
                origin="initial",
                raw_completion=raw_batch,
                extracted_code="",
                validation_error=str(exc),
                batch_id=batch_id,
            )
            codes = []
        console.status(
            f"Gen0 batch produced {len(codes)}/{needed} function source(s)",
            stage="Stage I",
        )
        attempt = 0
        for code in codes:
            if len(out) >= needed:
                break
            attempt += 1
            cand = self._attempt_candidate(
                code,
                origin="initial",
                phase="gen0_batch",
                attempt=attempt,
                raw_completion=raw_batch,
                batch_id=batch_id,
            )
            if cand.valid:
                out.append(cand)
                console.status(
                    f"accepted {cand.candidate_id} ({len(out)}/{needed})",
                    stage="Stage I",
                )

        remaining = needed - len(out)
        if remaining <= 0:
            return out

        console.status(
            f"regenerating {remaining} invalid Gen0 slot(s)",
            stage="Stage I",
        )

        def _one(attempt_no: int) -> RewardCandidate:
            prompt = format_d1_initial(
                func_name=cfg.func_name,
                include_seed=True,
                include_external_knowledge=cfg.include_external_knowledge,
                reflection=self.reflection,
            )
            raw = self._llm_raw(prompt)
            code = self._completion_to_code(raw)
            return self._attempt_candidate(
                code,
                origin="initial",
                phase="gen0_regen",
                attempt=attempt_no,
                raw_completion=raw,
                batch_id=None,
            )

        out.extend(
            self._fill_valid(
                remaining,
                _one,
                max_extra=cfg.max_invalid_replacements,
                phase="gen0_regen",
            )
        )
        return out

    # ------------------------------------------------------------------ scoring

    def score_population(self, population: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        scored: List[RewardCandidate] = []
        n = len(population)
        for i, cand in enumerate(population, start=1):
            console.status(
                f"scoring candidate {cand.candidate_id} ({i}/{n})",
                stage="Stage I",
            )
            if not cand.is_executable:
                scored.append(
                    replace(cand, score=float("-inf"), metadata={**cand.metadata, "unscored": True})
                )
                continue
            try:
                raw = self.score_fn(
                    cand.as_reward_function(),
                    candidate_id=cand.candidate_id,
                )
            except Exception as exc:  # noqa: BLE001
                console.fail(
                    f"Score1 crashed for {cand.candidate_id}: {exc}",
                    stage="Stage I",
                )
                raise
            meta = dict(cand.metadata or {})
            # Score1Result (preferred) or plain float (smoke / test helpers).
            if hasattr(raw, "score") and hasattr(raw, "degenerate_fraction"):
                value = float(raw.score)
                meta["degenerate_fraction"] = float(raw.degenerate_fraction)
                meta["score1_n_pairs"] = int(getattr(raw, "n_pairs", 0) or 0)
                meta["score1_n_degenerate"] = int(
                    getattr(raw, "n_degenerate", 0) or 0
                )
                if isinstance(raw, Score1Result):
                    hints = format_score1_failure_hints(raw)
                    if hints:
                        meta["score1_failure_hints"] = hints
                    if raw.scenario_scores:
                        meta["score1_scenario_scores"] = dict(raw.scenario_scores)
                if getattr(raw, "rejected", False):
                    meta["score1_rejected"] = True
                    meta["score1_reject_reason"] = getattr(raw, "reject_reason", None)
                    value = float("-inf")
                    console.warn(
                        f"{cand.candidate_id}: Score1 hard-reject "
                        f"({getattr(raw, 'reject_reason', 'unknown')}; "
                        f"degen={float(raw.degenerate_fraction):.2%})",
                        stage="Stage I",
                    )
                elif float(raw.degenerate_fraction) >= 0.5:
                    console.warn(
                        f"{cand.candidate_id}: high Score1 degeneracy "
                        f"({float(raw.degenerate_fraction):.2%} of pairs "
                        f"had undefined Spearman)",
                        stage="Stage I",
                    )
                if getattr(raw, "holdout_score", None) is not None:
                    meta["score1_holdout"] = float(raw.holdout_score)
                if getattr(raw, "train_score", None) is not None:
                    meta["score1_train"] = float(raw.train_score)
                if getattr(raw, "raw_score", None) is not None:
                    meta["score1_raw"] = float(raw.raw_score)
            else:
                value = float(raw)
            scored.append(replace(cand, score=value, metadata=meta))
        scored.sort(
            key=lambda c: float("-inf") if c.score is None else float(c.score),
            reverse=True,
        )
        return scored

    # ------------------------------------------------------------------ evolution ops

    def _try_crossover(
        self,
        parent_a: RewardCandidate,
        parent_b: RewardCandidate,
        reflection: str,
        *,
        phase: str,
        attempt: int,
    ) -> RewardCandidate:
        prompt = format_d2_crossover(
            parent_a.code,
            parent_b.code,
            reflection,
            func_name=self.config.func_name,
        )
        raw = self._llm_raw(prompt)
        code = self._completion_to_code(raw)
        return self._attempt_candidate(
            code,
            origin="crossover",
            phase=phase,
            attempt=attempt,
            raw_completion=raw,
            parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
        )

    def _try_mutate(
        self,
        parent: RewardCandidate,
        reflection: str,
        *,
        phase: str,
        attempt: int,
    ) -> RewardCandidate:
        weakness = (
            f"Parent {parent.candidate_id} score={parent.score}. "
            f"Weakness focus: improve analytical Score1 relative to elites. "
            f"Global reflection: {reflection}"
        )
        parent_hints = (parent.metadata or {}).get("score1_failure_hints")
        if parent_hints:
            weakness = (
                f"{weakness}\nScore1 diagnostics for this parent: {parent_hints}"
            )
        prompt = format_d2_mutation(
            parent.code,
            weakness,
            func_name=self.config.func_name,
        )
        raw = self._llm_raw(prompt)
        code = self._completion_to_code(raw)
        return self._attempt_candidate(
            code,
            origin="mutation",
            phase=phase,
            attempt=attempt,
            raw_completion=raw,
            parent_ids=(parent.candidate_id,),
            metadata={"weakness": weakness},
        )

    def _try_random_restart(self, *, phase: str, attempt: int) -> RewardCandidate:
        prompt = format_d1_initial(
            func_name=self.config.func_name,
            include_seed=True,
            include_external_knowledge=self.config.include_external_knowledge,
            reflection=self.reflection,
        )
        raw = self._llm_raw(prompt)
        code = self._completion_to_code(raw)
        return self._attempt_candidate(
            code,
            origin="random",
            phase=phase,
            attempt=attempt,
            raw_completion=raw,
        )

    def _fill_random_batch(self, needed: int) -> List[RewardCandidate]:
        """Batch D.1-style random restarts (one LLM call when ``needed > 1``)."""
        if needed <= 0:
            return []
        if needed == 1:
            return self._fill_valid(
                1,
                lambda n: self._try_random_restart(phase="random", attempt=n),
                max_extra=self.config.max_invalid_replacements,
                phase="random",
            )
        batch_id = f"random_batch_{self._counter:04d}"
        prompt = format_d1_initial_batch(
            needed,
            func_name=self.config.func_name,
            include_seed=True,
            include_external_knowledge=self.config.include_external_knowledge,
            reflection=self.reflection,
        )
        raw = self._llm_raw(prompt, max_tokens=self._batch_max_tokens(needed))
        codes = split_reward_function_sources(raw, func_name=self.config.func_name)
        out: List[RewardCandidate] = []
        attempt = 0
        for code in codes:
            if len(out) >= needed:
                break
            attempt += 1
            cand = self._attempt_candidate(
                code,
                origin="random",
                phase="random_batch",
                attempt=attempt,
                raw_completion=raw,
                batch_id=batch_id,
            )
            if cand.valid:
                out.append(cand)
        remaining = needed - len(out)
        if remaining > 0:
            out.extend(
                self._fill_valid(
                    remaining,
                    lambda n: self._try_random_restart(phase="random_regen", attempt=n),
                    max_extra=self.config.max_invalid_replacements,
                    phase="random_regen",
                )
            )
        return out

    def _next_generation(self, ranked: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        cfg = self.config
        if len(ranked) < 2:
            raise ValueError("Need at least 2 ranked candidates to evolve.")

        top_a, top_b = ranked[0], ranked[1]
        generated_crossover = cfg.n_crossover
        generated_mutation = cfg.n_mutation
        generated_random = cfg.n_random
        keep_elite = bool(getattr(cfg, "keep_runtime_elite", False))
        if keep_elite:
            if generated_random > 0:
                generated_random -= 1
            elif generated_mutation > 0:
                generated_mutation -= 1
            elif generated_crossover > 0:
                generated_crossover -= 1
            else:
                raise ValueError("At least one next-generation bucket must be positive.")
        # Lower performers: bottom half (paper: mutate using their weakness).
        lower = list(ranked[len(ranked) // 2 :])
        if not lower:
            lower = list(ranked[-1:])

        next_pop: List[RewardCandidate] = []
        if keep_elite:
            next_pop.append(top_a)

        # Crossover from top-2.
        for i in range(generated_crossover):
            def _cross(attempt_no: int, _i=i) -> RewardCandidate:
                del _i
                return self._try_crossover(
                    top_a, top_b, self.reflection, phase="crossover", attempt=attempt_no
                )

            next_pop.extend(
                self._fill_valid(
                    1,
                    _cross,
                    max_extra=cfg.max_invalid_replacements,
                    phase="crossover",
                    fallback=lambda: self._clone_valid(
                        top_a.code,
                        origin="crossover_fallback",
                        parent_ids=(top_a.candidate_id, top_b.candidate_id),
                    ),
                )
            )

        # Mutation on lower performers (cycle if fewer parents than slots).
        for i in range(generated_mutation):
            parent = lower[i % len(lower)]

            def _mut(attempt_no: int, _p=parent) -> RewardCandidate:
                return self._try_mutate(
                    _p, self.reflection, phase="mutation", attempt=attempt_no
                )

            next_pop.extend(
                self._fill_valid(
                    1,
                    _mut,
                    max_extra=cfg.max_invalid_replacements,
                    phase="mutation",
                    fallback=lambda _p=parent: self._clone_valid(
                        _p.code,
                        origin="mutation_fallback",
                        parent_ids=(_p.candidate_id,),
                    ),
                )
            )

        # Occasional random restarts (D.1-style) — batched when n_random > 1.
        if generated_random > 0:
            next_pop.extend(self._fill_random_batch(generated_random))

        assert len(next_pop) == cfg.population_size
        return next_pop

    def _build_reflection(
        self, ranked: Sequence[RewardCandidate], *, generation: int
    ) -> str:
        """
        Reflective note for the next generation (§4.2).

        Accumulates concise notes across generations (bounded), including
        Score1 for all candidates — not only best/worst.
        """
        score_bits = []
        for c in ranked:
            sc = c.score if c.score is not None else float("nan")
            score_bits.append(f"{c.candidate_id}={sc}")
        best = ranked[0]
        worst = ranked[-1]
        lower = ranked[len(ranked) // 2 :]
        lower_ids = ", ".join(c.candidate_id for c in lower[:4]) if lower else "(none)"
        note = (
            f"Gen{generation} scores[{', '.join(score_bits)}]. "
            f"Best={best.candidate_id} score={best.score}; "
            f"Worst={worst.candidate_id} score={worst.score}. "
            f"Underperformers ({lower_ids}): strengthen goal progress and "
            f"collision/discomfort penalties while keeping dense shaping. "
            f"Prefer combining elite safety terms with efficient progress; "
            f"avoid near-constant rewards."
        )
        diag_bits = []
        for c in (worst, *(lower[:2])):
            hints = (c.metadata or {}).get("score1_failure_hints")
            if hints:
                diag_bits.append(f"{c.candidate_id}: {hints}")
        if diag_bits:
            # Deduplicate while preserving order.
            seen = set()
            uniq = []
            for b in diag_bits:
                if b not in seen:
                    seen.add(b)
                    uniq.append(b)
            note = note + " Diagnostics: " + " | ".join(uniq[:3])
        if not self.reflection.strip():
            return note
        parts = [p.strip() for p in self.reflection.split(" || ") if p.strip()]
        parts.append(note)
        # Bound prompt size: keep the most recent notes.
        return " || ".join(parts[-3:])

    # ------------------------------------------------------------------ run

    def run(self) -> List[RewardCandidate]:
        """
        Run Gen0 + G1 evolutionary generations.

        Returns the final ranked population (best first).
        """
        console.banner("Stage I - analytical Score1 evolution")
        with console.timed("initialize Gen0 population", stage="Stage I"):
            population = self.initialize_population()
        ranked = self.score_population(population)
        self.global_best = ranked[0]
        self.reflection = self._build_reflection(ranked, generation=0)
        self.history.append(
            GenerationRecord(
                generation=0,
                population=list(ranked),
                ranking=[c.candidate_id for c in ranked],
                reflection=self.reflection,
                best_score=ranked[0].score,
                best_id=ranked[0].candidate_id,
            )
        )
        console.stage1_gen_summary(
            0, self.config.generations, ranked, global_best=self.global_best
        )

        for g in range(1, self.config.generations + 1):
            previous_best = ranked[0]
            console.status(
                f"Gen {g}/{self.config.generations} - evolving next population "
                f"(crossover={self.config.n_crossover}, "
                f"mutation={self.config.n_mutation}, "
                f"random={self.config.n_random})",
                stage="Stage I",
            )
            with console.timed(f"Gen {g} evolve+score", stage="Stage I"):
                population = self._next_generation(ranked)
                ranked = self.score_population(population)
            if ranked[0].score < previous_best.score:
                console.warn(
                    f"regression: gen={g} best {ranked[0].candidate_id} "
                    f"score={ranked[0].score:.6f} below previous "
                    f"{previous_best.candidate_id} score={previous_best.score:.6f}",
                    stage="Stage I",
                )
            if ranked[0].score > self.global_best.score:
                self.global_best = ranked[0]
            self.reflection = self._build_reflection(ranked, generation=g)
            self.history.append(
                GenerationRecord(
                    generation=g,
                    population=list(ranked),
                    ranking=[c.candidate_id for c in ranked],
                    reflection=self.reflection,
                    best_score=ranked[0].score,
                    best_id=ranked[0].candidate_id,
                )
            )
            console.stage1_gen_summary(
                g, self.config.generations, ranked, global_best=self.global_best
            )

        return ranked


def seed_reward_code() -> str:
    """Appendix D.5 seed, already adapted to ``compute_reward(state)``."""
    return D5_SEED_FUNCTION
