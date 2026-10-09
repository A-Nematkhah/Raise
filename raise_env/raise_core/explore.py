"""
RAISE reward evolution operators (LLM proposal + Score1 scoring).

``StageIEvolver`` provides Gen0 proposal, Score1 scoring, and next-generation
breeding (crossover of the top-2, mutation of lower performers with Stage II
evaluation feedback, random restarts). The closed-loop runner drives it.

Invalid sandbox candidates are replaced by regenerating; after the retry
budget the domain seed (or the parent source) fills the slot.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from domains.highway import prompts as highway_prompts
from raise_core.llm import (
    EmptyCompletionError,
    LLMClient,
    extract_python_code,
    is_recoverable_llm_error,
    normalize_to_compute_reward,
    split_reward_function_sources,
)
from raise_core.scoring import Score1Result
from raise_core import console
from raise_core.rejection_log import categorize_validation_error
from raise_core.sandbox import RewardValidator
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.runtime import RewardFunction, SandboxedReward

logger = logging.getLogger(__name__)

_DIAGNOSIS_FENCE_RE = re.compile(r"```(?:python)?\s*\n", re.IGNORECASE)
_CANDIDATE_SERIAL_RE = re.compile(r"^[A-Za-z]+_(\d+)")


def extract_llm_diagnosis(raw: str, *, max_chars: int = 500) -> Optional[str]:
    """Plain text preceding the first code fence, or None if absent."""
    text = str(raw or "")
    match = _DIAGNOSIS_FENCE_RE.search(text)
    if match is None:
        return None
    preamble = text[: match.start()].strip()
    if not preamble:
        return None
    return preamble[: int(max_chars)]


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


class StageIEvolver:
    """LLM reward proposal, Score1 scoring and next-generation breeding."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        score_fn: Optional[Callable[..., float]] = None,
        validator: Optional[RewardValidator] = None,
        config: Optional[StageIConfig] = None,
        rejection_log_path: Optional[str] = None,
        prompts: Any = None,
    ) -> None:
        self.llm = llm
        if score_fn is None:
            raise ValueError(
                "StageIEvolver requires score_fn= (pack.make_score_fn(...))."
            )
        if validator is None:
            raise ValueError(
                "StageIEvolver requires validator= (make_validator_for_domain(pack))."
            )
        self.score_fn = score_fn
        self.validator = validator
        self.config = config or StageIConfig()
        self._prompts = prompts if prompts is not None else highway_prompts
        self.system_prompt = str(self._prompts.D1_SYSTEM_PROMPT)
        self.rejection_log_path = rejection_log_path
        self._counter = 0
        self.reflection: str = ""
        self.global_best: Optional[RewardCandidate] = None
        # Best-fitness trend across generations (numbers only; fed into reflection).
        self._best_fitness_trend: List[float] = []

        n = self.config.population_size
        if self.config.n_crossover + self.config.n_mutation + self.config.n_random != n:
            raise ValueError(
                "n_crossover + n_mutation + n_random must equal population_size "
                f"({self.config.n_crossover}+{self.config.n_mutation}+"
                f"{self.config.n_random} != {n})"
            )

    def _format_d1_initial(self, **kwargs: Any) -> str:
        return self._prompts.format_d1_initial(**kwargs)

    def _format_d1_initial_batch(self, n: int, **kwargs: Any) -> str:
        return self._prompts.format_d1_initial_batch(n, **kwargs)

    def _format_d2_crossover(self, *args: Any, **kwargs: Any) -> str:
        return self._prompts.format_d2_crossover(*args, **kwargs)

    def _format_d2_mutation(self, *args: Any, **kwargs: Any) -> str:
        return self._prompts.format_d2_mutation(*args, **kwargs)

    # ------------------------------------------------------------------ helpers

    def _next_id(self, prefix: str) -> str:
        cid = f"{prefix}_{self._counter:04d}"
        self._counter += 1
        return cid

    @property
    def id_counter(self) -> int:
        return int(self._counter)

    def restore_id_counter(
        self,
        saved: Optional[int] = None,
        *,
        existing_ids: Sequence[str] = (),
    ) -> int:
        """Advance the ID serial past ``saved`` and every ``<prefix>_<NNNN>`` id.

        Resumed runs must not re-issue ids already present in the population,
        epoch history or Stage II training folders.
        """
        nxt = max(int(self._counter), int(saved or 0))
        for cid in existing_ids:
            match = _CANDIDATE_SERIAL_RE.match(str(cid))
            if match:
                nxt = max(nxt, int(match.group(1)) + 1)
        self._counter = nxt
        return nxt

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

    def _seed_code(self) -> str:
        return normalize_to_compute_reward(str(self._prompts.D5_SEED_FUNCTION))

    def _seed_fallback_candidate(self, *, origin: str = "seed_fallback") -> RewardCandidate:
        return self._clone_valid(self._seed_code(), origin=origin)

    def _llm_raw(self, user_prompt: str, *, max_tokens: Optional[int] = None) -> str:
        sys_txt = self.system_prompt
        full = f"{sys_txt}\n\n{user_prompt}"
        raw = self.llm.complete(full, max_tokens=max_tokens)
        if not raw or not str(raw).strip():
            raise EmptyCompletionError(
                "LLM returned empty completion (retry layer should raise)."
            )
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
        md = dict(metadata or {})
        if raw_completion:
            md["llm_diagnosis"] = extract_llm_diagnosis(raw_completion)
        cand = self._make_candidate(
            code,
            origin=origin,
            parent_ids=parent_ids,
            metadata=md,
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
            try:
                cand = factory(attempts + 1)
            except Exception as exc:  # noqa: BLE001
                if not is_recoverable_llm_error(exc):
                    raise
                attempts += 1
                logger.warning(
                    "Phase %s: recoverable LLM failure on attempt %d/%d (%s): %s",
                    phase,
                    attempts,
                    budget,
                    type(exc).__name__,
                    exc,
                )
                continue
            attempts += 1
            if cand.valid:
                out.append(cand)
        # Default fallback: domain seed so closed-loop never dies on LLM flakes.
        pad = fallback if fallback is not None else self._seed_fallback_candidate
        while len(out) < needed:
            fb = pad()
            if not fb.valid:
                raise RuntimeError(
                    f"Could not obtain {needed} valid candidates after {attempts} "
                    f"attempts ({len(out)} valid); fallback also invalid."
                )
            logger.warning(
                "Phase %s: using fallback after %d failed LLM attempt(s) "
                "(%d/%d slots filled)",
                phase,
                attempts,
                len(out) + 1,
                needed,
            )
            self._log_rejection(
                phase=phase,
                attempt=attempts + len(out) + 1,
                origin=str(fb.origin),
                raw_completion="",
                extracted_code=fb.code,
                validation_error=None,
                batch_id=None,
                accepted=True,
            )
            out.append(fb)
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
            batch_prompt = self._format_d1_initial_batch(
                needed,
                func_name=cfg.func_name,
                include_seed=True,
                include_external_knowledge=cfg.include_external_knowledge,
                reflection=self.reflection,
            )
            try:
                raw_batch = self._llm_raw(
                    batch_prompt,
                    max_tokens=self._batch_max_tokens(needed),
                )
            except Exception as exc:  # noqa: BLE001
                if not is_recoverable_llm_error(exc):
                    raise
                logger.warning(
                    "Gen0 batch LLM failed (%s): %s; falling back to per-slot regen",
                    type(exc).__name__,
                    exc,
                )
                self._log_rejection(
                    phase="gen0_batch",
                    attempt=1,
                    origin="initial",
                    raw_completion="",
                    extracted_code="",
                    validation_error=str(exc),
                    batch_id=batch_id,
                )
                raw_batch = ""
        try:
            codes = (
                split_reward_function_sources(
                    raw_batch,
                    func_name=cfg.func_name,
                    max_tokens=self._batch_max_tokens(needed),
                )
                if raw_batch
                else []
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
            prompt = self._format_d1_initial(
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
            except RewardSandboxError as exc:
                # The smoke states cannot cover every dataset frame; a reward
                # that fails on one is rejected like a Stage II training failure.
                raw = Score1Result(
                    score=-1.0,
                    degenerate_fraction=1.0,
                    rejected=True,
                    reject_reason=f"score1_error: {type(exc).__name__}: {exc}"[:300],
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
                if isinstance(raw, Score1Result) and raw.scenario_scores:
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
        prompt = self._format_d2_crossover(
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
        proxy_fb = (parent.metadata or {}).get("proxy_feedback")
        if proxy_fb:
            # Evidence only — LLM diagnoses from numbers in the block.
            weakness = (
                f"Parent {parent.candidate_id} Score1={parent.score}.\n"
                f"{proxy_fb}\n"
                f"Global reflection: {reflection}"
            )
        else:
            weakness = (
                f"Parent {parent.candidate_id} score={parent.score}. "
                f"Weakness focus: improve analytical Score1 relative to elites. "
                f"Global reflection: {reflection}"
            )
        prompt = self._format_d2_mutation(
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
        prompt = self._format_d1_initial(
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
        prompt = self._format_d1_initial_batch(
            needed,
            func_name=self.config.func_name,
            include_seed=True,
            include_external_knowledge=self.config.include_external_knowledge,
            reflection=self.reflection,
        )
        try:
            raw = self._llm_raw(prompt, max_tokens=self._batch_max_tokens(needed))
            codes = split_reward_function_sources(raw, func_name=self.config.func_name)
        except Exception as exc:  # noqa: BLE001
            if not is_recoverable_llm_error(exc):
                raise
            logger.warning(
                "random batch LLM failed (%s): %s; regenerating per-slot",
                type(exc).__name__,
                exc,
            )
            raw = ""
            codes = []
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

    def _build_evidence_reflection(
        self, ranked: Sequence[RewardCandidate], *, generation: int
    ) -> str:
        """Fitness-ordered population table + best-fitness trend + diagnoses."""

        def _num(source: Dict[str, Any], *keys: str) -> float:
            for key in keys:
                if source.get(key) is not None:
                    try:
                        return float(source[key])
                    except (TypeError, ValueError):
                        pass
            return float("nan")

        rows: List[Dict[str, Any]] = []
        best_fitness: Optional[float] = None
        for c in ranked:
            md = c.metadata or {}
            m = md.get("last_metrics")
            if not isinstance(m, dict):
                m = {}
            holdout = m.get("holdout") if isinstance(m.get("holdout"), dict) else m
            fitness = _num(md, "fitness")
            if fitness == fitness and (best_fitness is None or fitness > best_fitness):
                best_fitness = fitness
            rows.append(
                {
                    "id": str(c.candidate_id),
                    "SR": _num(holdout, "SR", "sr"),
                    "CR": _num(holdout, "CR", "cr"),
                    "mean_speed": _num(holdout, "mean_speed", "ITR"),
                    "fitness": fitness,
                    "diagnosis": md.get("llm_diagnosis"),
                }
            )

        if best_fitness is not None:
            self._best_fitness_trend.append(float(best_fitness))
            self._best_fitness_trend = self._best_fitness_trend[-5:]
        trend = (
            f"Gen{generation} best_fitness_trend=["
            + ", ".join(f"{v:.3f}" for v in self._best_fitness_trend)
            + "]"
        )
        table_bits = [
            f"{r['id']}: SR={r['SR']:.2f} CR={r['CR']:.2f} "
            f"mean_speed={r['mean_speed']:.1f} fitness={r['fitness']:.3f}"
            for r in rows
        ]
        table = "Population(fitness desc): " + (
            "; ".join(table_bits) if table_bits else "(empty)"
        )
        diag_bits: List[str] = []
        for r in rows:
            d = r.get("diagnosis")
            if d:
                diag_bits.append(f"{r['id']}: {str(d).strip()}")
            if len(diag_bits) >= 3:
                break
        diag = (
            "PriorLLMDiagnoses: " + " | ".join(diag_bits)
            if diag_bits
            else "PriorLLMDiagnoses: (none)"
        )
        return f"{trend}. {table}. {diag}"

    def _build_reflection(
        self, ranked: Sequence[RewardCandidate], *, generation: int
    ) -> str:
        """Reflective note for the next generation (last three notes kept)."""
        note = self._build_evidence_reflection(ranked, generation=generation)
        if not self.reflection.strip():
            return note
        parts = [p.strip() for p in self.reflection.split(" || ") if p.strip()]
        parts.append(note)
        return " || ".join(parts[-3:])
