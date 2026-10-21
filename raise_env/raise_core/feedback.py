"""
Feedback-card plumbing shared by every domain: config, candidate archive,
lineage-based history selection and structural reward-code summaries.

The archive is a view of evaluated candidates (code, lineage, Stage II
metadata). It is persisted through the per-epoch ``pop_epoch_XXX.json``
files the closed-loop runner already writes, and rebuilt from them on resume.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
from dataclasses import dataclass, field

from raise_core.config_defaults import (
    feedback_component_instrumentation,
    feedback_enabled,
    feedback_history_entries,
    feedback_max_input_tokens,
    feedback_max_trajectories,
    feedback_random_calls,
    feedback_target_input_tokens,
    feedback_training_dynamics,
)
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Candidate metadata the card reads (all written by Stage I / Stage II).
ARCHIVED_METADATA_KEYS = (
    "last_metrics",
    "fitness",
    "train_steps",
    "eval_episodes",
    "eval_deterministic",
    "training_dynamics",
    "llm_diagnosis",
    "llm_diagnosis_subjects",
    "score1_scenario_scores",
    "score1_n_pairs",
    "score1_rejected",
    "score1_reject_reason",
    "score1_raw",
    "llm_fallback_clone",
    "eval_generation",
    "fitness_history",
    "checkpoint_path",
    "train_wall_seconds",
    "train_steps_per_second",
    "train_seed",
    "ppo_n_steps",
    "eval_seed_base",
    "eval_split",
    "actual_train_steps",
    "behavior_fp",
)
_POP_EPOCH_RE = re.compile(r"pop_epoch_(\d+)\.json$")


@dataclass
class FeedbackCardConfig:
    """Diagnostic feedback card + LLM input budget (budget applies even when disabled)."""

    enabled: bool = field(default_factory=feedback_enabled)
    max_input_tokens: int = field(default_factory=feedback_max_input_tokens)
    target_input_tokens: int = field(default_factory=feedback_target_input_tokens)
    history_entries: int = field(default_factory=feedback_history_entries)
    max_trajectories: int = field(default_factory=feedback_max_trajectories)
    # Random restarts get a "reference" card of the best evaluated genome;
    # off keeps them free of it (exploration not pulled towards the best).
    random_calls: bool = field(default_factory=feedback_random_calls)
    # Stage II instrumentation feeding the card (no reward / PPO change).
    training_dynamics: bool = field(default_factory=feedback_training_dynamics)
    component_instrumentation: bool = field(
        default_factory=feedback_component_instrumentation
    )

    def __post_init__(self) -> None:
        if int(self.max_input_tokens) < 1:
            raise ValueError("max_input_tokens must be >= 1")
        if not 1 <= int(self.target_input_tokens) <= int(self.max_input_tokens):
            raise ValueError("target_input_tokens must be in [1, max_input_tokens]")
        if int(self.history_entries) < 0 or int(self.max_trajectories) < 0:
            raise ValueError("history_entries and max_trajectories must be >= 0")


@dataclass
class ArchiveRecord:
    """One evaluated (or failed) candidate as the card sees it."""

    candidate_id: str
    generation: int
    origin: str = "unknown"
    parent_ids: Tuple[str, ...] = ()
    code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> Dict[str, Any]:
        m = self.metadata.get("last_metrics")
        return m if isinstance(m, dict) else {}

    @property
    def evaluated(self) -> bool:
        return self.metadata.get("fitness") is not None and bool(self.metrics)

    @classmethod
    def from_candidate(cls, cand: Any, *, generation: int) -> "ArchiveRecord":
        md = getattr(cand, "metadata", None) or {}
        return cls(
            candidate_id=str(cand.candidate_id),
            generation=int(generation),
            origin=str(getattr(cand, "origin", "unknown")),
            parent_ids=tuple(str(p) for p in (getattr(cand, "parent_ids", ()) or ())),
            code=str(getattr(cand, "code", "") or ""),
            metadata={k: md[k] for k in ARCHIVED_METADATA_KEYS if md.get(k) is not None},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "origin": self.origin,
            "parent_ids": list(self.parent_ids),
            "code": self.code,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, generation: int) -> "ArchiveRecord":
        return cls(
            candidate_id=str(d.get("candidate_id")),
            generation=int(d.get("generation", generation)),
            origin=str(d.get("origin") or "unknown"),
            parent_ids=tuple(str(p) for p in (d.get("parent_ids") or ())),
            code=str(d.get("code") or ""),
            metadata={k: d[k] for k in ARCHIVED_METADATA_KEYS if d.get(k) is not None},
        )


class CandidateArchive:
    """Every candidate seen by the closed loop, keyed by id (first generation wins)."""

    def __init__(self) -> None:
        self._records: Dict[str, ArchiveRecord] = {}

    def __contains__(self, candidate_id: object) -> bool:
        return str(candidate_id) in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, candidate_id: str) -> Optional[ArchiveRecord]:
        return self._records.get(str(candidate_id))

    def records(self) -> List[ArchiveRecord]:
        return list(self._records.values())

    def add(self, record: ArchiveRecord) -> None:
        old = self._records.get(record.candidate_id)
        if old is None:
            self._records[record.candidate_id] = record
        elif record.evaluated:
            # Carried genomes are evaluated again each generation: latest wins.
            old.metadata = dict(record.metadata)

    def add_population(self, candidates: Iterable[Any], *, generation: int) -> None:
        for cand in candidates:
            self.add(ArchiveRecord.from_candidate(cand, generation=generation))

    @classmethod
    def from_epoch_files(cls, directory: str) -> "CandidateArchive":
        """Rebuild from ``pop_epoch_XXX.json`` files (resume)."""
        archive = cls()
        paths = []
        for path in glob.glob(os.path.join(directory, "pop_epoch_*.json")):
            match = _POP_EPOCH_RE.search(path)
            if match:
                paths.append((int(match.group(1)), path))
        for epoch, path in sorted(paths):
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            for row in payload.get("candidates") or []:
                if isinstance(row, dict) and row.get("candidate_id"):
                    archive.add(ArchiveRecord.from_dict(row, generation=epoch))
        return archive

    # ------------------------------------------------------------ lineage

    def ancestors(self, candidate_id: str) -> List[Tuple[ArchiveRecord, int]]:
        """Archived ancestors with their distance (1 = parent), nearest first."""
        out: List[Tuple[ArchiveRecord, int]] = []
        seen = {str(candidate_id)}
        start = self.get(candidate_id)
        frontier = list(start.parent_ids) if start else []
        depth = 1
        while frontier:
            nxt: List[str] = []
            for pid in frontier:
                if pid in seen:
                    continue
                seen.add(pid)
                rec = self.get(pid)
                if rec is None:
                    continue
                out.append((rec, depth))
                nxt.extend(rec.parent_ids)
            frontier = nxt
            depth += 1
        return out

    def children_of(self, candidate_id: str) -> List[ArchiveRecord]:
        cid = str(candidate_id)
        return [r for r in self._records.values() if cid in r.parent_ids]

    def select_history(
        self,
        subject_ids: Sequence[str],
        n: int,
        *,
        exclude: Iterable[str] = (),
    ) -> List[Tuple[ArchiveRecord, str]]:
        """
        Up to ``n`` evaluated relatives of the subjects, as ``(record, relation)``.

        Relatives are interleaved across ancestors beyond the parents (nearest
        first), earlier children of a subject (what previous edits did) and
        siblings sharing a parent (newest first). Only when relatives run out
        are the best-fitness candidates of the latest generation added.
        """
        if n <= 0:
            return []
        taken = {str(x) for x in exclude} | {str(s) for s in subject_ids}
        ancestors: List[Tuple[ArchiveRecord, str]] = []
        children: List[Tuple[ArchiveRecord, str]] = []
        siblings: List[Tuple[ArchiveRecord, str]] = []
        for sid in subject_ids:
            for rec, depth in self.ancestors(sid):
                if depth >= 2:
                    label = "grandparent" if depth == 2 else f"ancestor ({depth} generations up)"
                    ancestors.append((rec, f"{label} of {sid}"))
            for rec in sorted(self.children_of(sid), key=lambda r: -r.generation):
                children.append((rec, f"earlier child of {sid}"))
            subject = self.get(sid)
            for pid in subject.parent_ids if subject else ():
                for rec in sorted(self.children_of(pid), key=lambda r: -r.generation):
                    siblings.append((rec, f"sibling of {sid} (shares parent {pid})"))

        out: List[Tuple[ArchiveRecord, str]] = []
        queues = [ancestors, children, siblings]
        while len(out) < n and any(queues):
            for queue in queues:
                while queue:
                    rec, relation = queue.pop(0)
                    if rec.candidate_id in taken or not rec.evaluated:
                        continue
                    taken.add(rec.candidate_id)
                    out.append((rec, relation))
                    break
                if len(out) >= n:
                    break
        if len(out) < n:
            pool = [r for r in self._records.values() if r.evaluated and r.candidate_id not in taken]
            pool.sort(key=lambda r: (-r.generation, -_fitness(r)))
            for rec in pool[: n - len(out)]:
                out.append((rec, "non-relative (recent, high fitness)"))
        return out


def _fitness(record: ArchiveRecord) -> float:
    try:
        return float(record.metadata.get("fitness"))
    except (TypeError, ValueError):
        return float("-inf")


# ---------------------------------------------------------------- code summary


@dataclass(frozen=True)
class CodeStructure:
    components: Tuple[str, ...]
    constants: Tuple[Tuple[str, float], ...]
    n_lines: int
    parsed: bool


def _numeric_literal(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and _numeric_literal(node.operand) is not None
    ):
        value = _numeric_literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def code_structure(code: str) -> CodeStructure:
    """Component keys (dict-literal string keys) and top-level numeric locals."""
    text = str(code or "")
    n_lines = len([ln for ln in text.splitlines() if ln.strip()])
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return CodeStructure((), (), n_lines, False)
    components: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value not in components:
                        components.append(key.value)
    constants: Dict[str, float] = {}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for stmt in funcs[0].body if funcs else ():
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            value = _numeric_literal(stmt.value)
            if value is not None and stmt.targets[0].id not in constants:
                constants[stmt.targets[0].id] = value
    return CodeStructure(tuple(components), tuple(constants.items()), n_lines, True)


def component_diversity(codes: Sequence[str]) -> Optional[float]:
    """Mean pairwise Jaccard distance of the reward-component sets (0 = same).

    Two codes without any component count as identical; ``None`` below 2 codes.
    """
    sets = [set(code_structure(code).components) for code in codes]
    distances = [
        1.0 - len(a & b) / len(a | b) if a | b else 0.0
        for i, a in enumerate(sets)
        for b in sets[i + 1 :]
    ]
    return sum(distances) / len(distances) if distances else None


def _num(v: float) -> str:
    return f"{v:g}"


def summarize_code(code: str, *, max_constants: int = 6) -> str:
    s = code_structure(code)
    if not s.parsed:
        return f"{s.n_lines} lines (unparseable)"
    comps = ", ".join(s.components) if s.components else "none"
    consts = ", ".join(f"{k}={_num(v)}" for k, v in s.constants[:max_constants])
    extra = len(s.constants) - max_constants
    if extra > 0:
        consts += f", +{extra} more"
    return f"{s.n_lines} lines; components [{comps}]; constants [{consts or 'none'}]"


def diff_code(old_code: str, new_code: str, *, max_items: int = 6) -> str:
    """Structural change from ``old_code`` to ``new_code`` (components, constants)."""
    if str(old_code or "").strip() == str(new_code or "").strip():
        return "identical code"
    old, new = code_structure(old_code), code_structure(new_code)
    items: List[str] = []
    items += [f"+component {c}" for c in new.components if c not in old.components]
    items += [f"-component {c}" for c in old.components if c not in new.components]
    old_c, new_c = dict(old.constants), dict(new.constants)
    for k, v in new_c.items():
        if k not in old_c:
            items.append(f"+{k}={_num(v)}")
        elif old_c[k] != v:
            items.append(f"{k} {_num(old_c[k])}->{_num(v)}")
    items += [f"-{k}" for k in old_c if k not in new_c]
    if not items:
        return f"expression edits only ({old.n_lines}->{new.n_lines} lines)"
    extra = len(items) - max_items
    text = "; ".join(items[:max_items])
    return text + (f"; +{extra} more" if extra > 0 else "")


__all__ = [
    "ARCHIVED_METADATA_KEYS",
    "ArchiveRecord",
    "CandidateArchive",
    "CodeStructure",
    "FeedbackCardConfig",
    "code_structure",
    "diff_code",
    "summarize_code",
]
