"""Dataclass defaults for budgets/env must come from raise_config (via factories)."""

from __future__ import annotations

import ast
from pathlib import Path

_RAISE_ENV = Path(__file__).resolve().parents[2]

WATCHED = [
    _RAISE_ENV / "raise_core" / "pipeline.py",
    _RAISE_ENV / "raise_core" / "raise_loop" / "config.py",
    _RAISE_ENV / "raise_core" / "refine.py",
    _RAISE_ENV / "raise_core" / "validate.py",
    _RAISE_ENV / "raise_core" / "explore.py",
    _RAISE_ENV / "raise_core" / "evo_selection" / "config.py",
    _RAISE_ENV / "raise_core" / "sandbox" / "config.py",
    _RAISE_ENV / "domains" / "crowdnav" / "settings.py",
]

FORBIDDEN = {12000, 70000, 100, 500, 6, 4, 8, 10, 0.25}


def _field_default_constants(class_node: ast.ClassDef) -> list[tuple[int, object]]:
    out: list[tuple[int, object]] = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Constant):
            out.append((stmt.lineno, stmt.value.value))
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            continue
    return out


def test_watched_dataclasses_have_no_budget_literals():
    violations: list[str] = []
    for path in WATCHED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for lineno, val in _field_default_constants(node):
                if val in FORBIDDEN:
                    violations.append(
                        f"{path.name}:{lineno}: {node.name} default literal {val!r}"
                    )
    assert not violations, "\n".join(violations)
