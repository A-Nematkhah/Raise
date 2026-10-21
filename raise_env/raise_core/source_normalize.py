"""
AST-based normalization of LLM reward source before sandbox validation.

Moved out of ``raise_core.llm`` so the sandbox validator does not import the
LLM client stack. Re-exported from ``raise_core.llm`` for callers.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Sequence, Tuple

# Above the sandbox's max_code_length; anything longer is rejected later anyway.
_MAX_PARSE_CHARS = 50_000

_ALIAS_NAME_RE = re.compile(
    r"^(cal_reward|seed_reward_func|compute_reward_v\d+|[\w]+_v2)$"
)

# Scalar CrowdNavRewardState / RobotState / HumanState fields — never arrays.
# ``action`` is deliberately absent (a tuple in continuous mode).
_SCALAR_FIELD_ATTRS = frozenset(
    {
        "px",
        "py",
        "vx",
        "vy",
        "gx",
        "gy",
        "theta",
        "radius",
        "v_pref",
        "collision",
        "reached_goal",
        "timeout",
        "min_separation",
        "min_separation_post",
        "discomfort_dist",
        "goal_distance",
        "progress",
        "time_step",
        "global_time",
        "time_limit",
        "danger",
        "success_reward",
        "collision_penalty",
        "discomfort_penalty_factor",
        "potential_delta",
        "potential_factor",
        "kinematics",
        "visible",
    }
)


def _safe_parse(text: str) -> Optional[ast.Module]:
    """``ast.parse`` that never raises on hostile LLM output (None instead)."""
    if len(text) > _MAX_PARSE_CHARS:
        return None
    try:
        return ast.parse(text, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _is_trivial_int_index(slice_node: ast.AST) -> bool:
    """True for ``[0]``, ``[1]``, ``[-1]`` (common LLM tensor-habits)."""
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
        return slice_node.value in (0, 1, -1)
    if (
        isinstance(slice_node, ast.UnaryOp)
        and isinstance(slice_node.op, ast.USub)
        and isinstance(slice_node.operand, ast.Constant)
        and isinstance(slice_node.operand.value, int)
        and slice_node.operand.value == 1
    ):
        return True
    return False


class _StripScalarSubscripts(ast.NodeTransformer):
    """Rewrite ``state.robot.px[0]`` → ``state.robot.px`` (and nested repeats)."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node, ast.Subscript):
            raise TypeError(
                f"expected Subscript node in strip_redundant_scalar_subscripts, "
                f"got {type(node).__name__}"
            )
        if not isinstance(node.value, ast.Attribute):
            return node
        if node.value.attr not in _SCALAR_FIELD_ATTRS:
            return node
        if not _is_trivial_int_index(node.slice):
            return node
        return node.value


def strip_redundant_scalar_subscripts(code: str) -> str:
    """
    Drop ``[0]``/``[1]``/``[-1]`` on known float/bool state fields.

    Does **not** touch ``state.action[...]`` (a tuple in continuous mode).
    On parse failure, or when nothing changes, returns ``code`` unchanged so
    LLM comments are preserved (``ast.unparse`` would strip them).
    """
    text = (code or "").strip()
    if not text:
        return text
    tree = _safe_parse(text)
    if tree is None:
        return text
    try:
        before = ast.dump(tree)
        new_tree = _StripScalarSubscripts().visit(tree)
        if ast.dump(new_tree) == before:
            return text
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception:  # noqa: BLE001 — keep original if transform/unparse fails
        return text


def _line_starts(text: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _abs_offset(starts: Sequence[int], line: int, col: int) -> int:
    return starts[line - 1] + col


def _pick_target_function(
    funcs: Sequence[ast.FunctionDef], target_name: str
) -> Optional[ast.FunctionDef]:
    """Choose which top-level def to normalize (never a helper when target exists)."""
    for fn in funcs:
        if fn.name == target_name:
            return fn
    for fn in funcs:
        if _ALIAS_NAME_RE.match(fn.name):
            return fn
    if len(funcs) == 1:
        return funcs[0]
    return None


def _arg_names(fn: ast.FunctionDef) -> List[str]:
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    return [a.arg for a in positional]


def _rewrite_def_line(
    text: str, fn: ast.FunctionDef, target_name: str
) -> Tuple[str, List[str]]:
    """Rename the function and/or add ``memory`` using source positions."""
    repairs: List[str] = []
    starts = _line_starts(text)
    # ``def name(`` span
    name_start = _abs_offset(starts, fn.lineno, fn.col_offset)
    # Skip leading ``def`` + whitespace already at col_offset of FunctionDef
    # (col_offset points at ``def``).
    def_prefix = text[name_start : name_start + 4]
    if not def_prefix.startswith("def"):
        return text, repairs
    # Find the name token after ``def``.
    after_def = name_start + 3
    while after_def < len(text) and text[after_def].isspace():
        after_def += 1
    name_end = after_def
    while name_end < len(text) and (text[name_end].isalnum() or text[name_end] == "_"):
        name_end += 1
    old_name = text[after_def:name_end]
    if old_name != target_name:
        text = text[:after_def] + target_name + text[name_end:]
        name_end = after_def + len(target_name)
        repairs.append("renamed_function")
        starts = _line_starts(text)

    args = _arg_names(fn)
    if args == ["state"]:
        # Replace the argument list ``(state...)`` with ``(state, memory)``.
        # Locate ``(`` after the (possibly renamed) name.
        paren = name_end
        while paren < len(text) and text[paren] != "(":
            paren += 1
        if paren >= len(text) or text[paren] != "(":
            return text, repairs
        depth = 0
        close = paren
        for i in range(paren, len(text)):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        text = text[: paren + 1] + "state, memory" + text[close:]
        repairs.append("added_memory_arg")
    return text, repairs


def normalize_to_compute_reward_with_report(
    code: str, target_name: str = "compute_reward"
) -> Tuple[str, List[str]]:
    """
    Ensure the reward entrypoint is ``compute_reward(state, memory)``.

    Returns ``(normalized_source, repairs)`` where ``repairs`` lists applied
    fixes such as ``renamed_function``, ``added_memory_arg``,
    ``stripped_scalar_subscripts``.
    """
    repairs: List[str] = []
    text = (code or "").strip()
    if not text:
        return text, repairs

    tree = _safe_parse(text)
    if tree is None:
        stripped = strip_redundant_scalar_subscripts(text)
        if stripped != text:
            repairs.append("stripped_scalar_subscripts")
        return stripped, repairs

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    target = _pick_target_function(funcs, target_name)
    if target is not None:
        need_rename = target.name != target_name
        need_memory = _arg_names(target) == ["state"]
        if need_rename or need_memory:
            text, rewrite_repairs = _rewrite_def_line(text, target, target_name)
            repairs.extend(rewrite_repairs)

    stripped = strip_redundant_scalar_subscripts(text)
    if stripped != text:
        repairs.append("stripped_scalar_subscripts")
        text = stripped
    return text, repairs


def normalize_to_compute_reward(code: str, target_name: str = "compute_reward") -> str:
    """Wrapper around :func:`normalize_to_compute_reward_with_report`."""
    return normalize_to_compute_reward_with_report(code, target_name)[0]


__all__ = [
    "normalize_to_compute_reward",
    "normalize_to_compute_reward_with_report",
    "strip_redundant_scalar_subscripts",
]
