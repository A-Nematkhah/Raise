"""
AST / structural policy for LLM-generated reward code.

Ported from mobile_robot_env.rewards.sandbox.ast_policy.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from raise_core.sandbox.config import (
    SAFE_MATH_ATTRIBUTES,
    SAFE_MATH_CONSTANTS,
    SAFE_METHOD_ATTRIBUTES,
    SandboxConfig,
)

_IMPORT_HINT = (
    "math is pre-injected as name `math` - do not import it; call e.g. "
    "math.sqrt / math.exp / math.tanh directly; "
    "never use __import__/getattr/hasattr"
)
from raise_core.sandbox.errors import RewardSandboxError


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.reasons: List[str] = []
        self._allowed_attrs: Optional[set[str]] = None
        if config.allowed_attributes is not None:
            self._allowed_attrs = set(config.allowed_attributes) | set(
                SAFE_METHOD_ATTRIBUTES
            )

    def visit_Import(self, node: ast.Import) -> None:
        if not self.config.allow_imports:
            self.reasons.append(f"import is forbidden ({_IMPORT_HINT})")
        else:
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in self.config.allowed_modules:
                    self.reasons.append(f"import is forbidden ({alias.name})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.config.allow_imports:
            self.reasons.append(f"from-import is forbidden ({_IMPORT_HINT})")
        else:
            root = (node.module or "").split(".", 1)[0]
            if not root or root not in self.config.allowed_modules:
                self.reasons.append(f"from-import is forbidden ({node.module or '*'})")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if not self.config.allow_while:
            self.reasons.append("while loops are forbidden")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.reasons.append(
            "list comprehensions are forbidden "
            "(use an explicit for-loop; unbounded comps are a DoS risk)"
        )
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.reasons.append(
            "set comprehensions are forbidden "
            "(use an explicit for-loop; unbounded comps are a DoS risk)"
        )
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.reasons.append(
            "dict comprehensions are forbidden "
            "(use an explicit for-loop; unbounded comps are a DoS risk)"
        )
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.reasons.append(
            "generator expressions are forbidden "
            "(use an explicit for-loop; unbounded comps are a DoS risk)"
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.reasons.append(f"class definitions are forbidden ({node.name})")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.reasons.append("async functions are forbidden")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self.reasons.append("await is forbidden")
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.reasons.append("yield is forbidden")
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.reasons.append("yield from is forbidden")
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.reasons.append("with statements are forbidden")
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.reasons.append("async with is forbidden")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.reasons.append("global is forbidden")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.reasons.append("nonlocal is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.reasons.append(f"attribute {node.attr!r} is forbidden")
        elif isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr not in SAFE_MATH_ATTRIBUTES:
                self.reasons.append(
                    f"math.{node.attr} is not allowed "
                    f"(allowed: {', '.join(SAFE_MATH_ATTRIBUTES)})"
                )
        elif self._allowed_attrs is not None and node.attr not in self._allowed_attrs:
            self.reasons.append(
                f"attribute {node.attr!r} is not on the domain allowlist "
                f"(hallucinated field?). Use only documented state/ego/others "
                f"fields and memory dict methods (get/items/...)."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.config.forbidden_names or node.id.startswith("__"):
            self.reasons.append(f"name {node.id!r} is forbidden")
        self.generic_visit(node)


def parse_reward_code(code: str, config: SandboxConfig) -> ast.Module:
    if not code or not code.strip():
        raise RewardSandboxError("source code is empty")
    if len(code) > config.max_code_length:
        raise RewardSandboxError(
            f"source code length {len(code)} exceeds max_code_length {config.max_code_length}"
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise RewardSandboxError(f"syntax error: {exc}") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        raise RewardSandboxError(
            f"unparseable source ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(tree, ast.Module):
        raise RewardSandboxError("code must be a module of statements")
    return tree


def check_structure(tree: ast.Module, config: SandboxConfig) -> None:
    visitor = _PolicyVisitor(config)
    try:
        visitor.visit(tree)
    except RecursionError as exc:
        raise RewardSandboxError("expression nesting is too deep") from exc
    if visitor.reasons:
        raise RewardSandboxError("; ".join(visitor.reasons))


def check_interface(tree: ast.Module, config: SandboxConfig) -> ast.FunctionDef:
    """Require exactly one top-level function: compute_reward(state, memory)."""
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    other = [
        node
        for node in tree.body
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.Pass,
                ast.Import,
                ast.ImportFrom,
            ),
        )
    ]
    if other:
        kinds = sorted({type(node).__name__ for node in other})
        raise RewardSandboxError(f"unsupported top-level statements: {', '.join(kinds)}")
    if len(functions) != 1:
        raise RewardSandboxError(
            f"expected exactly one top-level function {config.required_function_name}(), "
            f"found {len(functions)}"
        )
    fn = functions[0]
    if fn.name != config.required_function_name:
        raise RewardSandboxError(
            f"top-level function must be named {config.required_function_name!r}, got {fn.name!r}"
        )
    if fn.args.vararg is not None or fn.args.kwarg is not None:
        raise RewardSandboxError("compute_reward must not use *args or **kwargs")
    if fn.args.kwonlyargs:
        raise RewardSandboxError("compute_reward must not use keyword-only arguments")
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    expected = (config.required_arg_name, config.required_memory_arg_name)
    if len(positional) != 2:
        raise RewardSandboxError(
            f"compute_reward must take exactly two arguments {expected}, "
            f"found {len(positional)}"
        )
    for i, name in enumerate(expected):
        if positional[i].arg != name:
            raise RewardSandboxError(
                f"compute_reward argument {i + 1} must be named {name!r}, "
                f"got {positional[i].arg!r}"
            )
    if fn.args.defaults:
        raise RewardSandboxError("compute_reward must not use default argument values")
    if config.restrict_top_level:
        check_top_level_constants(tree)
    return fn


_CONSTANT_EXPR_NODES = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.UnaryOp,
    ast.BinOp,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.unaryop,
    ast.operator,
)


def _is_math_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
        and node.attr in SAFE_MATH_CONSTANTS
    )


def check_top_level_constants(tree: ast.Module) -> None:
    """Module-level code may only bind constant expressions (no import-time work)."""
    for node in tree.body:
        if isinstance(node, ast.Expr):
            if not (
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            ):
                raise RewardSandboxError(
                    "top-level expressions are forbidden (only a docstring is allowed)"
                )
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(isinstance(t, ast.Name) for t in targets):
                raise RewardSandboxError(
                    "top-level assignments must bind plain names (NAME = constant)"
                )
            if node.value is None:
                continue
            for sub in ast.walk(node.value):
                if _is_math_constant(sub):
                    continue
                if isinstance(sub, ast.Name) and sub.id == "math":
                    continue
                if not isinstance(sub, _CONSTANT_EXPR_NODES):
                    raise RewardSandboxError(
                        "top-level assignments may only bind constant expressions "
                        f"(found {type(sub).__name__}); move logic into compute_reward"
                    )
