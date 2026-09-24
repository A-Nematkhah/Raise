"""
AST / structural policy for LLM-generated reward code.

Ported from mobile_robot_env.rewards.sandbox.ast_policy.
"""

from __future__ import annotations

import ast
from typing import List

from raise_core.sandbox.config import SandboxConfig
from raise_core.sandbox.errors import RewardSandboxError


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.reasons: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        if not self.config.allow_imports:
            self.reasons.append(
                "import is forbidden (math is pre-injected as name `math` - "
                "do not import; prefer ** 0.5 for square roots; "
                "never use __import__/getattr/hasattr)"
            )
        else:
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in self.config.allowed_modules:
                    self.reasons.append(f"import is forbidden ({alias.name})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.config.allow_imports:
            self.reasons.append(
                "from-import is forbidden (math is pre-injected as name `math` - "
                "do not import; prefer ** 0.5 for square roots; "
                "never use __import__/getattr/hasattr)"
            )
        else:
            root = (node.module or "").split(".", 1)[0]
            if not root or root not in self.config.allowed_modules:
                self.reasons.append(f"from-import is forbidden ({node.module or '*'})")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if not self.config.allow_while:
            self.reasons.append("while loops are forbidden")
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
    if not isinstance(tree, ast.Module):
        raise RewardSandboxError("code must be a module of statements")
    return tree


def check_structure(tree: ast.Module, config: SandboxConfig) -> None:
    visitor = _PolicyVisitor(config)
    visitor.visit(tree)
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
    return fn
