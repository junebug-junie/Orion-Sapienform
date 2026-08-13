"""
Evaluation of pipeline step `when:` guards from config/vision_profiles.yaml.

Kept in its own module, free of torch/PIL/numpy, so the guard semantics are
testable without importing the whole inference runner.

Guards are repo-controlled YAML, never request data. Even so, expressions are
validated against an AST allowlist rather than handed to a bare `eval` with
emptied builtins -- `{"__builtins__": {}}` is not a sandbox (`().__class__.
__bases__[0].__subclasses__()` walks straight out of it), and a guard language
of "compare a request flag to a constant" does not need more than the allowlist
permits.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Set

from loguru import logger

# Flags a guard may legitimately name without the caller having sent them.
# Anything outside this set is treated as a typo and warned about once, so a
# renamed flag cannot silently disable a pipeline step forever.
KNOWN_GUARD_FLAGS: Set[str] = {
    "is_video",
    "want_caption",
    "want_embeddings",
    "want_masks",
    "want_pose",
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not,
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Attribute, ast.Name, ast.Load, ast.Constant,
)

_warned_unknown_flags: Set[str] = set()


class RequestView:
    """
    Attribute view over a request dict, for use inside a `when:` expression.

    A step's guard names a request flag the caller is not obliged to send --
    `request.is_video`, `request.want_masks`, `request.want_embeddings`.
    Absent means "not requested", so a missing key resolves to None (falsy)
    rather than raising.

    The original implementation wrapped the request in a SimpleNamespace, which
    raises AttributeError on any key the caller omitted. Every optional step in
    every pipeline is guarded on exactly such a key, so each one raised, got
    swallowed as a false, and logged a warning on every task.
    """

    __slots__ = ("_request",)

    def __init__(self, request: Dict[str, Any]) -> None:
        self._request = request

    def __getattr__(self, name: str) -> Any:
        if name not in self._request:
            _warn_unknown_flag_once(name)
        return self._request.get(name)


def _warn_unknown_flag_once(name: str) -> None:
    if name in KNOWN_GUARD_FLAGS or name in _warned_unknown_flags:
        return
    _warned_unknown_flags.add(name)
    logger.warning(
        f"[PIPE] guard names unknown request flag '{name}' -- treating as unset. "
        "If this flag was renamed, the guarded step is now permanently skipped."
    )


def _validate(tree: ast.Expression) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ("request", "true", "false"):
            raise ValueError(f"disallowed name: {node.id}")


def safe_when(expr: str, request: Dict[str, Any]) -> bool:
    """Evaluate a step guard. An empty guard means the step always runs."""
    if not expr:
        return True
    try:
        tree = ast.parse(expr, mode="eval")
        _validate(tree)
        # `true`/`false` are bound as names rather than rewritten into the
        # source. Textual substitution corrupted string literals and any
        # attribute whose name contained the word.
        return bool(
            eval(  # noqa: S307 -- AST-validated above, guards come from repo YAML
                compile(tree, "<when>", "eval"),
                {"__builtins__": {}},
                {"request": RequestView(request), "true": True, "false": False},
            )
        )
    except Exception as e:
        logger.warning(f"[PIPE] when eval failed expr={expr} err={e}")
        return False
