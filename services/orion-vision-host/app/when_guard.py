"""
Evaluation of pipeline step `when:` guards from config/vision_profiles.yaml.

Kept in its own module, free of torch/PIL/numpy, so the guard semantics are
testable without importing the whole inference runner.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from loguru import logger


class RequestView:
    """
    Attribute view over a request dict, for use inside a `when:` expression.

    A step's guard names a request flag the caller is not obliged to send --
    `request.is_video`, `request.want_masks`, `request.want_embeddings`.
    Absent means "not requested", so a missing key resolves to None (falsy)
    rather than raising.

    The previous implementation wrapped the request in a SimpleNamespace, which
    raises AttributeError on any key the caller omitted. Every optional step in
    every pipeline is guarded on exactly such a key, so each one raised, got
    swallowed as a false, and logged a warning on every task.
    """

    __slots__ = ("_request",)

    def __init__(self, request: Dict[str, Any]) -> None:
        object.__setattr__(self, "_request", request)

    def __getattr__(self, name: str) -> Any:
        return self._request.get(name)


def safe_when(expr: str, request: Dict[str, Any]) -> bool:
    """Evaluate a step guard. Empty guard means the step always runs."""
    if not expr:
        return True
    # Whole-word only: a bare str.replace would corrupt a key like `want_true_color`.
    expr = re.sub(r"\btrue\b", "True", expr)
    expr = re.sub(r"\bfalse\b", "False", expr)
    try:
        return bool(eval(expr, {"__builtins__": {}}, {"request": RequestView(request)}))
    except Exception as e:
        logger.warning(f"[PIPE] when eval failed expr={expr} err={e}")
        return False
