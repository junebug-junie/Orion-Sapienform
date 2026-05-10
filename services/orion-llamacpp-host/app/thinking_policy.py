from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .profiles import LlamaCppConfig


def _enable_thinking_from_kwargs(kwargs: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not kwargs:
        return None
    if "enable_thinking" not in kwargs:
        return None
    return bool(kwargs["enable_thinking"])


@dataclass(frozen=True)
class ThinkingLaunchPolicy:
    """Launch-time only; per-request overrides are v2 (gateway)."""

    intent_label: str  # e.g. "off_kwargs", "on_kwargs", "explicit_budget_0", "default"
    effective_reasoning_budget: Optional[int]  # value to pass to append_flag if not None
    require_jinja: bool  # OR with reasoning-format path in main


def resolve_thinking_launch_policy(
    cfg: LlamaCppConfig,
    supported_flags: Optional[Set[str]],
) -> ThinkingLaunchPolicy:
    kwargs = cfg.chat_template_kwargs
    et = _enable_thinking_from_kwargs(kwargs)

    has_budget_flag = supported_flags is None or "--reasoning-budget" in supported_flags

    explicit_budget = cfg.reasoning_budget
    if explicit_budget is not None:
        b = int(explicit_budget)
        # Budget 0 needs jinja for Qwen3 (PR #13771 thread); any chat_template_kwargs emission uses jinja path.
        need_jinja = b == 0 or kwargs is not None
        return ThinkingLaunchPolicy(
            intent_label="explicit_budget",
            effective_reasoning_budget=b,
            require_jinja=need_jinja,
        )

    if et is False and has_budget_flag:
        return ThinkingLaunchPolicy(
            intent_label="off_kwargs_implicit_budget",
            effective_reasoning_budget=0,
            require_jinja=True,
        )

    if et is False and not has_budget_flag:
        return ThinkingLaunchPolicy(
            intent_label="off_kwargs_no_budget_flag",
            effective_reasoning_budget=None,
            require_jinja=kwargs is not None,  # kwargs still need jinja when emitted
        )

    if et is True:
        return ThinkingLaunchPolicy(
            intent_label="on_kwargs",
            effective_reasoning_budget=None,
            require_jinja=kwargs is not None,
        )

    return ThinkingLaunchPolicy(
        intent_label="default",
        effective_reasoning_budget=None,
        require_jinja=kwargs is not None,
    )
