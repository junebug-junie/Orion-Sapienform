"""Derive the per-turn introspect binding from runtime facts, never from the model."""
from __future__ import annotations

import os

from orion.fcc.github_repo_context import harness_mcp_enabled
from orion.schemas.introspect import IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolBindingV1

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in _TRUTHY


def introspect_enabled() -> bool:
    return harness_mcp_enabled() and _env_truthy("HARNESS_FCC_INTROSPECT_ENABLED")


def outward_tools_attached() -> bool:
    """True when this turn can speak to someone other than Juniper (AI Town today)."""
    return _env_truthy("HARNESS_AITOWN_ENABLED")


def introspect_binding_for_turn(
    reading_binding: ReadingToolBindingV1 | None, *, reading_only: bool = False,
) -> IntrospectToolBindingV1 | None:
    if reading_only or reading_binding is None or not introspect_enabled():
        return None
    return IntrospectToolBindingV1(
        invocation_context=reading_binding.invocation_context,
        parent_run_id=reading_binding.parent_run_id,
        parent_trace_id=reading_binding.parent_trace_id,
        memory_allowed=not outward_tools_attached(),
    )
