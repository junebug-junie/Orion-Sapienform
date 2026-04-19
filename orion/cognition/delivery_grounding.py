"""Shared helpers for grounding delivery-oriented answers (re-exports answer grounding)."""

from __future__ import annotations

from orion.cognition.answer_grounding import (
    build_answer_grounding_context,
    delivery_grounding_mode,
    extract_trace_preferred_output,
)


def build_delivery_grounding_context(*, user_text: str, output_mode: str | None) -> dict[str, Any]:
    """Backward-compatible name for build_answer_grounding_context."""
    return build_answer_grounding_context(user_text=user_text, output_mode=output_mode)


__all__ = [
    "build_delivery_grounding_context",
    "delivery_grounding_mode",
    "extract_trace_preferred_output",
]
