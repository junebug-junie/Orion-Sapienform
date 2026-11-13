"""Request-level thinking default for the fast llama.cpp class."""

from __future__ import annotations

from typing import Any


def with_thinking_off_default(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep explicit per-call choices; make an omitted choice unambiguously off."""
    result = dict(payload)
    kwargs = result.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    kwargs.setdefault("enable_thinking", False)
    result["chat_template_kwargs"] = kwargs
    return result
