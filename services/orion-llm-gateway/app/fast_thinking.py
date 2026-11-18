"""Request defaults for the shared GPU3 llama.cpp worker."""

from __future__ import annotations

from typing import Any

GPU3_WORK_CLASSES = frozenset({"fast", "metacog", "operator_fast", "operator_metacog"})


def with_thinking_off_default(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep explicit per-call choices; make an omitted choice unambiguously off."""
    result = dict(payload)
    kwargs = result.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    if kwargs.get("enable_thinking") is None:
        kwargs["enable_thinking"] = False
    result["chat_template_kwargs"] = kwargs
    return result


def with_gpu3_route_defaults(payload: dict[str, Any], work_class: str) -> dict[str, Any]:
    """Preserve caller choices while separating fast and metacog policy on one worker."""
    if work_class not in GPU3_WORK_CLASSES:
        return dict(payload)
    result = with_thinking_off_default(payload)
    if work_class in {"metacog", "operator_metacog"}:
        defaults = {
            "temperature": 0.35,
            "top_p": 0.90,
            "presence_penalty": 0.20,
            "max_tokens": 4096 if result["chat_template_kwargs"]["enable_thinking"] else 1024,
        }
        for key, value in defaults.items():
            if result.get(key) is None:
                result[key] = value
    return result
