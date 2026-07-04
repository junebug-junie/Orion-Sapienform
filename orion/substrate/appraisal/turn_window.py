from __future__ import annotations

from typing import Any, Iterable

from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1

_ALLOWED_ROLES = frozenset({"user", "assistant", "system"})


def build_turn_window(
    messages: Iterable[dict[str, Any] | TurnWindowMessageV1],
    *,
    max_turns: int = 8,
) -> list[TurnWindowMessageV1]:
    """Normalize chat messages[] into a bounded paired turn window."""
    cap = max(1, int(max_turns))
    out: list[TurnWindowMessageV1] = []
    for item in messages:
        if isinstance(item, TurnWindowMessageV1):
            if item.content.strip():
                out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in _ALLOWED_ROLES:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        out.append(TurnWindowMessageV1(role=role, content=content))  # type: ignore[arg-type]
    if len(out) > cap:
        out = out[-cap:]
    return out
