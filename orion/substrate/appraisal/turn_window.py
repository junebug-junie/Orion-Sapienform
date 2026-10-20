from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1

_ALLOWED_ROLES = frozenset({"user", "assistant", "system"})


def build_turn_window(
    messages: Iterable[dict[str, Any] | TurnWindowMessageV1],
    *,
    max_turns: int = 8,
    keep_timestamps: bool = False,
) -> list[TurnWindowMessageV1]:
    """Normalize chat messages[] into a bounded paired turn window.

    ``keep_timestamps`` copies a message dict's ``ts`` (ISO-8601 string or
    datetime; naive means UTC) onto ``TurnWindowMessageV1.at``. Off by
    default so the pre-turn appraisal request stays unchanged; the harness
    recent-turns path turns it on. A missing or unparseable ``ts`` leaves
    ``at`` as None -- never invented.
    """
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
        at = _parse_ts(item.get("ts")) if keep_timestamps else None
        out.append(TurnWindowMessageV1(role=role, content=content, at=at))  # type: ignore[arg-type]
    if len(out) > cap:
        out = out[-cap:]
    return out


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
