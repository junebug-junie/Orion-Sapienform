from __future__ import annotations

from datetime import datetime

from app.boundary import legacy_close_reason, should_close_window  # noqa: F401
from app.settings import settings
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


def should_close_by_time_gap(turns: list[dict], *, gap_sec: int) -> bool:
    if len(turns) < 2:
        return False
    last_two = turns[-2:]
    ts_values = []
    for t in last_two:
        ts = t.get("memory_classify_ts") or t.get("created_at")
        if isinstance(ts, str):
            try:
                ts_values.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                continue
    if len(ts_values) < 2:
        return False
    return (ts_values[1] - ts_values[0]).total_seconds() >= gap_sec


def turn_has_phase(turn: MemoryTurnPersistedV1) -> bool:
    phase = (turn.spark_meta.get("conversation_phase") or {}).get("phase_change")
    return bool(phase)


def legacy_close_decision(
    turn: MemoryTurnPersistedV1,
    scores: dict,
    *,
    window_turns: list[dict],
) -> str | None:
    """The live (legacy) window-close decision, as a named reason or None."""
    reason = legacy_close_reason(turn, scores, settings)
    if reason is not None:
        return reason
    if turn_has_phase(turn):
        return None
    if should_close_by_time_gap(window_turns, gap_sec=int(settings.MEMORY_WINDOW_FALLBACK_GAP_SEC)):
        return "legacy:time_gap"
    return None


def should_close_turn(
    turn: MemoryTurnPersistedV1,
    scores: dict,
    *,
    window_turns: list[dict],
) -> bool:
    return legacy_close_decision(turn, scores, window_turns=window_turns) is not None
