from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from orion.cognition.chat_history_compactor.constants import (
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_TIMEZONE,
)
from orion.cognition.compactor.index import build_compactor_index
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1


@dataclass(frozen=True)
class ResolvedChatCompactorWindow:
    mode: Literal["day", "rolling"]
    compactor_index: str
    window_start: datetime
    window_end: datetime
    lookback_seconds: int
    lookback_hours: int | None
    calendar_date: str | None
    timezone_name: str


_HOURS_RE = re.compile(
    r"\blast\s+(\d+)\s+hours?\b|\b(\d+)\s*h(?:our)?s?\b",
    re.IGNORECASE,
)


def _parse_lookback_hours_from_text(user_text: str) -> int | None:
    m = _HOURS_RE.search(user_text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        hours = int(raw)
    except (TypeError, ValueError):
        return None
    if hours < 1:
        return None
    return min(hours, 24 * 14)


def resolve_chat_compactor_window(
    *,
    window_mode: str | None,
    lookback_hours: int | None,
    now: datetime,
    user_text: str,
    workflow_request: dict[str, Any],
) -> ResolvedChatCompactorWindow:
    tz_name = DEFAULT_TIMEZONE
    tz = ZoneInfo(tz_name)
    now_local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=timezone.utc).astimezone(tz)

    req = workflow_request if isinstance(workflow_request, dict) else {}
    mode_raw = (window_mode or req.get("window_mode") or "").strip().lower()
    if not mode_raw:
        mode_raw = "day" if req.get("scheduled_dispatch") else "rolling"
    if mode_raw not in ("day", "rolling"):
        raise ValueError(f"unsupported_chat_compactor_window_mode:{mode_raw}")

    hours_raw = lookback_hours if lookback_hours is not None else req.get("lookback_hours")
    hours: int | None = None
    if hours_raw is not None:
        try:
            hours = max(1, min(int(hours_raw), 24 * 14))
        except (TypeError, ValueError):
            hours = None
    if hours is None:
        hours = _parse_lookback_hours_from_text(user_text)

    if mode_raw == "day":
        yesterday = now_local.date() - timedelta(days=1)
        start_local = datetime.combine(yesterday, time.min, tzinfo=tz)
        end_local = datetime.combine(yesterday, time.max, tzinfo=tz)
        calendar_date = yesterday.isoformat()
        index = build_compactor_index(
            kind="chat_history_log",
            mode="day",
            calendar_date=calendar_date,
        )
        lookback_seconds = int((end_local - start_local).total_seconds()) + 1
        return ResolvedChatCompactorWindow(
            mode="day",
            compactor_index=index,
            window_start=start_local.astimezone(timezone.utc),
            window_end=end_local.astimezone(timezone.utc),
            lookback_seconds=lookback_seconds,
            lookback_hours=None,
            calendar_date=calendar_date,
            timezone_name=tz_name,
        )

    # rolling (default for on-demand)
    hours = hours or DEFAULT_LOOKBACK_HOURS
    end_local = now_local
    start_local = end_local - timedelta(hours=hours)
    index = build_compactor_index(
        kind="chat_history_log",
        mode="rolling",
        lookback_hours=hours,
        window_start=start_local,
    )
    return ResolvedChatCompactorWindow(
        mode="rolling",
        compactor_index=index,
        window_start=start_local.astimezone(timezone.utc),
        window_end=end_local.astimezone(timezone.utc),
        lookback_seconds=max(1, int(hours * 3600)),
        lookback_hours=hours,
        calendar_date=None,
        timezone_name=tz_name,
    )


def _is_workflow_notification_response(response: str) -> bool:
    """True for a turn whose response is our own workflow-status text, not organic
    conversation — e.g. a Skill Runner trigger like "Compact the last 6 hours of
    chat..." replied to with the templated "Workflow: ...\\nStatus: ..." block from
    `_workflow_summary_text`. Left in the digest window, these turns make the
    compactor recursively summarize its own (or another workflow's) invocations
    instead of what was actually discussed.

    Matches only the exact literal prefixes cortex-orch's workflow_runtime emits,
    so organic chat that happens to mention "workflow" cannot false-positive.
    """
    lines = (response or "").strip().split("\n")
    if not lines:
        return False
    first = lines[0]
    if first.startswith("Workflow request accepted:"):
        return True
    if first.strip() == "Workflow failed before completion.":
        return True
    if first.startswith("Workflow: ") and len(lines) > 1 and lines[1].startswith("Status: "):
        return True
    # execute_workflow_schedule_management's summary, e.g.
    # "Workflow schedule cancel: ok" / "Workflow schedule list: failed".
    if first.startswith("Workflow schedule "):
        return True
    # main.py's outer-exception fallback when a matched workflow errors before
    # dispatch, e.g. "Workflow 'chat_history_compactor_pass' failed and was not
    # replaced with chat_general."
    if first.startswith("Workflow '") and "failed and was not replaced with chat_general" in first:
        return True
    return False


def _rebuild_transcript_text(turns: list[DiscussionWindowTurnV1]) -> str:
    lines: list[str] = []
    for t in turns:
        ts = t.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        head = f"[{ts}]"
        if t.source:
            head += f" source={t.source}"
        if t.correlation_id:
            head += f" correlation_id={t.correlation_id}"
        lines.append(f"{head}\nUser: {t.prompt.strip()}\nAssistant: {t.response.strip()}\n")
    return "\n".join(lines).strip()


def exclude_workflow_notification_turns(window: DiscussionWindowResultV1) -> DiscussionWindowResultV1:
    """Drop turns that are workflow trigger/status round-trips rather than organic
    conversation, so the chat digest reflects what was actually discussed."""
    kept = [t for t in window.turns if not _is_workflow_notification_response(t.response)]
    if len(kept) == len(window.turns):
        return window
    return window.model_copy(
        update={
            "turns": kept,
            "turn_count": len(kept),
            "transcript_text": _rebuild_transcript_text(kept),
        }
    )
