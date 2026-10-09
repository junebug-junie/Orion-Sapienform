from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from orion.cognition.compactor.calendar_day import DEFAULT_COMPACTOR_TIMEZONE, previous_local_day_window


@dataclass(frozen=True)
class ResolvedGithubCompactorWindow:
    mode: Literal["day", "rolling"]
    window_label: str
    window_start: datetime  # UTC
    window_end: datetime  # UTC
    lookback_days: int
    calendar_date: str | None
    timezone_name: str


def resolve_github_compactor_window(
    *,
    workflow_request: dict[str, Any] | None,
    now: datetime,
    lookback_days: int,
) -> ResolvedGithubCompactorWindow:
    """Pick the GitHub compactor window.

    - ``day``: the full previous Denver calendar day (same bounds as the chat
      compactor's scheduled window), filtered on ``merged_at``. Default for a
      scheduled dispatch, so the 06:10 run covers yesterday exactly once.
    - ``rolling``: ``now - lookback_days .. now``. Default for an on-demand run
      (the pre-existing behavior); the label stays today's UTC date.
    """
    req = workflow_request if isinstance(workflow_request, dict) else {}
    mode_raw = str(req.get("window_mode") or "").strip().lower()
    if not mode_raw:
        mode_raw = "day" if req.get("scheduled_dispatch") else "rolling"
    if mode_raw not in ("day", "rolling"):
        raise ValueError(f"unsupported_github_compactor_window_mode:{mode_raw}")
    aware_now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if mode_raw == "day":
        day = previous_local_day_window(aware_now, tz_name=DEFAULT_COMPACTOR_TIMEZONE)
        return ResolvedGithubCompactorWindow(
            mode="day",
            window_label=day.calendar_date,
            window_start=day.window_start,
            window_end=day.window_end,
            lookback_days=max(1, int(lookback_days)),
            calendar_date=day.calendar_date,
            timezone_name=day.timezone_name,
        )
    days = max(1, int(lookback_days))
    end = aware_now.astimezone(timezone.utc)
    return ResolvedGithubCompactorWindow(
        mode="rolling",
        window_label=end.strftime("%Y-%m-%d"),
        window_start=end - timedelta(days=days),
        window_end=end,
        lookback_days=days,
        calendar_date=None,
        timezone_name="UTC",
    )
