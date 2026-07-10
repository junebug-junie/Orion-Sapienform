from __future__ import annotations

from datetime import datetime
from typing import Literal


def build_compactor_index(
    *,
    kind: str,
    mode: Literal["day", "rolling"],
    calendar_date: str | None = None,
    lookback_hours: int | None = None,
    window_start: datetime | None = None,
) -> str:
    """Stable window key for indexed compactor memory cards.

    v1 kind is always ``chat_history_log``. Future kinds (e.g. journal) may
    add filter dimensions to the key — do not invent them here.
    """
    _ = kind  # reserved for future key namespaces; v1 always chat_compactor prefix
    if mode == "day":
        date = (calendar_date or "").strip()
        if not date:
            raise ValueError("calendar_date required for mode=day")
        return f"chat_compactor:day:{date}"
    if mode == "rolling":
        if lookback_hours is None or int(lookback_hours) < 1:
            raise ValueError("lookback_hours required for mode=rolling")
        if window_start is None:
            raise ValueError("window_start required for mode=rolling")
        floored = window_start.replace(second=0, microsecond=0)
        start_iso = floored.isoformat()
        return f"chat_compactor:rolling:{int(lookback_hours)}h:{start_iso}"
    raise ValueError(f"unsupported_compactor_mode:{mode}")
