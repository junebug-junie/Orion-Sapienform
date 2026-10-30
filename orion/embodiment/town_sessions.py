"""AI Town session ledger: the cap arithmetic and the Redis I/O (orion/schemas/aitown_session.py).

Pure functions plus three small async Redis helpers. Two readers:

* orion-embodiment's session controller (services/orion-embodiment/app/town_sessions.py): the start
  gate and the stop check.
* orion-durable-runs' physiology report (``wants.town``): what Orion and Juniper spent today.

The cap is per LOCAL day (America/Denver by default). A session that crosses local midnight is
split: the part before midnight counts for the old day, the part after for the new one. Times are
epoch seconds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

from orion.schemas.aitown_session import (
    AITOWN_SESSION_CONFIG_REDIS_KEY,
    AITOWN_SESSION_LEDGER_REDIS_KEY,
    AITownSessionConfigV1,
    AITownSessionRowV1,
)

logger = logging.getLogger("orion.embodiment.town_sessions")

DEFAULT_TZ = "America/Denver"
#: An open row whose last tick is older than this is not counted past its last tick + this.
STALE_SEC = 60.0
#: Rows kept in the ledger (pruned on write).
KEEP_DAYS = 35


def local_day_start(now: float, tz: str = DEFAULT_TZ) -> float:
    """Epoch seconds of local midnight that starts the local day containing ``now``."""
    local = datetime.fromtimestamp(now, tz=ZoneInfo(tz))
    return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def next_local_midnight(now: float, tz: str = DEFAULT_TZ) -> float:
    local = datetime.fromtimestamp(now, tz=ZoneInfo(tz))
    # Wall-clock arithmetic on a ZoneInfo datetime: the offset is recomputed for the new wall time,
    # so a DST change makes this local day 23 or 25 hours long, as it really is.
    return (local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()


def local_date(now: float, tz: str = DEFAULT_TZ) -> str:
    return datetime.fromtimestamp(now, tz=ZoneInfo(tz)).strftime("%Y-%m-%d")


def row_end(row: AITownSessionRowV1, now: float) -> float:
    """Where a row stops counting: its end, else its last tick (+ STALE_SEC grace), never past now."""
    if row.ended_at is not None:
        return row.ended_at
    return min(now, row.last_seen_at + STALE_SEC)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def seconds_today(rows: Iterable[AITownSessionRowV1], now: float, *, kind: str,
                  tz: str = DEFAULT_TZ) -> float:
    """Seconds of ``kind`` sessions inside the local day that contains ``now``, up to ``now``."""
    start = local_day_start(now, tz)
    return sum(_overlap(r.started_at, row_end(r, now), start, now) for r in rows if r.kind == kind)


def open_row(rows: Iterable[AITownSessionRowV1], kind: str) -> Optional[AITownSessionRowV1]:
    opened = [r for r in rows if r.kind == kind and r.ended_at is None]
    return max(opened, key=lambda r: r.started_at) if opened else None


def start_block_reason(rows: Optional[Sequence[AITownSessionRowV1]], now: float, *, enabled: bool,
                       daily_cap_sec: float, min_session_sec: float, min_gap_sec: float,
                       tz: str = DEFAULT_TZ) -> Optional[str]:
    """The named reason an Orion session may not start now, or None. No I/O.

    ``rows`` None means the ledger could not be read: refused ``ledger_unknown`` (the cap is not
    enforced on a guess)."""
    if not enabled:
        return "disabled"
    if rows is None:
        return "ledger_unknown"
    if open_row(rows, "orion") is not None:
        return "session_open"
    if open_row(rows, "manual") is not None:
        return "manual_session_open"
    if daily_cap_sec - seconds_today(rows, now, kind="orion", tz=tz) < min_session_sec:
        return "daily_cap"
    ends = [row_end(r, now) for r in rows if r.kind == "orion"]
    if ends and now - max(ends) < min_gap_sec:
        return "cooldown"
    return None


def retry_after_sec(reason: str, rows: Sequence[AITownSessionRowV1], now: float, *, min_gap_sec: float,
                    tz: str = DEFAULT_TZ) -> Optional[float]:
    """How long a refusal holds, from the gate's own clock. None: no clock (the caller's default)."""
    if reason == "daily_cap":
        return max(1.0, next_local_midnight(now, tz) - now)
    if reason == "cooldown":
        ends = [row_end(r, now) for r in rows if r.kind == "orion"]
        return max(1.0, min_gap_sec - (now - max(ends))) if ends else None
    return None


def stop_reason(row: AITownSessionRowV1, rows: Sequence[AITownSessionRowV1], now: float, *, enabled: bool,
                session_max_sec: float, daily_cap_sec: float, tz: str = DEFAULT_TZ) -> Optional[str]:
    """Why an open Orion session must end now, or None. The daily cap is checked against the local
    day that contains ``now``: a session that crosses midnight starts the new day at 0."""
    if not enabled:
        return "disabled"
    if now - row.started_at >= session_max_sec:
        return "session_cap"
    live = [r for r in rows if r.session_id != row.session_id] + [row.model_copy(update={"last_seen_at": now})]
    if seconds_today(live, now, kind="orion", tz=tz) >= daily_cap_sec:
        return "daily_cap"
    return None


def summarize(rows: Sequence[AITownSessionRowV1], now: float, *, since: float, daily_cap_sec: Optional[float],
              tz: str = DEFAULT_TZ, limit: int = 50) -> dict:
    """The report's ``wants.town`` block: today's use against the cap, and the sessions in the window
    (newest first)."""
    in_window = [r for r in rows if row_end(r, now) >= since]
    in_window.sort(key=lambda r: r.started_at, reverse=True)
    sessions = []
    for r in in_window[:limit]:
        end = row_end(r, now)
        sessions.append({
            "session_id": r.session_id, "kind": r.kind,
            "started_at": datetime.fromtimestamp(r.started_at, tz=timezone.utc),
            "ended_at": datetime.fromtimestamp(r.ended_at, tz=timezone.utc) if r.ended_at is not None else None,
            "seconds": round(max(0.0, end - r.started_at), 1),
            "start_reason": r.start_reason, "stop_reason": r.stop_reason,
            "utterances": r.utterances,
        })
    live = open_row(rows, "orion") or open_row(rows, "manual")
    return {
        "local_date": local_date(now, tz),
        "orion_seconds_today": round(seconds_today(rows, now, kind="orion", tz=tz), 1),
        "manual_seconds_today": round(seconds_today(rows, now, kind="manual", tz=tz), 1),
        "daily_cap_seconds": daily_cap_sec,
        "open_kind": live.kind if live is not None else None,
        "sessions": sessions,
    }


# --- Redis -----------------------------------------------------------------------------------


def _text(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def parse_rows(raw: Any) -> list[AITownSessionRowV1]:
    """HGETALL output -> rows. A malformed row is skipped (logged), never fatal."""
    out: list[AITownSessionRowV1] = []
    for key, value in (raw or {}).items():
        try:
            out.append(AITownSessionRowV1.model_validate(json.loads(_text(value))))
        except Exception:  # noqa: BLE001
            logger.warning("aitown_session_row_malformed field=%s", _text(key)[:80])
    return out


async def load_rows(redis: Any) -> list[AITownSessionRowV1]:
    """All rows. Raises on a Redis failure (the caller decides: the gate refuses, the report says why)."""
    return parse_rows(await redis.hgetall(AITOWN_SESSION_LEDGER_REDIS_KEY))


async def save_row(redis: Any, row: AITownSessionRowV1) -> None:
    await redis.hset(AITOWN_SESSION_LEDGER_REDIS_KEY, row.session_id, row.model_dump_json())


async def save_config(redis: Any, cfg: AITownSessionConfigV1) -> None:
    await redis.set(AITOWN_SESSION_CONFIG_REDIS_KEY, cfg.model_dump_json())


async def load_config(redis: Any) -> Optional[AITownSessionConfigV1]:
    """The controller's limits, or None when it never wrote them (or they do not parse)."""
    raw = await redis.get(AITOWN_SESSION_CONFIG_REDIS_KEY)
    if not raw:
        return None
    try:
        return AITownSessionConfigV1.model_validate(json.loads(_text(raw)))
    except Exception:  # noqa: BLE001
        logger.warning("aitown_session_config_malformed")
        return None


async def prune(redis: Any, rows: Sequence[AITownSessionRowV1], now: float, *, keep_days: int = KEEP_DAYS) -> int:
    """Delete closed rows that ended more than ``keep_days`` ago. Returns how many."""
    old = [r.session_id for r in rows if r.ended_at is not None and now - r.ended_at > keep_days * 86400]
    if old:
        await redis.hdel(AITOWN_SESSION_LEDGER_REDIS_KEY, *old)
    return len(old)


__all__ = [
    "DEFAULT_TZ",
    "KEEP_DAYS",
    "STALE_SEC",
    "load_config",
    "load_rows",
    "local_date",
    "local_day_start",
    "next_local_midnight",
    "open_row",
    "parse_rows",
    "prune",
    "retry_after_sec",
    "row_end",
    "save_config",
    "save_row",
    "seconds_today",
    "start_block_reason",
    "stop_reason",
    "summarize",
]
