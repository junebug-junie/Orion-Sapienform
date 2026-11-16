"""Juniper's reply base rate for an outreach act's time of day (issue #3392, option A).

An outreach act is graded ``juniper_replied``: 1 if Juniper starts a chat turn within the act's
horizon (10 h), else 0. Most of that grade is her daily rhythm, not the message (live 2026-11-16:
the raw grade runs 0.30 for acts opened 00:00-06:00 Denver and 0.68 for 12:00-18:00). This module
gives the expectation to grade against: on how many of the previous ``BASE_RATE_DAYS`` days did
she start a turn in the same window of local clock time, whether or not Orion sent anything.

The ledger stamps that rate as the act's ``predicted`` at open, so ``error = outcome - predicted``
reads "more or less than usual for this time of day"; 0 means "as usual". The outcome row in
``substrate_action_outcomes`` gets it as ``baseline`` (its ``baseline_bin`` is the decile, the same
fixed-width bins as orion/autonomy/contrast.py) and ``observed_delta = outcome - baseline``.
``outcome_value`` stays the plain 0/1, so Juniper's vote in orion/physiology/weights.py is unchanged.

Metric gate (live, 2026-11-16, 303 resolved outreach acts over 30 days): mean outcome 0.531, mean
base rate 0.540, mean residual -0.009; base rate range 0.07..0.82, correlation with the outcome
0.31. Every rate looks only at days strictly before the act (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import text

from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE

BASE_RATE_ACTORS = frozenset({"hub.outreach"})
BASE_RATE_SIGNAL = "juniper_replied"
BASE_RATE_DAYS = 28
# Fewer days of history than this and there is no base rate: the ledger keeps the posterior stamp.
BASE_RATE_MIN_DAYS = 7
BASE_RATE_TZ = "America/Denver"

_ENABLED = True


def configure_reply_base_rate(enabled: bool) -> None:
    """SQL_WRITER_ACT_REPLY_BASE_RATE_ENABLED. Off: outreach acts are stamped from the posterior."""
    global _ENABLED
    _ENABLED = bool(enabled)


def applies(actor: Optional[str], signal_id: Optional[str]) -> bool:
    return _ENABLED and actor in BASE_RATE_ACTORS and signal_id == BASE_RATE_SIGNAL


@dataclass(frozen=True)
class ReplyBaseRate:
    rate: float          # Jeffreys-smoothed: (replied_days + 0.5) / (days + 1), never exactly 0 or 1
    days: int            # days with history in the look-back
    replied_days: int    # of those, days she started a turn inside the window

    def note(self) -> str:
        return f"usual for this time of day {self.rate:.2f} ({self.replied_days} of {self.days} days)"


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def base_rate_from_turns(turn_starts: Iterable[datetime], *, opened_at: datetime, horizon_sec: float,
                         first_turn_at: Optional[datetime], days: int = BASE_RATE_DAYS,
                         min_days: int = BASE_RATE_MIN_DAYS, tz: str = BASE_RATE_TZ) -> Optional[ReplyBaseRate]:
    """Pure. For each day k in 1..days: the same local wall-clock time k days before ``opened_at``;
    did a turn start in (that time, that time + horizon]? Days before her first turn ever, or whose
    window would reach past ``opened_at``, are not history. None under ``min_days``."""
    if first_turn_at is None or horizon_sec <= 0:
        return None
    zone = ZoneInfo(tz)
    local = _aware(opened_at).astimezone(zone)
    opened = _aware(opened_at)
    first = _aware(first_turn_at)
    starts = sorted(_aware(t) for t in turn_starts)
    horizon = timedelta(seconds=horizon_sec)
    n = k_replied = 0
    for k in range(1, days + 1):
        # Wall-clock arithmetic on the zone-aware time, then back to an instant: DST keeps the hour.
        lo = (local - timedelta(days=k)).astimezone(timezone.utc)
        hi = lo + horizon
        if lo < first or hi > opened:
            continue
        n += 1
        if any(lo < t <= hi for t in starts):
            k_replied += 1
    if n < min_days:
        return None
    return ReplyBaseRate(rate=(k_replied + 0.5) / (n + 1), days=n, replied_days=k_replied)


# chat_history_log.created_at is naive on the server clock (see orion/regulation/juniper_turns.py).
_TURNS_SQL = text(
    f"""
    SELECT created_at AT TIME ZONE current_setting('TimeZone') AS ts
      FROM chat_history_log
     WHERE {JUNIPER_TURN_PREDICATE}
       AND created_at > CAST(:since AS timestamptz) AT TIME ZONE current_setting('TimeZone')
       AND created_at <= CAST(:until AS timestamptz) AT TIME ZONE current_setting('TimeZone')
    """
)
_FIRST_TURN_SQL = text(
    f"SELECT min(created_at) AT TIME ZONE current_setting('TimeZone') FROM chat_history_log "
    f"WHERE {JUNIPER_TURN_PREDICATE}"
)


def juniper_reply_base_rate(conn: Any, *, opened_at: datetime, horizon_sec: float,
                            days: int = BASE_RATE_DAYS) -> Optional[ReplyBaseRate]:
    """Read-only. Juniper's logged turns (the resolver also counts an ingress hop with no logged row,
    an errored turn; the base rate does not, so it can read a little low)."""
    opened = _aware(opened_at)
    first = conn.execute(_FIRST_TURN_SQL).scalar()
    if first is None:
        return None
    rows = conn.execute(_TURNS_SQL, {"since": opened - timedelta(days=days + 2), "until": opened}).fetchall()
    return base_rate_from_turns((r[0] for r in rows), opened_at=opened, horizon_sec=horizon_sec,
                                first_turn_at=first, days=days)
