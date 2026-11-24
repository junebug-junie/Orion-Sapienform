"""Back off after Claude itself says the usage limit is hit (#3655).

Why this exists: the transcript meter (`orion.dev_economics.rate_limit_events
.observe`) only sees Claude Code sessions. It was wrong both ways:

* 2026-11-19..23: no Claude Code session ran, so it read `unknown` and 13
  hires were refused `claude_budget_unobserved` while the account was idle.
* 2026-11-17 20:28Z and 2026-11-24 09:12Z / 11:23Z: it read `clear`, and
  Claude (via orion-room-companion) answered "You've hit your session limit ·
  resets 11pm (UTC)". The second 11-24 hire repeated a failure the first one
  had already announced.

Juniper's rule (2026-11-24): try, back off on refusal. The helper's own last
Claude outcome is the second input. A failure that names a session/weekly/usage
limit holds every hire until the stated reset; an unparseable reset holds for a
bounded default. Any later Claude outcome (ok, or a non-limit failure) clears it.

This is helper-local. `observe()` and `ask_claude_trigger._budget_refusal`
keep their meaning for every other caller.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.dev_economics.rate_limit_events import parse_reset_at

logger = logging.getLogger("orion-curiosity-peer.claude_limit_backoff")

# Only failures inside this window seed the back-off at boot -- the same 5 h
# window as Claude's session limit and as observe()'s default.
SEED_LOOKBACK = timedelta(hours=5)

# Claude's first-party limit phrasings. Lower-cased substring match on a FAILED
# Claude brief only (never on ok summaries, which may discuss limits).
_LIMIT_PHRASES = (
    "hit your session limit",
    "hit your weekly limit",
    "hit your usage limit",
    "usage limit reached",
)

# "resets 12:10pm (UTC)" / "resets 11pm (America/Denver)"
_RESET_TZ_RE = re.compile(
    r"resets\s+[^()\n]*?(?:am|pm)\s*\((?P<tz>[A-Za-z_]+(?:/[A-Za-z_+\-]+)*)\)",
    re.IGNORECASE,
)


def is_limit_failure_text(text: str) -> bool:
    low = (text or "").lower()
    return any(phrase in low for phrase in _LIMIT_PHRASES)


def parse_limit_reset(text: str, at: datetime) -> Optional[datetime]:
    """Reset time Claude stated, as UTC. None when the text names none.

    Reuses `rate_limit_events.parse_reset_at` (which reads wall-clock times as
    UTC). When Claude names another zone, the event time is shifted into that
    zone's wall clock first, parsed, then converted back.
    """
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    zone: Any = timezone.utc
    match = _RESET_TZ_RE.search(text or "")
    if match and match.group("tz").upper() not in ("UTC", "GMT", "Z"):
        try:
            zone = ZoneInfo(match.group("tz"))
        except (ZoneInfoNotFoundError, ValueError):
            zone = timezone.utc
    wall = at.astimezone(zone).replace(tzinfo=timezone.utc)
    parsed = parse_reset_at(text or "", wall)
    if parsed is None:
        return None
    return parsed.replace(tzinfo=None).replace(tzinfo=zone).astimezone(timezone.utc)


@dataclass(frozen=True)
class ClaudeLimitHold:
    failed_at: datetime
    resets_at: datetime
    reset_stated: bool  # False = bounded default, Claude named no reset time
    text: str


def hold_from_failure(
    text: str, at: datetime, *, default_sec: float
) -> Optional[ClaudeLimitHold]:
    """A hold when `text` is a Claude limit refusal, else None."""
    if not is_limit_failure_text(text):
        return None
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    resets_at = parse_limit_reset(text, at)
    stated = resets_at is not None
    if resets_at is None:
        resets_at = at + timedelta(seconds=max(0.0, float(default_sec)))
    return ClaudeLimitHold(failed_at=at, resets_at=resets_at, reset_stated=stated, text=text)


class ClaudeLimitMemory:
    """The helper's most recent Claude outcome, in process memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hold: Optional[ClaudeLimitHold] = None

    def reset(self) -> None:
        with self._lock:
            self._hold = None

    def record_outcome(
        self, *, status: str, text: str, at: datetime, default_sec: float
    ) -> Optional[ClaudeLimitHold]:
        """Record the result of a real Claude call. Ok or any non-limit
        failure clears the hold; a limit failure sets it."""
        hold = hold_from_failure(text, at, default_sec=default_sec) if status == "failed" else None
        with self._lock:
            self._hold = hold
        return hold

    def active(self, now: datetime) -> Optional[ClaudeLimitHold]:
        with self._lock:
            hold = self._hold
        if hold is None or now >= hold.resets_at:
            return None
        return hold

    def seed_from_rows(
        self, rows: Iterable[dict[str, Any]], *, now: datetime, default_sec: float
    ) -> Optional[ClaudeLimitHold]:
        """Seed from the newest claude_room ok/failed brief in the last 5 h.

        Rows come from `RECENT_CLAUDE_OUTCOME_CYPHER` (written_at = ms epoch).
        Only seeds when nothing was recorded in this process yet.
        """
        newest: Optional[tuple[datetime, dict[str, Any]]] = None
        for row in rows or []:
            raw = row.get("written_at")
            try:
                at = datetime.fromtimestamp(float(raw) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            if now - at > SEED_LOOKBACK or at > now + timedelta(minutes=5):
                continue
            if newest is None or at > newest[0]:
                newest = (at, row)
        if newest is None:
            return None
        at, row = newest
        text = " ".join(
            str(row.get(k) or "") for k in ("refusal_reason", "summary")
        ).strip()
        hold = (
            hold_from_failure(text, at, default_sec=default_sec)
            if str(row.get("status") or "") == "failed"
            else None
        )
        with self._lock:
            if self._hold is None:
                self._hold = hold
        return hold


# Read path for boot seeding: the same worldview PeerBrief nodes this service
# MERGEs (`peer_brief_merge_cypher`, written_at = timestamp() ms on create).
RECENT_CLAUDE_OUTCOME_CYPHER = (
    "MATCH (b:PeerBrief) "
    "WHERE b.peer = 'claude_room' AND b.status IN ['ok', 'failed'] "
    "AND b.written_at >= $since_ms "
    "RETURN b.written_at AS written_at, b.status AS status, "
    "b.refusal_reason AS refusal_reason, b.summary AS summary "
    "ORDER BY b.written_at DESC LIMIT 1"
)


def seed_memory_from_graph(
    memory: ClaudeLimitMemory,
    graph_query: Callable[[str, dict[str, Any]], Any],
    *,
    now: datetime,
    default_sec: float,
) -> Optional[ClaudeLimitHold]:
    since_ms = int((now - SEED_LOOKBACK).timestamp() * 1000)
    rows = graph_query(RECENT_CLAUDE_OUTCOME_CYPHER, {"since_ms": since_ms})
    return memory.seed_from_rows(list(rows or []), now=now, default_sec=default_sec)


def describe_hold(hold: ClaudeLimitHold) -> str:
    """Plain sentence Orion reads (before the `[code]` block)."""
    reset = hold.resets_at.strftime("%Y-%m-%d %H:%MZ")
    failed = hold.failed_at.strftime("%H:%MZ")
    if hold.reset_stated:
        return (
            f"Claude helper refused: Claude said at {failed} that its usage "
            f"limit was hit and resets at {reset}, so this hire waits until then."
        )
    return (
        f"Claude helper refused: Claude said at {failed} that its usage limit "
        f"was hit without a reset time, so this hire waits until {reset}."
    )


MEMORY = ClaudeLimitMemory()
