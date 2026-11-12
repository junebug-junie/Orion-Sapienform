"""room_id -> Claude session, persisted across restarts.

One room is one durable Claude session per Denver day. That is what makes
Claude a participant rather than a stateless helper: the CLI keeps the
conversation on its own side under `--resume`, so a room turn sends only what
is new.

Each entry holds three things (2026-11-12, design
docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md):

- ``session_id``: the Claude session the room resumes.
- ``day``: the Denver "room day" the session was minted on. A day starts at
  04:00 America/Denver, so the first invite after 04:00 mints a new session
  that is reseeded from the whole conversation window. This bounds what a
  resumed session carries, and so its per-turn cost.
- ``sent_up_to``: the newest transcript ``created_at`` this session has been
  sent. On ``--resume`` the companion sends only lines newer than this, so a
  resumed turn never gets duplicates of lines Claude already holds.

Legacy entries (a bare session-id string, before 2026-11-12) carry no day, so
the next invite rotates them: the first turn after this deploy reseeds Claude
with the whole conversation instead of resuming a 12-turn session.

Deliberately a small JSON file rather than Postgres or Redis: the mapping is
tiny, single-writer (this service is the only producer of room utterances),
and losing it is recoverable -- a missing entry mints a new session and the
next turn seeds it from the transcript carried in the request.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

# Juniper's clock. A "room day" starts at ROOM_DAY_START_HOUR local time.
ROOM_TZ_NAME = "America/Denver"
ROOM_DAY_START_HOUR = 4


def room_day(now: Optional[datetime] = None) -> str:
    """The Denver room day for ``now``, as YYYY-MM-DD. 03:59 local still
    belongs to the previous day; 04:00 starts a new one."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        local = now.astimezone(ZoneInfo(ROOM_TZ_NAME))
    except Exception:  # noqa: BLE001 -- missing tzdata: a fixed MST offset is close enough to rotate daily
        local = now.astimezone(timezone(timedelta(hours=-7)))
    return (local - timedelta(hours=ROOM_DAY_START_HOUR)).date().isoformat()


@dataclass(frozen=True)
class SessionSlot:
    session_id: str
    resume: bool
    # Newest transcript created_at already sent to this session (ISO-8601), or
    # None for a fresh session.
    sent_up_to: Optional[str]
    day: str
    # Why a new session was minted: "new", "rotated" (a new room day or a
    # legacy entry). None when resuming.
    minted_reason: Optional[str] = None


def _read_all(path: str | Path) -> Dict[str, Any]:
    """Best-effort read. Any unreadable/corrupt state degrades to "no sessions
    known", which costs continuity but never a turn.

    Catches broadly on purpose: a narrow (FileNotFoundError, JSONDecodeError)
    still let UnicodeDecodeError, IsADirectoryError and PermissionError escape
    into run_turn, where they killed the turn before `claude` ran and -- since
    the consumer's blanket handler drops the request -- published nothing at
    all. Silence is the one failure mode this service must not have.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        # Corrupt or unreadable. Do NOT let _write_all then overwrite the file
        # wholesale -- that would discard every other room's mapping too.
        _quarantine(p)
        return {}
    if not isinstance(raw, dict):
        _quarantine(p)
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, (str, dict))}


def _quarantine(path: Path) -> None:
    """Move a corrupt state file aside so the next write starts clean and the
    bad content is still there to look at."""
    try:
        path.rename(path.with_suffix(path.suffix + ".corrupt"))
    except Exception:
        pass


def _write_all(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a crash mid-write cannot leave a truncated file
    # that would silently reset every room's session on next boot.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".sessions-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _entry(value: Any) -> Dict[str, Any]:
    """Normalize one stored value. A legacy bare string has no day."""
    if isinstance(value, str):
        return {"session_id": value}
    if isinstance(value, dict) and isinstance(value.get("session_id"), str):
        return dict(value)
    return {}


def get_session(path: str | Path, room_key: str) -> str | None:
    return _entry(_read_all(path).get(room_key)).get("session_id")


def peek_or_mint_session(
    path: str | Path,
    room_key: str,
    *,
    now: Optional[datetime] = None,
    rotate_daily: bool = True,
) -> SessionSlot:
    """The session this turn should run under, WITHOUT persisting a new one.

    ``resume`` is False only for a brand-new session, which is exactly when
    `--session-id` rather than `--resume` is the right flag. A stored session
    from an earlier room day (or a legacy entry with no day) is rotated: a new
    id is minted and the turn is a first turn again.

    Nothing is written here on purpose. Persisting at mint time recorded a
    session the CLI had not created yet, so a failed first turn left the map
    pointing at nothing and the next turn burned a doomed `--resume` before
    recovering. Callers persist via `remember_session` once a turn succeeds.

    ``rotate_daily=False`` (consult rooms): a consult can run across 04:00 and
    must keep its one session for its whole life, so it never rotates.
    """
    today = room_day(now)
    entry = _entry(_read_all(path).get(room_key))
    existing = entry.get("session_id")
    if existing and (entry.get("day") == today or not rotate_daily):
        mark = entry.get("sent_up_to")
        return SessionSlot(
            session_id=existing, resume=True,
            sent_up_to=mark if isinstance(mark, str) and mark else None, day=today,
        )
    return SessionSlot(
        session_id=str(uuid4()), resume=False, sent_up_to=None, day=today,
        minted_reason="rotated" if existing else "new",
    )


def remember_session(
    path: str | Path,
    room_key: str,
    session_id: str,
    *,
    day: Optional[str] = None,
    sent_up_to: Optional[str] = None,
) -> None:
    """Record the session a successful turn actually ran under, its room day,
    and the newest transcript line it has now been sent.

    The mark only moves forward: a turn that carried no newer lines keeps the
    old mark rather than dropping it."""
    data = _read_all(path)
    old = _entry(data.get(room_key))
    mark = sent_up_to
    if old.get("session_id") == session_id and isinstance(old.get("sent_up_to"), str):
        mark = latest_iso(old["sent_up_to"], sent_up_to)
    new = {"session_id": session_id, "day": day or room_day()}
    if mark:
        new["sent_up_to"] = mark
    if data.get(room_key) == new:
        return
    data[room_key] = new
    _write_all(path, data)


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    """ISO-8601 -> aware datetime (naive means UTC); None when unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def latest_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """The later of two ISO times; an unparseable one loses."""
    pa, pb = parse_iso(a), parse_iso(b)
    if pa is None:
        return b if pb is not None else None
    if pb is None:
        return a
    return a if pa >= pb else b


def forget_session(path: str | Path, room_key: str) -> None:
    """Drop a room's session so the next turn starts fresh.

    Needed because a `--resume` against a session the CLI no longer has on
    disk fails every turn forever otherwise -- the room would be permanently
    wedged with no way back short of hand-editing state.
    """
    data = _read_all(path)
    if data.pop(room_key, None) is not None:
        _write_all(path, data)
