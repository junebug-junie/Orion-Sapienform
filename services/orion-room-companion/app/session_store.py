"""room_id -> Claude session uuid, persisted across restarts.

One room is one durable Claude session. That is what makes Claude a
participant rather than a stateless helper: the CLI keeps the conversation
on its own side under `--resume`, so a room turn sends only the new message.

Deliberately a small JSON file rather than Postgres or Redis: the mapping is
tiny, single-writer (this service is the only producer of room utterances),
and losing it is recoverable -- a missing entry mints a new session and the
next turn seeds it from the transcript carried in the request.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from uuid import uuid4


def _read_all(path: str | Path) -> Dict[str, str]:
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
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def _quarantine(path: Path) -> None:
    """Move a corrupt state file aside so the next write starts clean and the
    bad content is still there to look at."""
    try:
        path.rename(path.with_suffix(path.suffix + ".corrupt"))
    except Exception:
        pass


def _write_all(path: str | Path, data: Dict[str, str]) -> None:
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


def get_session(path: str | Path, room_key: str) -> str | None:
    return _read_all(path).get(room_key)


def peek_or_mint_session(path: str | Path, room_key: str) -> tuple[str, bool]:
    """Return (session_id, resume) WITHOUT persisting a newly minted id.

    `resume` is False only for a brand-new session, which is exactly when
    `--session-id` rather than `--resume` is the right flag.

    Nothing is written here on purpose. Persisting at mint time recorded a
    session the CLI had not created yet, so a failed first turn left the map
    pointing at nothing and the next turn burned a doomed `--resume` before
    recovering. Callers persist via `remember_session` once a turn succeeds.
    """
    existing = _read_all(path).get(room_key)
    if existing:
        return existing, True
    return str(uuid4()), False


def remember_session(path: str | Path, room_key: str, session_id: str) -> None:
    """Record the session a successful turn actually ran under."""
    data = _read_all(path)
    if data.get(room_key) == session_id:
        return
    data[room_key] = session_id
    _write_all(path, data)


def forget_session(path: str | Path, room_key: str) -> None:
    """Drop a room's session so the next turn starts fresh.

    Needed because a `--resume` against a session the CLI no longer has on
    disk fails every turn forever otherwise -- the room would be permanently
    wedged with no way back short of hand-editing state.
    """
    data = _read_all(path)
    if data.pop(room_key, None) is not None:
        _write_all(path, data)
