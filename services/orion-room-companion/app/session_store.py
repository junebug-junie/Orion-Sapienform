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
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


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


def get_or_create_session(path: str | Path, room_key: str) -> tuple[str, bool]:
    """Return (session_id, resume). `resume` is False only on a brand-new
    session, which is exactly when `--session-id` rather than `--resume` is
    the right flag."""
    data = _read_all(path)
    existing = data.get(room_key)
    if existing:
        return existing, True
    minted = str(uuid4())
    data[room_key] = minted
    _write_all(path, data)
    return minted, False


def forget_session(path: str | Path, room_key: str) -> None:
    """Drop a room's session so the next turn starts fresh.

    Needed because a `--resume` against a session the CLI no longer has on
    disk fails every turn forever otherwise -- the room would be permanently
    wedged with no way back short of hand-editing state.
    """
    data = _read_all(path)
    if data.pop(room_key, None) is not None:
        _write_all(path, data)
