"""Operator cursor reset auth, validation, and audit trail."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import Header, HTTPException

from app.settings import get_settings
from app.store import GRAMMAR_CURSOR_REGISTRY

logger = logging.getLogger("orion.substrate.cursor_reset")

ALLOWED_MODES = frozenset({"earliest", "tail", "timestamp"})
KNOWN_CURSORS = frozenset(GRAMMAR_CURSOR_REGISTRY.keys())


@dataclass(frozen=True)
class CursorResetRecord:
    at: datetime
    cursor_name: str
    mode: str
    requested_timestamp: str | None
    prior_created_at: str | None
    prior_event_id: str | None
    new_created_at: str
    new_event_id: str
    actor: str
    history_may_be_skipped: bool


_lock = Lock()
_reset_records: list[CursorResetRecord] = []


def clear_cursor_resets_for_tests() -> None:
    with _lock:
        _reset_records.clear()


def validate_cursor_name(cursor_name: str) -> str:
    name = cursor_name.strip()
    if name not in KNOWN_CURSORS:
        raise ValueError(
            f"invalid cursor_name={cursor_name!r}; known: {', '.join(sorted(KNOWN_CURSORS))}"
        )
    return name


def validate_mode(mode: str) -> str:
    mode_norm = mode.strip().lower()
    if mode_norm not in ALLOWED_MODES:
        raise ValueError(f"invalid mode={mode!r}; allowed: earliest | tail | timestamp")
    return mode_norm


def parse_timestamp_at(raw: str) -> datetime:
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {raw}") from exc
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {raw}")
    return ts


def require_operator_token(x_orion_operator_token: str | None = Header(default=None)) -> str:
    expected = str(get_settings().substrate_cursor_reset_operator_token or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="cursor_reset_operator_token_not_configured")
    provided = str(x_orion_operator_token or "").strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="cursor_reset_unauthorized")
    return provided


def history_may_be_skipped_for_mode(mode: str) -> bool:
    return mode in {"tail", "timestamp"}


def record_cursor_reset(
    *,
    cursor_name: str,
    mode: str,
    requested_timestamp: str | None,
    prior_created_at: str | None,
    prior_event_id: str | None,
    new_created_at: str,
    new_event_id: str,
    actor: str,
    history_may_be_skipped: bool,
) -> CursorResetRecord:
    record = CursorResetRecord(
        at=datetime.now(timezone.utc),
        cursor_name=cursor_name,
        mode=mode,
        requested_timestamp=requested_timestamp,
        prior_created_at=prior_created_at,
        prior_event_id=prior_event_id,
        new_created_at=new_created_at,
        new_event_id=new_event_id,
        actor=actor,
        history_may_be_skipped=history_may_be_skipped,
    )
    with _lock:
        _reset_records.append(record)
        if len(_reset_records) > 200:
            del _reset_records[: len(_reset_records) - 200]

    logger.warning(
        "operator_cursor_reset cursor=%s mode=%s at=%s prior=(%s,%s) new=(%s,%s) "
        "actor=%s history_may_be_skipped=%s",
        cursor_name,
        mode,
        requested_timestamp,
        prior_created_at,
        prior_event_id,
        new_created_at,
        new_event_id,
        actor,
        history_may_be_skipped,
    )
    return record


def cursor_reset_snapshot() -> dict[str, Any]:
    with _lock:
        records = list(_reset_records)
    latest = records[-1] if records else None
    return {
        "count": len(records),
        "last": (
            {
                "at": latest.at.isoformat(),
                "cursor_name": latest.cursor_name,
                "mode": latest.mode,
                "requested_timestamp": latest.requested_timestamp,
                "prior_created_at": latest.prior_created_at,
                "prior_event_id": latest.prior_event_id,
                "new_created_at": latest.new_created_at,
                "new_event_id": latest.new_event_id,
                "actor": latest.actor,
                "history_may_be_skipped": latest.history_may_be_skipped,
            }
            if latest
            else None
        ),
        "recent": [
            {
                "at": r.at.isoformat(),
                "cursor_name": r.cursor_name,
                "mode": r.mode,
                "history_may_be_skipped": r.history_may_be_skipped,
            }
            for r in records[-10:]
        ],
    }


def last_reset_skipped_history() -> bool:
    with _lock:
        if not _reset_records:
            return False
        return _reset_records[-1].history_may_be_skipped
