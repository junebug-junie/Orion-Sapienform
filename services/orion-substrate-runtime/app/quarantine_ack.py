"""Operator quarantine acknowledgement auth, validation, and audit trail."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import Header, HTTPException

from app.settings import get_settings
from app.store import GRAMMAR_CURSOR_REGISTRY

logger = logging.getLogger("orion.substrate.quarantine_ack")

KNOWN_CURSORS = frozenset(GRAMMAR_CURSOR_REGISTRY.keys())
CURSOR_BY_REDUCER: dict[str, str] = {
    "biometrics": "biometrics_grammar_consumer",
    "execution_trajectory": "execution_grammar_reducer",
    "transport_bus": "transport_grammar_reducer",
}


@dataclass(frozen=True)
class QuarantineAckRecord:
    at: datetime
    cursor_name: str
    reducer_key: str
    event_id: str | None
    ack_all: bool
    actor: str
    acknowledged_count: int


_lock = Lock()
_ack_records: list[QuarantineAckRecord] = []


def clear_quarantine_acks_for_tests() -> None:
    with _lock:
        _ack_records.clear()


def validate_cursor_name(cursor_name: str) -> str:
    name = cursor_name.strip()
    if name not in KNOWN_CURSORS:
        raise ValueError(
            f"invalid cursor_name={cursor_name!r}; known: {', '.join(sorted(KNOWN_CURSORS))}"
        )
    return name


def reducer_key_for_cursor(cursor_name: str) -> str:
    for reducer_key, cursor in CURSOR_BY_REDUCER.items():
        if cursor == cursor_name:
            return reducer_key
    return cursor_name


def require_operator_token(x_orion_operator_token: str | None = Header(default=None)) -> str:
    expected = str(get_settings().substrate_cursor_reset_operator_token or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="cursor_reset_operator_token_not_configured")
    provided = str(x_orion_operator_token or "").strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="quarantine_ack_unauthorized")
    return provided


def record_quarantine_ack(
    *,
    cursor_name: str,
    reducer_key: str,
    event_id: str | None,
    ack_all: bool,
    actor: str,
    acknowledged_count: int,
) -> QuarantineAckRecord:
    record = QuarantineAckRecord(
        at=datetime.now(timezone.utc),
        cursor_name=cursor_name,
        reducer_key=reducer_key,
        event_id=event_id,
        ack_all=ack_all,
        actor=actor,
        acknowledged_count=acknowledged_count,
    )
    with _lock:
        _ack_records.append(record)
        if len(_ack_records) > 200:
            del _ack_records[: len(_ack_records) - 200]

    logger.warning(
        "operator_quarantine_ack cursor=%s reducer=%s event_id=%s ack_all=%s "
        "actor=%s acknowledged_count=%d",
        cursor_name,
        reducer_key,
        event_id,
        ack_all,
        actor,
        acknowledged_count,
    )
    return record


def quarantine_ack_snapshot() -> dict[str, Any]:
    with _lock:
        records = list(_ack_records)
    latest = records[-1] if records else None
    return {
        "count": len(records),
        "last": (
            {
                "at": latest.at.isoformat(),
                "cursor_name": latest.cursor_name,
                "reducer_key": latest.reducer_key,
                "event_id": latest.event_id,
                "ack_all": latest.ack_all,
                "actor": latest.actor,
                "acknowledged_count": latest.acknowledged_count,
            }
            if latest
            else None
        ),
        "recent": [
            {
                "at": r.at.isoformat(),
                "cursor_name": r.cursor_name,
                "event_id": r.event_id,
                "ack_all": r.ack_all,
                "acknowledged_count": r.acknowledged_count,
            }
            for r in records[-10:]
        ],
    }
