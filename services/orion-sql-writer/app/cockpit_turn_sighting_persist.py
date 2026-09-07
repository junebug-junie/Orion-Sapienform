"""Idempotent cockpit_turn_sighting append keyed by (correlation_id, seq).

One bus kind (cockpit.hop.v1) lands one row. Conflict is a republish, not a
merge -- on_conflict_do_nothing and return False. Same insert idiom as
app/harness_turn_trace_persist.py, but do-nothing instead of do-update.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.cockpit_turn_sighting import CockpitTurnSightingSQL

_EXPECTED_SCHEMA = "cockpit.hop.v1"


def _insert_ctor(sess: Session):
    dialect = sess.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert


def _coerce_ts(raw: Any, fallback: datetime) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def append_cockpit_hop(sess: Session, payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    schema = str(payload.get("schema_version") or "").strip()
    if schema and schema != _EXPECTED_SCHEMA:
        return False
    corr_id = str(payload.get("correlation_id") or "").strip()
    seq_raw = payload.get("seq")
    if not corr_id or seq_raw is None:
        return False
    try:
        seq = int(seq_raw)
    except (TypeError, ValueError):
        return False
    if seq < 0:
        return False

    now = datetime.now(timezone.utc)
    values = {
        "correlation_id": corr_id,
        "seq": seq,
        "ts": _coerce_ts(payload.get("ts"), now),
        "stage": str(payload.get("stage") or ""),
        "visor_line": str(payload.get("visor_line") or ""),
        "status": str(payload.get("status") or ""),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "raw": payload.get("raw") if isinstance(payload.get("raw"), dict) else {},
        "producer": str(payload.get("producer") or ""),
        "created_at": now,
    }

    insert = _insert_ctor(sess)
    stmt = insert(CockpitTurnSightingSQL).values(**values)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[
            CockpitTurnSightingSQL.correlation_id,
            CockpitTurnSightingSQL.seq,
        ]
    )
    result = sess.execute(stmt)
    sess.commit()
    return (result.rowcount or 0) > 0
