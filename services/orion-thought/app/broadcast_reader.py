"""Direct read of the current attention-broadcast projection from Postgres.

orion-thought is a bus service, but reverie is *self-driven* and has no incoming
request payload (unlike the evoked path, which gets the coalition from
`request.association.broadcast`). So reverie reads the latest broadcast the same
place hub/self-state read it — `substrate_attention_broadcast_projection` — via a
minimal direct query.

Deliberately does NOT import `orion.substrate.felt_state_reader`: that runs the
heavy `orion.substrate` package `__init__`, which pulls the full graph engine
(`requests` etc.) that this thin service does not ship. Here we need only
sqlalchemy + a psycopg driver.

Fail-open: any failure (missing table, unavailable DB, stale row) returns None so
a reverie tick degrades to "no coalition" — never raises.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1

logger = logging.getLogger("orion-thought.broadcast_reader")

# Reverie may narrate a slightly-aged coalition, but not a dead one. Broadcasts
# tick ~30s; reverie ~90s. Skip anything older than this (substrate not ticking).
DEFAULT_MAX_AGE_SEC = 300.0

_engine = None


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def read_latest_broadcast(
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
) -> AttentionBroadcastProjectionV1 | None:
    """Latest fresh attention-broadcast projection, or None. Never raises."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT projection_json, generated_at "
                        "FROM substrate_attention_broadcast_projection "
                        "ORDER BY generated_at DESC LIMIT 1"
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            logger.info("no attention broadcast row yet; reverie skips")
            return None
        gen = row.get("generated_at")
        if isinstance(gen, datetime):
            ts = gen if gen.tzinfo else gen.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > max_age_sec:
                logger.info("attention broadcast stale (%.0fs > %.0fs); reverie skips", age, max_age_sec)
                return None
        payload = row["projection_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return AttentionBroadcastProjectionV1.model_validate(payload)
    except Exception as exc:
        logger.warning("attention broadcast read failed: %s", exc)
        return None
