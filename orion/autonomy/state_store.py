"""Postgres-backed persistence for the latest AutonomyStateV2 per subject.

Closes the reducer's own fold loop: `chat_stance.py:_run_autonomy_reducer()`
currently computes a fresh AutonomyStateV2 every turn and discards it, so
`previous_state` always falls back to the V1/graph baseline. This module gives
that reducer output somewhere to live between turns.

Mirrors `orion/autonomy/action_outcomes.py`'s SQL connection convention
exactly: SQLAlchemy `create_engine`, a process-level engine cache keyed by
DB URL, and a dedicated env var naming the DSN. This is a separate table from
the graph/Fuseki-backed homeostatic drives system -- it must not be read from
or written to here.

Both functions fail open. This store is read/written on the hot chat-turn
path; a DB hiccup must never block or fail a turn.
"""
from __future__ import annotations

import json
import logging
import os

from orion.autonomy.models import AutonomyStateV2

logger = logging.getLogger("orion.autonomy.state_store")

# Process-level engine cache keyed by database URL (SQLAlchemy engines are pooled).
_ENGINE_CACHE: dict[str, object] = {}


def _db_url() -> str | None:
    url = os.getenv("ORION_AUTONOMY_STATE_DB_URL", "").strip()
    return url or None


def _get_engine(url: str):
    from sqlalchemy import create_engine

    engine = _ENGINE_CACHE.get(url)
    if engine is None:
        # setdefault keeps the cache race-free if two callers build concurrently:
        # the first inserted engine wins and any extra is discarded.
        engine = _ENGINE_CACHE.setdefault(url, create_engine(url, pool_pre_ping=True))
    return engine


def load_autonomy_state_v2(subject: str) -> AutonomyStateV2 | None:
    """Load the most recently persisted AutonomyStateV2 for a subject.

    Returns None when no DSN is configured, no row exists yet, or anything
    goes wrong reading/parsing the row. Never raises -- callers should treat
    None as "no persisted V2 state yet" and fall back to their own baseline.
    """
    url = _db_url()
    if not url:
        return None
    try:
        from sqlalchemy import text

        engine = _get_engine(url)
        query = text("SELECT state FROM autonomy_state_v2 WHERE subject = :subject")
        with engine.connect() as conn:
            row = conn.execute(query, {"subject": subject}).mappings().first()
        if row is None:
            return None
        raw = row["state"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return None
        return AutonomyStateV2.model_validate(raw)
    except Exception as exc:
        logger.warning(
            "autonomy_state_v2_load_failed subject=%s error=%s", subject, exc
        )
        return None


def save_autonomy_state_v2(subject: str, state: AutonomyStateV2) -> None:
    """Upsert the latest AutonomyStateV2 for a subject.

    No-ops when no DSN is configured. Fails open on any write error: logs and
    returns rather than raising, so a DB hiccup never blocks a chat turn.
    """
    url = _db_url()
    if not url:
        return
    try:
        from sqlalchemy import text

        engine = _get_engine(url)
        payload = json.dumps(state.model_dump(mode="json"))
        query = text(
            """
            INSERT INTO autonomy_state_v2 (subject, state, updated_at)
            VALUES (:subject, :state, CURRENT_TIMESTAMP)
            ON CONFLICT (subject) DO UPDATE
            SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at
            """
        )
        with engine.begin() as conn:
            conn.execute(query, {"subject": subject, "state": payload})
    except Exception as exc:
        logger.warning(
            "autonomy_state_v2_save_failed subject=%s error=%s", subject, exc
        )
        return
