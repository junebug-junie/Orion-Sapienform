"""Reverie thought persistence (Phase A store).

Best-effort writer for `SpontaneousThoughtV1` into `substrate_reverie_thought`
(migration `manual_migration_substrate_reverie_thought.sql`). Backs the hub
`_reverie_section` panel. Reuses the sqlalchemy engine pattern already pulled in
by `felt_state_reader` — no new dependency.

Discipline: persistence is best-effort. A DB failure degrades to a logged miss
(returns False) and never breaks the reverie tick. Idempotent on `thought_id`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger("orion-thought.store")

if TYPE_CHECKING:
    from orion.schemas.reverie import SpontaneousThoughtV1

_engine = None


def _database_url() -> str:
    # Prefer the URI the hub observability panel reads, so writes land where the
    # panel looks; fall back to the substrate felt-state DB the reader uses.
    from orion.substrate.felt_state_reader import substrate_felt_state_database_url

    return os.getenv("POSTGRES_URI", "").strip() or substrate_felt_state_database_url()


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def persist_reverie_thought(thought: "SpontaneousThoughtV1") -> bool:
    """Insert one spontaneous thought. Returns True on write, False on any miss.

    Never raises — a persistence failure must not break the tick.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_thought
                        (thought_id, correlation_id, created_at, salience,
                         interpretation, thought_json)
                    VALUES
                        (:thought_id, :correlation_id, :created_at, :salience,
                         :interpretation, CAST(:thought_json AS jsonb))
                    ON CONFLICT (thought_id) DO NOTHING
                    """
                ),
                {
                    "thought_id": thought.thought_id,
                    "correlation_id": thought.correlation_id,
                    "created_at": thought.created_at,
                    "salience": float(thought.salience),
                    "interpretation": thought.interpretation,
                    "thought_json": json.dumps(thought.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("reverie thought persist failed id=%s err=%s", thought.thought_id, exc)
        return False
