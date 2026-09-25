"""Dream cycle v2 store. Reads four producer tables; writes only v2 tables.

Read-only on every source (orion_metacog, dream_compaction_request_queue,
substrate_reverie_resonance_alert, memory_crystallizations, chat_history_log).
Writes exactly CYCLE_WRITE_TABLES -- a test pins that no canonical memory
table ever appears in a write statement here.

Every loader degrades to empty/None on failure and logs why: a missing table
(migration not applied yet) must read as "nothing to replay", never raise.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from orion.schemas.dream_cycle import DreamCycleV1

from app.settings import settings

logger = logging.getLogger("orion-dream.cycle_store")

CYCLE_WRITE_TABLES = ("dream_cycle", "dream_replay_item", "dream_hypothesis")

# `timestamp` on orion_metacog is a producer-written ISO string, not a
# timestamptz. Cast only well-formed values so one bad row cannot fail the read
# (CASE, not AND: Postgres does not promise AND short-circuits).
SOURCE_QUERIES: dict[str, str] = {
    "metacog": """
        SELECT id, summary, severity, trigger_kind, tags FROM orion_metacog
         WHERE severity IN ('degraded', 'critical')
           AND CASE WHEN timestamp ~ '^\\d{4}-\\d{2}-\\d{2}T'
                    THEN CAST(timestamp AS timestamptz) END > :since
         ORDER BY timestamp DESC LIMIT :limit
    """,
    "compaction_request": """
        SELECT request_id, theme, reason FROM dream_compaction_request_queue
         WHERE created_at > :since
         ORDER BY created_at DESC LIMIT :limit
    """,
    "resonance": """
        SELECT alert_id, theme_key, violation_count FROM substrate_reverie_resonance_alert
         WHERE created_at > :since
         ORDER BY created_at DESC LIMIT :limit
    """,
    "crystallization": """
        SELECT crystallization_id, subject, summary, salience, tags FROM memory_crystallizations
         WHERE status = 'active' AND updated_at > :since
         ORDER BY salience DESC, updated_at DESC LIMIT :limit
    """,
}

# chat_history_log.created_at is `timestamp without time zone` defaulted by the
# server's now(), so compare against LOCALTIMESTAMP on the same server clock.
IDLE_MINUTES_SQL = """
    SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - max(created_at))) / 60.0 AS idle
      FROM chat_history_log
"""

LAST_CYCLE_END_SQL = "SELECT max(ended_at) AS ended_at FROM dream_cycle"

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(settings.POSTGRES_URI, pool_pre_ping=True)
    return _engine


def load_source_rows(since: datetime, limit_per_source: int) -> dict[str, list[dict[str, Any]]]:
    """source_kind -> rows. A failing source is empty and logged, not fatal."""
    from sqlalchemy import text

    out: dict[str, list[dict[str, Any]]] = {}
    engine = _get_engine()
    for kind, sql in SOURCE_QUERIES.items():
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(sql), {"since": since, "limit": int(limit_per_source)}).mappings().all()
            out[kind] = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("dream_cycle source read failed kind=%s err=%s", kind, exc)
            out[kind] = []
    return out


def load_idle_minutes() -> Optional[float]:
    """Minutes since the last chat turn. None = unknown (treated as NOT idle)."""
    try:
        from sqlalchemy import text

        with _get_engine().connect() as conn:
            row = conn.execute(text(IDLE_MINUTES_SQL)).mappings().first()
        if row is None or row["idle"] is None:
            return None
        return float(row["idle"])
    except Exception as exc:
        logger.warning("dream_cycle idle read failed err=%s", exc)
        return None


def load_last_cycle_end() -> Optional[datetime]:
    try:
        from sqlalchemy import text

        with _get_engine().connect() as conn:
            row = conn.execute(text(LAST_CYCLE_END_SQL)).mappings().first()
        return row["ended_at"] if row else None
    except Exception as exc:
        logger.warning("dream_cycle last-cycle read failed err=%s", exc)
        return None


def persist_cycle(cycle: DreamCycleV1) -> bool:
    """One transaction: cycle row, replay items, hypotheses. Never raises."""
    try:
        from sqlalchemy import text

        with _get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dream_cycle
                        (cycle_id, trigger, status, started_at, ended_at, pressure,
                         replay_count, hypothesis_count, no_link_count, llm_failures,
                         compaction_delta_id, cycle_json)
                    VALUES
                        (:cycle_id, :trigger, :status, :started_at, :ended_at, :pressure,
                         :replay_count, :hypothesis_count, :no_link_count, :llm_failures,
                         :compaction_delta_id, CAST(:cycle_json AS jsonb))
                    ON CONFLICT (cycle_id) DO NOTHING
                    """
                ),
                {
                    "cycle_id": cycle.cycle_id,
                    "trigger": cycle.trigger,
                    "status": cycle.status,
                    "started_at": cycle.started_at,
                    "ended_at": cycle.ended_at,
                    "pressure": cycle.pressure.pressure,
                    "replay_count": len(cycle.replay),
                    "hypothesis_count": len(cycle.hypotheses),
                    "no_link_count": cycle.no_link_count,
                    "llm_failures": cycle.llm_failures,
                    "compaction_delta_id": cycle.compaction_delta_id,
                    "cycle_json": json.dumps(cycle.model_dump(mode="json")),
                },
            )
            for rank, item in enumerate(cycle.replay):
                conn.execute(
                    text(
                        """
                        INSERT INTO dream_replay_item
                            (cycle_id, ref_id, rank, source_kind, weight, reason)
                        VALUES (:cycle_id, :ref_id, :rank, :source_kind, :weight, :reason)
                        ON CONFLICT (cycle_id, ref_id) DO NOTHING
                        """
                    ),
                    {
                        "cycle_id": cycle.cycle_id,
                        "ref_id": item.ref_id,
                        "rank": rank,
                        "source_kind": item.source_kind,
                        "weight": item.weight,
                        "reason": item.reason,
                    },
                )
            for h in cycle.hypotheses:
                conn.execute(
                    text(
                        """
                        INSERT INTO dream_hypothesis
                            (hypothesis_id, cycle_id, arm, claim, why, ref_a, ref_b,
                             created_at, expires_at)
                        VALUES (:hypothesis_id, :cycle_id, :arm, :claim, :why, :ref_a, :ref_b,
                                :created_at, :expires_at)
                        ON CONFLICT (hypothesis_id) DO NOTHING
                        """
                    ),
                    h.model_dump(),
                )
        return True
    except Exception as exc:
        logger.warning("dream_cycle persist failed id=%s err=%s", cycle.cycle_id, exc)
        return False
