"""The self_monitor node's reads and writes, on the service's psycopg pool (autocommit, dict rows).

Reads: every bet source in ``orion.self_monitor.collect.SOURCES`` (the same SQL text the report runs,
in psycopg's parameter style), each in its own read-only transaction with its own statement timeout,
so one slow or missing table costs only its own faculties (named in ``source_errors``).

Writes:
* one ``substrate_self_monitor_history`` row per evaluation, and a prune past the retention
  (``manual_migration_self_monitor_history_v1.sql``);
* ``faculty_status_changed`` rows in ``temporal_self_event`` (``orion.self_monitor.host.event_row``),
  idempotent on their deterministic event id, in one transaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from orion.schemas.self_monitor import SelfMonitorStateV1
from orion.self_monitor.collect import SOURCES, SourceResult, psycopg_sql
from orion.self_monitor.host import FACULTY_EVENT_KIND, FACULTY_EVENT_TABLE

logger = logging.getLogger("orion-durable-runs.self_monitor_store")

# 7 sources x 15 s = 105 s (orion_ask added 2026-11-11; it reads in ~ms), under the node's 120 s ceiling, so a slow source fails alone and is named
# (review finding: at 45 s two slow sources timed out the whole evaluation).
STATEMENT_TIMEOUT = "15s"
WRITE_TIMEOUTS = ("SET LOCAL statement_timeout = '15s'", "SET LOCAL lock_timeout = '5s'")

INSERT_HISTORY_SQL = """
INSERT INTO substrate_self_monitor_history (computed_at, state_json)
VALUES (%(computed_at)s, %(state)s::jsonb)
ON CONFLICT (computed_at) DO NOTHING
"""
PRUNE_HISTORY_SQL = "DELETE FROM substrate_self_monitor_history WHERE computed_at < %(cutoff)s"

INSERT_EVENT_SQL = f"""
INSERT INTO temporal_self_event (event_id, day_id, occurred_at, source_kind, source_table, source_ref,
                                 subject_ref, label, verdict, payload_json)
VALUES (%(event_id)s, %(day_id)s, %(occurred_at)s, '{FACULTY_EVENT_KIND}', '{FACULTY_EVENT_TABLE}',
        %(source_ref)s, %(subject_ref)s, %(label)s, %(label)s, %(payload)s::jsonb)
ON CONFLICT (event_id) DO NOTHING
"""


async def read_sources(pool: Any, now: datetime, window_days: float) -> dict[str, SourceResult]:
    """Rows per source, or the exception that stopped that source's read."""
    since = now - timedelta(days=window_days)
    out: dict[str, SourceResult] = {}
    for name, sql, _ in SOURCES:
        try:
            async with pool.connection() as conn:
                async with conn.transaction():
                    await conn.execute("SET TRANSACTION READ ONLY")
                    await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                    rows = await (await conn.execute(psycopg_sql(sql), {"since": since})).fetchall()
            out[name] = [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001 - named in source_errors by build_state
            logger.warning("self_monitor_source_read_failed source=%s error=%s", name, exc)
            out[name] = exc
    return out


async def record_history(pool: Any, state: SelfMonitorStateV1, *, retention_days: int) -> None:
    """Append the evaluation and prune past the retention. Raises on failure (the node warns)."""
    cutoff = state.computed_at - timedelta(days=retention_days)
    async with pool.connection() as conn:
        async with conn.transaction():
            for sql in WRITE_TIMEOUTS:
                await conn.execute(sql)
            await conn.execute(INSERT_HISTORY_SQL, {"computed_at": state.computed_at,
                                                    "state": state.model_dump_json()})
            await conn.execute(PRUNE_HISTORY_SQL, {"cutoff": cutoff})


async def record_events(pool: Any, rows: list[dict]) -> bool:
    if not rows:
        return True
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                for sql in WRITE_TIMEOUTS:
                    await conn.execute(sql)
                for row in rows:
                    await conn.execute(INSERT_EVENT_SQL, row)
        return True
    except Exception:  # noqa: BLE001 - retried next evaluation (the carry keeps them unrecorded)
        logger.warning("self_monitor_event_write_failed n=%s", len(rows), exc_info=True)
        return False
