"""Tiny read-only Postgres seam for the walkway-camera journal enrichments.

orion-actions had no database access before this; the walkway tables
(`vision_unresolved`, `vision_percept_expectation`,
`vision_individual_sighting`) live in the shared `conjourney` database and are
written by orion-sql-writer and orion-vision-council. This module reads them
and nothing else.

Contract: never raises. `None` means the read did not happen (no DSN, driver
missing, table missing because the manual migration has not been applied,
timeout). `[]` means the database answered and there were no rows. Callers
must keep those apart -- an unapplied migration is not "a quiet street".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

logger = logging.getLogger("orion-actions.vision_pg")

# A journal enrichment is never worth holding a scheduler tick for.
_STATEMENT_TIMEOUT_MS = 5000
_CONNECT_TIMEOUT_SEC = 5


def _fetch_blocking(dsn: str, sql: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    import psycopg2  # inside, so a missing driver degrades instead of aborting import
    import psycopg2.extras

    conn = psycopg2.connect(
        dsn,
        connect_timeout=_CONNECT_TIMEOUT_SEC,
        options=f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}",
    )
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, dict(params))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


async def fetch_rows(
    dsn: str | None, sql: str, params: Mapping[str, Any], *, label: str
) -> list[dict[str, Any]] | None:
    if not (dsn or "").strip():
        logger.info("vision_pg_no_dsn label=%s", label)
        return None
    try:
        return await asyncio.to_thread(_fetch_blocking, dsn.strip(), sql, params)
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("vision_pg_read_failed label=%s err=%s: %s", label, type(exc).__name__, exc)
        return None
