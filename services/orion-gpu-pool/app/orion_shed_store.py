"""Postgres ledger for Orion's learned shed (``public.gpu_pool_orion_shed``).

One row per ``orion_self_shed`` request (active, settled or refused). Read by the pool at boot (so a
restart cannot reset the daily cap or orphan an active shed), by execution-dispatch to settle the
dispatch, and by the feedback runtime for the manipulation check.

The table is created lazily (``CREATE TABLE IF NOT EXISTS`` under a short, transaction-local
``lock_timeout``) when the pool starts and again on a set after a ledger failure -- never inside a
lease RPC, and a failure never blocks the lease path; the same DDL is in
services/orion-sql-db/manual_migration_gpu_pool_orion_shed_v1.sql for the operator. A ledger that cannot
be created or written refuses every ``set`` (orion/gpu_pool/orion_shed.py), and the lease path is never
affected. Schema-qualified on purpose: the pool's search_path puts the checkpoint schema first.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("orion-gpu-pool.orion_shed_store")

TABLE = "public.gpu_pool_orion_shed"
COLUMNS = ("shed_id", "dispatch_id", "reason", "state", "refusal", "ttl_sec", "requested_at", "started_at",
           "valid_until", "ended_at", "drained_at", "grants_withheld", "delayed_grant_sec",
           "background_live_at_start", "correlation", "detail")
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    shed_id text PRIMARY KEY,
    dispatch_id text NOT NULL,
    reason text NOT NULL DEFAULT 'orion_self_shed',
    state text NOT NULL,
    refusal text,
    ttl_sec double precision,
    requested_at timestamptz NOT NULL,
    started_at timestamptz,
    valid_until timestamptz,
    ended_at timestamptz,
    drained_at timestamptz,
    grants_withheld integer NOT NULL DEFAULT 0,
    delayed_grant_sec double precision NOT NULL DEFAULT 0,
    background_live_at_start integer NOT NULL DEFAULT 0,
    correlation jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS gpu_pool_orion_shed_dispatch_idx ON {TABLE} (dispatch_id);
CREATE INDEX IF NOT EXISTS gpu_pool_orion_shed_requested_idx ON {TABLE} (requested_at DESC);
"""


class PostgresOrionShedLedger:
    def __init__(self, pool: Any, *, lock_timeout_ms: int = 3000):
        self.pool = pool
        self.lock_timeout_ms = lock_timeout_ms
        self._ready = False

    async def ensure(self) -> bool:
        try:
            async with self.pool.connection() as conn:
                # SET LOCAL inside one transaction: the timeout can never leak to a pooled connection
                # the lease path reuses, even when a CREATE fails.
                async with conn.transaction():
                    await conn.execute(f"SET LOCAL lock_timeout = '{int(self.lock_timeout_ms)}ms'")
                    for stmt in (s.strip() for s in DDL.split(";")):
                        if stmt:
                            await conn.execute(stmt)
            self._ready = True
        except Exception as exc:  # noqa: BLE001 -- the caller refuses every set until it can
            logger.warning("gpu_pool_orion_shed_ledger_ensure_failed err=%s", str(exc)[:300])
            self._ready = False
        return self._ready

    async def _rows(self, sql: str, args: tuple) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            return [dict(r) for r in await (await conn.execute(sql, args)).fetchall()]

    async def recent(self, since):
        return await self._rows(f"SELECT * FROM {TABLE} WHERE requested_at >= %s ORDER BY requested_at", (since,))

    async def by_dispatch(self, dispatch_id):
        rows = await self._rows(f"SELECT * FROM {TABLE} WHERE dispatch_id = %s", (dispatch_id,))
        return rows[0] if rows else None

    async def by_id(self, shed_id):
        rows = await self._rows(f"SELECT * FROM {TABLE} WHERE shed_id = %s", (shed_id,))
        return rows[0] if rows else None

    async def upsert(self, row):
        vals = [json.dumps(row[c], default=str) if c in ("correlation", "detail") else row.get(c) for c in COLUMNS]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in COLUMNS if c not in ("shed_id", "dispatch_id", "requested_at"))
        sql = (f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}, updated_at) "
               f"VALUES ({', '.join(['%s'] * len(COLUMNS))}, now()) "
               f"ON CONFLICT (shed_id) DO UPDATE SET {sets}, updated_at = now()")
        async with self.pool.connection() as conn:
            await conn.execute(sql, vals)
