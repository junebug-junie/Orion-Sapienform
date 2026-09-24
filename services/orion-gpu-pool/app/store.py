"""The pool's fenced projection: one row per lease, one row per card.

The LangGraph checkpoint is the lease's history and the source of replay. This projection
exists so the scheduler reads "everything live" in one indexed query, and so there is exactly
one writer: the runtime holds a Postgres advisory lock for its whole life (``leader``), and
only the lock holder subscribes to the lease channels. This is the one direct Postgres writer
in the pool; lease *history* goes bus -> sql-writer (spec, "Transport and telemetry" item 4).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

LEASE_COLUMNS = (
    "lease_id", "request_id", "holder", "work_class", "priority", "kind", "status", "role",
    "attempt", "generation", "operator", "min_ctx_tokens", "needs_vision", "retryable", "created_at",
    "queued_since", "granted_at", "recall_by", "not_before", "deadline_at", "expires_at",
    "turn_correlation_id", "parent_lease_id", "reason", "updated_at",
)
CARD_COLUMNS = ("card", "lent", "swapped_in", "swap_state", "cooldown_until", "last_active_at",
                "updated_at", "updated_by")
LIVE_STATUSES = ("queued", "backlogged", "granted", "recalling", "retry_wait", "dead_letter", "unavailable")
ADVISORY_KEY = 0x6770755F706F6F6C  # "gpu_pool"


class Store(Protocol):
    async def leader_alive(self) -> bool: ...
    async def upsert_lease(self, row: dict[str, Any]) -> None: ...
    async def lease(self, lease_id: str) -> dict[str, Any] | None: ...
    async def lease_by_request(self, request_id: str) -> dict[str, Any] | None: ...
    async def live_leases(self) -> list[dict[str, Any]]: ...
    async def find_leases(self, *, work_class: str | None, holder: str | None, status: str | None,
                          since: datetime | None, until: datetime | None, limit: int) -> list[dict[str, Any]]: ...
    async def cards(self) -> list[dict[str, Any]]: ...
    async def upsert_card(self, row: dict[str, Any]) -> None: ...


class MemoryStore:
    """Same contract, in memory: tests and the eval."""

    def __init__(self) -> None:
        self.leases: dict[str, dict[str, Any]] = {}
        self._cards: dict[str, dict[str, Any]] = {}

    async def upsert_lease(self, row):
        self.leases[row["lease_id"]] = {**self.leases.get(row["lease_id"], {}), **row}

    async def lease(self, lease_id):
        return self.leases.get(lease_id)

    async def lease_by_request(self, request_id):
        return next((r for r in self.leases.values() if r["request_id"] == request_id), None)

    async def live_leases(self):
        return [r for r in self.leases.values() if r["status"] in LIVE_STATUSES]

    async def find_leases(self, *, work_class, holder, status, since, until, limit):
        rows = [r for r in self.leases.values()
                if (work_class is None or r["work_class"] == work_class)
                and (holder is None or r["holder"] == holder)
                and (status is None or r["status"] == status)
                and (since is None or r["created_at"] >= since)
                and (until is None or r["created_at"] < until)]
        return sorted(rows, key=lambda r: r["created_at"])[:limit]

    async def cards(self):
        return list(self._cards.values())

    async def leader_alive(self):
        return True

    async def upsert_card(self, row):
        self._cards[row["card"]] = {**self._cards.get(row["card"], {}), **row}


class PostgresStore:
    def __init__(self, pool: Any, conninfo: str | None = None):
        self.pool = pool
        self.conninfo = conninfo
        self._leader_conn: Any = None

    async def leader(self) -> None:
        """Block until this process is the only pool writer.

        The advisory lock is a SESSION lock, so it lives on a dedicated connection outside the
        pool (a pooled connection could be recycled and silently drop it). ``leader_alive()``
        re-checks it; the service exits if it is ever lost, so two writers can never overlap."""
        import asyncio

        import psycopg
        from psycopg.rows import dict_row

        conninfo = self.conninfo or self.pool.conninfo
        self._leader_conn = await psycopg.AsyncConnection.connect(conninfo, autocommit=True, row_factory=dict_row)
        while True:
            row = await (await self._leader_conn.execute(
                "SELECT pg_try_advisory_lock(%s) AS ok", (ADVISORY_KEY,))).fetchone()
            if row["ok"]:
                return
            await asyncio.sleep(5)

    async def leader_alive(self) -> bool:
        if self._leader_conn is None or self._leader_conn.closed:
            return False
        try:
            row = await (await self._leader_conn.execute(
                "SELECT count(*) AS n FROM pg_locks WHERE locktype='advisory' AND granted "
                "AND pid=pg_backend_pid()")).fetchone()
            return int(row["n"]) > 0
        except Exception:  # noqa: BLE001
            return False

    async def close(self) -> None:
        if self._leader_conn is not None:
            try:
                await self._leader_conn.close()  # ending the session releases the lock
            finally:
                self._leader_conn = None

    async def check_schema(self) -> None:
        """The migration is operator-applied (services/orion-sql-db/manual_migration_gpu_pool_v1.sql).
        Refuse to start without it rather than run on an in-memory illusion."""
        async with self.pool.connection() as conn:
            await conn.execute("SELECT 1 FROM gpu_pool_leases LIMIT 0")
            await conn.execute("SELECT 1 FROM gpu_pool_cards LIMIT 0")

    async def upsert_lease(self, row):
        cols = [c for c in LEASE_COLUMNS if c in row]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "lease_id")
        sql = (f"INSERT INTO gpu_pool_leases ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT (lease_id) DO UPDATE SET {sets}")
        async with self.pool.connection() as conn:
            await conn.execute(sql, [row[c] for c in cols])

    async def _one(self, sql, args):
        async with self.pool.connection() as conn:
            return await (await conn.execute(sql, args)).fetchone()

    async def lease(self, lease_id):
        return await self._one("SELECT * FROM gpu_pool_leases WHERE lease_id=%s", (lease_id,))

    async def lease_by_request(self, request_id):
        return await self._one("SELECT * FROM gpu_pool_leases WHERE request_id=%s", (request_id,))

    async def live_leases(self):
        async with self.pool.connection() as conn:
            return await (await conn.execute(
                "SELECT * FROM gpu_pool_leases WHERE status = ANY(%s)", (list(LIVE_STATUSES),))).fetchall()

    async def find_leases(self, *, work_class, holder, status, since, until, limit):
        where, args = [], []
        for col, val in (("work_class", work_class), ("holder", holder), ("status", status)):
            if val is not None:
                where.append(f"{col}=%s")
                args.append(val)
        if since is not None:
            where.append("created_at>=%s")
            args.append(since)
        if until is not None:
            where.append("created_at<%s")
            args.append(until)
        sql = "SELECT * FROM gpu_pool_leases" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY created_at LIMIT %s"
        async with self.pool.connection() as conn:
            return await (await conn.execute(sql, [*args, limit])).fetchall()

    async def cards(self):
        async with self.pool.connection() as conn:
            return await (await conn.execute("SELECT * FROM gpu_pool_cards")).fetchall()

    async def upsert_card(self, row):
        cols = [c for c in CARD_COLUMNS if c in row]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "card")
        sql = (f"INSERT INTO gpu_pool_cards ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT (card) DO UPDATE SET {sets}")
        async with self.pool.connection() as conn:
            await conn.execute(sql, [row[c] for c in cols])
