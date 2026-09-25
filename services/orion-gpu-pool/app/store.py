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
    "turn_correlation_id", "parent_lease_id", "hold_lease_id", "reason", "updated_at",
)
CARD_COLUMNS = ("card", "lent", "swapped_in", "swap_state", "cooldown_until", "last_active_at",
                "swap_role", "swap_generation", "swap_action", "residency_until", "loaded_at",
                "updated_at", "updated_by")
# Columns added by services/orion-sql-db/manual_migration_gpu_pool_v2_holds.sql (stage 4.3).
V2_LEASE_COLUMNS = ("hold_lease_id",)
V2_CARD_COLUMNS = ("swap_role", "swap_generation", "swap_action", "residency_until", "loaded_at")
# LangGraph's checkpoint tables for lease threads live in their own schema. They used to share
# public.checkpoints with durable-runs, whose resume sweep lists EVERY checkpoint in that table
# (alist(None)) every 2 minutes: at one lease per inference the pool's threads became most of
# the table within minutes, the sweep's full scan ran 8+ s, and lease RPCs stalled behind it
# until callers timed out and left granted leases nobody held (2026-09-25).
CHECKPOINT_SCHEMA = "gpu_pool"
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
THREAD_PREFIX = "gpu_pool:"
PRUNABLE_STATUSES = ("released", "unavailable")  # terminal; dead_letter stays for operator replay
PRUNE_BATCH = 2000
LIVE_STATUSES = ("queued", "backlogged", "granted", "recalling", "retry_wait", "dead_letter", "unavailable")
ADVISORY_KEY = 0x6770755F706F6F6C  # "gpu_pool"


def pool_kwargs() -> dict[str, Any]:
    """Connection settings for the pool's Postgres pool. The search_path puts LangGraph's
    unqualified checkpoint tables in CHECKPOINT_SCHEMA; the projection tables (public) still
    resolve through the second entry."""
    from psycopg.rows import dict_row

    return {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
            "options": f"-c search_path={CHECKPOINT_SCHEMA},public"}


async def ensure_checkpoint_schema(conninfo: str) -> None:
    """Before the saver's setup(): a schema missing from search_path is skipped, so setup would
    silently create the tables in public again."""
    import psycopg

    async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
        await conn.execute("SET lock_timeout = '10s'")
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")


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
    async def prune_checkpoints(self, older_than: datetime) -> int: ...


class MemoryStore:
    """Same contract, in memory: tests and the eval."""

    async def prune_checkpoints(self, older_than):
        return 0  # the in-memory saver belongs to the test

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

    async def adopt_public_checkpoints(self) -> int:
        """Move lease threads written before CHECKPOINT_SCHEMA existed out of public's shared
        tables (idempotent; a no-op once done). Runs after saver.setup(), before any lease is
        resumed, so a lease granted by the previous process keeps its history."""
        moved = 0
        async with self.pool.connection() as conn:
            for table in CHECKPOINT_TABLES:
                exists = await (await conn.execute("SELECT to_regclass(%s) AS t", (f"public.{table}",))).fetchone()
                if not exists["t"]:
                    continue
                cols = [r["column_name"] for r in await (await conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
                    "AND column_name IN (SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name=%s) ORDER BY ordinal_position",
                    (CHECKPOINT_SCHEMA, table, table))).fetchall()]
                collist = ", ".join(cols)
                async with conn.transaction():
                    # SET LOCAL: a plain SET would stay on this pooled connection for its next user
                    await conn.execute("SET LOCAL lock_timeout = '10s'")
                    cur = await conn.execute(
                        f"INSERT INTO {CHECKPOINT_SCHEMA}.{table} ({collist}) SELECT {collist} FROM public.{table} "
                        f"WHERE thread_id LIKE %s ON CONFLICT DO NOTHING", (THREAD_PREFIX + "%",))
                    moved += cur.rowcount or 0
                    await conn.execute(f"DELETE FROM public.{table} WHERE thread_id LIKE %s", (THREAD_PREFIX + "%",))
        return moved

    async def prune_checkpoints(self, older_than, batch: int = PRUNE_BATCH) -> int:
        """Forget leases that ended (released or unavailable) before ``older_than``: their
        checkpoint thread AND their projection row, in batches. Dropping the row too keeps both
        tables about one retention window deep, so each pass only touches what aged out since
        the last one -- no scan that grows with all history. Dead letters are kept for operator
        replay. Backfill and the history walker reach back this far. Returns leases forgotten."""
        forgotten = 0
        while True:
            async with self.pool.connection() as conn, conn.transaction():
                rows = await (await conn.execute(
                    "SELECT lease_id FROM gpu_pool_leases WHERE status = ANY(%s) AND updated_at < %s "
                    "LIMIT %s FOR UPDATE SKIP LOCKED",
                    (list(PRUNABLE_STATUSES), older_than, batch))).fetchall()
                if not rows:
                    return forgotten
                threads = [THREAD_PREFIX + r["lease_id"] for r in rows]
                for table in reversed(CHECKPOINT_TABLES):
                    await conn.execute(f"DELETE FROM {CHECKPOINT_SCHEMA}.{table} WHERE thread_id = ANY(%s)", (threads,))
                await conn.execute("DELETE FROM gpu_pool_leases WHERE lease_id = ANY(%s)",
                                   ([r["lease_id"] for r in rows],))
            forgotten += len(rows)
            if len(rows) < batch:
                return forgotten

    async def check_schema(self) -> None:
        """The migrations are operator-applied (services/orion-sql-db/manual_migration_gpu_pool_v1.sql,
        then _v2_holds.sql). Refuse to start without them rather than run on an in-memory illusion:
        without v2 every hold/child row and every actuation state write would fail at runtime."""
        async with self.pool.connection() as conn:
            await conn.execute(f"SELECT lease_id, {', '.join(V2_LEASE_COLUMNS)} FROM gpu_pool_leases LIMIT 0")
            await conn.execute(f"SELECT card, {', '.join(V2_CARD_COLUMNS)} FROM gpu_pool_cards LIMIT 0")

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
        from psycopg.types.json import Jsonb

        cols = [c for c in CARD_COLUMNS if c in row]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "card")
        sql = (f"INSERT INTO gpu_pool_cards ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT (card) DO UPDATE SET {sets}")
        values = [Jsonb(row[c]) if isinstance(row[c], dict) else row[c] for c in cols]
        async with self.pool.connection() as conn:
            await conn.execute(sql, values)
