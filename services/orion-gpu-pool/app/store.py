"""The pool's fenced projection: one row per lease, one row per card.

The LangGraph checkpoint is the lease's history and the source of replay. This projection
exists so the scheduler reads "everything live" in one indexed query, and so there is exactly
one writer: the runtime holds a Postgres advisory lock for its whole life (``leader``), and
only the lock holder subscribes to the lease channels. This is the one direct Postgres writer
in the pool; lease *history* goes bus -> sql-writer (spec, "Transport and telemetry" item 4).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

LEASE_COLUMNS = (
    "lease_id", "request_id", "holder", "work_class", "priority", "kind", "status", "role",
    "attempt", "generation", "operator", "min_ctx_tokens", "needs_vision", "retryable", "created_at",
    "queued_since", "granted_at", "recall_by", "not_before", "deadline_at", "expires_at",
    "turn_correlation_id", "parent_lease_id", "hold_lease_id", "reason", "updated_at",
)
CARD_COLUMNS = ("card", "lent", "swapped_in", "swap_state", "cooldown_until", "last_active_at",
                "swap_role", "swap_generation", "swap_action", "residency_until", "loaded_at",
                "seen_ctx", "actuation_paused_at", "actuation_paused_by", "updated_at", "updated_by")
TABLE_KEYS = {"gpu_pool_leases": "lease_id", "gpu_pool_cards": "card"}
REQUIRED_COLUMNS = {"gpu_pool_leases": LEASE_COLUMNS, "gpu_pool_cards": CARD_COLUMNS}


@dataclass(frozen=True)
class AdditiveColumn:
    """One column a later projection migration added. ``ddl`` is exactly what follows the column
    name in the migration's ``ALTER TABLE .. ADD COLUMN IF NOT EXISTS`` (tests/test_schema_drift_gate.py
    holds the two in lockstep); ``default`` is what a row reads while the column is still missing."""
    table: str
    column: str
    ddl: str
    default: Any
    migration: str


# The pool applies these itself at boot (heal_schema), after taking the leader lock. The operator
# files stay the source a human runs; this list must match them column for column. ADDITIVE ONLY:
# nullable, or NOT NULL with a constant default (no table rewrite, instant ACCESS EXCLUSIVE lock).
# Anything else -- a new table, a type change, a drop, an index -- is operator-applied and a pool
# missing it refuses to boot. Indexes are deliberately not here (a CONCURRENTLY build cannot run in
# a transaction, and a plain one blocks writes for the whole build).
_V2 = "manual_migration_gpu_pool_v2_holds.sql"             # stage 4.3
_V3 = "manual_migration_gpu_pool_v3_actuation_pause.sql"   # stage 5.7: the emergency stop
BOOT_ADDITIVE_COLUMNS = (
    AdditiveColumn("gpu_pool_leases", "hold_lease_id", "text", None, _V2),
    AdditiveColumn("gpu_pool_cards", "swap_role", "text", None, _V2),
    AdditiveColumn("gpu_pool_cards", "swap_generation", "integer NOT NULL DEFAULT 0", 0, _V2),
    AdditiveColumn("gpu_pool_cards", "swap_action", "jsonb", None, _V2),
    AdditiveColumn("gpu_pool_cards", "residency_until", "timestamptz", None, _V2),
    AdditiveColumn("gpu_pool_cards", "loaded_at", "timestamptz", None, _V2),
    AdditiveColumn("gpu_pool_cards", "seen_ctx", "jsonb", None, _V2),
    AdditiveColumn("gpu_pool_cards", "actuation_paused_at", "timestamptz", None, _V3),
    AdditiveColumn("gpu_pool_cards", "actuation_paused_by", "text", None, _V3),
)
_BOOT_BY_KEY = {(c.table, c.column): c for c in BOOT_ADDITIVE_COLUMNS}
# Boot: nothing is serving yet, so a few seconds of queueing behind the ALTER costs no RPC. While
# serving, a waiting ALTER queues the pool's own lease writes behind it, so the retry waits less.
HEAL_LOCK_TIMEOUT_MS = 3000
HEAL_RETRY_LOCK_TIMEOUT_MS = 300
HEAL_STATEMENT_TIMEOUT_MS = 15000
HEAL_RETRY_BACKOFF_SEC = (5.0, 10.0, 30.0, 60.0, 120.0)


class SchemaNotHealable(RuntimeError):
    """The projection lacks something the pool will not create itself (a table, or a column not in
    BOOT_ADDITIVE_COLUMNS). The operator migration is required; the pool refuses to boot."""
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

logger = logging.getLogger("orion-gpu-pool.store")


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
    async def set_actuation_paused(self, at: datetime | None, by: str | None, now: datetime, actor: str) -> None: ...
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

    async def set_actuation_paused(self, at, by, now, actor):
        for card, row in self._cards.items():
            self._cards[card] = {**row, "actuation_paused_at": at, "actuation_paused_by": by,
                                 "updated_at": now, "updated_by": actor}


class PostgresStore:
    def __init__(self, pool: Any, conninfo: str | None = None):
        self.pool = pool
        self.conninfo = conninfo
        self._leader_conn: Any = None
        self._init_schema_state()

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
            for r in rows:
                self._overlay.pop(("gpu_pool_leases", r["lease_id"]), None)
            forgotten += len(rows)
            if len(rows) < batch:
                return forgotten

    # --- schema: boot self-heal, and a degraded mode that keeps serving -----------------------
    #
    # Three times (2026-09-26 v2, 2026-09-30 v3, and the stage-4 dry run) a pool image deployed
    # before its additive migration refused to boot, and because the gateway leases every LLM call
    # from the pool that was a total LLM outage until someone ran one ALTER TABLE. So the pool now
    # runs those ALTERs itself (heal_schema). If it cannot get the table lock in time (a pg_dump, a
    # long transaction), it does NOT exit: it serves in a degraded mode where the missing columns
    # live in an in-process overlay (so holds, swap state and the emergency stop still behave for
    # this process), retries in the background, and writes the overlay through once healed. What
    # a degraded pool loses is only persistence of those columns across a restart -- and it says
    # so, loudly, in the log and in /health (schema_status).

    async def check_schema(self) -> list[tuple[str, str]]:
        """Missing projection columns the pool can add itself, in BOOT_ADDITIVE_COLUMNS order.
        Raises SchemaNotHealable if a table, or any other column the store writes, is missing:
        that needs the operator migration and the pool must not start without it."""
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)", (list(REQUIRED_COLUMNS),))).fetchall()
        have = {(r["table_name"], r["column_name"]) for r in rows}
        tables = {t for t, _ in have}
        absent_tables = [t for t in REQUIRED_COLUMNS if t not in tables]
        if absent_tables:
            raise SchemaNotHealable(
                f"projection tables missing: {absent_tables} -- apply services/orion-sql-db/"
                f"manual_migration_gpu_pool_v1.sql (the pool creates columns, never tables)")
        missing = [(t, c) for t, cols in REQUIRED_COLUMNS.items() for c in cols if (t, c) not in have]
        unhealable = [k for k in missing if k not in _BOOT_BY_KEY]
        if unhealable:
            raise SchemaNotHealable(f"projection columns missing that boot will not add: {unhealable} -- "
                                    f"apply the operator migration for them")
        order = {k: i for i, k in enumerate(_BOOT_BY_KEY)}
        return sorted(missing, key=order.__getitem__)

    def _init_schema_state(self) -> None:
        if not hasattr(self, "_missing"):
            self._missing: set[tuple[str, str]] = set()
            self._overlay: dict[tuple[str, Any], dict[str, Any]] = {}
            self._schema: dict[str, Any] = {"state": "unchecked", "missing": [], "applied": [],
                                            "attempts": 0, "last_error": None, "last_attempt_at": None,
                                            "degraded_since": None, "healed_at": None}
            self._db_missing: list[tuple[str, str]] = []
            # Serializes every degraded-mode write (overlay update + its SQL) against the flush, so
            # a flush can never run between a write's overlay step and its INSERT (which would
            # UPDATE a row that does not exist yet and drop the value). Only taken while degraded.
            self._heal_lock = asyncio.Lock()

    def schema_status(self) -> dict[str, Any]:
        """For /health. state: ok | degraded (serving; ``in_memory`` columns are not persisted, so
        they do not survive a restart). ``missing`` is what the database still lacks."""
        self._init_schema_state()
        return {**self._schema, "missing": [f"{t}.{c}" for t, c in self._db_missing],
                "in_memory": [f"{t}.{c}" for t, c in sorted(self._missing)], "overlay_rows": len(self._overlay)}

    @property
    def schema_degraded(self) -> bool:
        self._init_schema_state()
        return bool(self._missing)

    async def heal_schema(self, *, lock_timeout_ms: int = HEAL_LOCK_TIMEOUT_MS,
                          statement_timeout_ms: int = HEAL_STATEMENT_TIMEOUT_MS) -> dict[str, Any]:
        """Add every missing BOOT_ADDITIVE_COLUMNS column, one ALTER per table, under
        lock_timeout + statement_timeout. Call it only as the leader (single writer). Never raises
        for a lock or DDL failure -- it records it and leaves the pool degraded; raises only
        SchemaNotHealable (and a dead database)."""
        self._init_schema_state()
        missing = await self.check_schema()
        self._schema["attempts"] += 1
        self._schema["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
        errors = []
        # One ALTER per table (atomic per table): under a dump each statement waits its full
        # lock_timeout, and while it waits Postgres queues every other read/write of the table
        # behind it -- so stop at the first failure rather than pay that once per column.
        for table in REQUIRED_COLUMNS:
            cols = [_BOOT_BY_KEY[k] for k in missing if k[0] == table]
            if not cols:
                continue
            adds = ", ".join(f"ADD COLUMN IF NOT EXISTS {c.column} {c.ddl}" for c in cols)
            try:
                async with self.pool.connection() as conn, conn.transaction():
                    # SET LOCAL: a plain SET would stay on this pooled connection for its next user
                    await conn.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'")
                    await conn.execute(f"SET LOCAL statement_timeout = '{int(statement_timeout_ms)}ms'")
                    # public.: check_schema reads public, and search_path puts gpu_pool first
                    await conn.execute(f"ALTER TABLE public.{table} {adds}")
            except Exception as exc:  # noqa: BLE001 -- recorded, retried; never a crash-loop
                errors.append(f"{table}: {type(exc).__name__}: {exc}"[:300])
                break
            for c in cols:
                self._schema["applied"].append(f"{c.table}.{c.column}")
                logger.warning("gpu_pool_schema_healed column=%s.%s ddl=%r migration=%s",
                               c.table, c.column, c.ddl, c.migration)
        still = await self.check_schema()
        self._db_missing = still
        self._missing.update(still)
        if still:
            self._schema["last_error"] = "; ".join(errors)[:1000] or "columns still missing"
            if self._schema["state"] != "degraded":
                self._schema["state"] = "degraded"
                self._schema["degraded_since"] = datetime.now(timezone.utc).isoformat()
            logger.critical(
                "gpu_pool_schema_degraded missing=%s err=%s -- SERVING with these columns held in memory "
                "only (not persisted across a restart); retrying in the background. Operator fix: apply %s",
                [f"{t}.{c}" for t, c in still], self._schema["last_error"],
                sorted({_BOOT_BY_KEY[k].migration for k in still}))
        else:
            async with self._heal_lock:
                await self._flush_overlay()
            if self._schema["state"] == "degraded":
                self._schema["healed_at"] = datetime.now(timezone.utc).isoformat()
                logger.warning("gpu_pool_schema_recovered applied=%s", self._schema["applied"])
            self._schema.update(state="ok", last_error=None, degraded_since=None)
        return self.schema_status()

    async def heal_forever(self, stop: asyncio.Event, backoff: tuple[float, ...] = HEAL_RETRY_BACKOFF_SEC) -> None:
        """Background retry while degraded. Returns once healed (or on stop)."""
        attempt = 0
        while self.schema_degraded and not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=backoff[min(attempt, len(backoff) - 1)])
                return
            except asyncio.TimeoutError:
                pass
            attempt += 1
            try:
                await self.heal_schema(lock_timeout_ms=HEAL_RETRY_LOCK_TIMEOUT_MS)
            except Exception as exc:  # noqa: BLE001 -- keep serving; the log carries it
                self._schema["last_error"] = f"{type(exc).__name__}: {exc}"[:1000]
                logger.exception("gpu_pool_schema_heal_retry_failed")

    async def _flush_overlay(self) -> None:
        """Write the in-memory values through, then stop overlaying. Caller holds _heal_lock, so no
        degraded write is between its overlay step and its SQL, and none starts until _missing is
        cleared (after which writes carry the real columns). One transaction: all or nothing."""
        from psycopg.types.json import Jsonb

        if self._overlay:
            async with self.pool.connection() as conn, conn.transaction():
                for (table, key), values in list(self._overlay.items()):   # prune may pop meanwhile
                    cols = list(values)
                    sets = ", ".join(f"{c}=%s" for c in cols)
                    args = [Jsonb(values[c]) if isinstance(values[c], dict) else values[c] for c in cols]
                    # 0 rows = the row was pruned meanwhile; nothing to keep
                    await conn.execute(f"UPDATE public.{table} SET {sets} WHERE {TABLE_KEYS[table]}=%s",
                                       [*args, key])
        self._overlay.clear()
        self._missing.clear()

    def _split(self, table: str, row: dict[str, Any]) -> list[str]:
        """Columns to write to Postgres; a missing column's value goes to the overlay instead."""
        self._init_schema_state()
        cols = [c for c in REQUIRED_COLUMNS[table] if c in row]
        if not self._missing:
            return cols
        key = (table, row[TABLE_KEYS[table]])
        # A default value for a row with nothing overlaid reads the same without an entry: skip it,
        # or every lease (hold_lease_id=None) would grow the overlay for as long as it is degraded.
        held = {c: row[c] for c in cols if (table, c) in self._missing
                and (key in self._overlay or row[c] != _BOOT_BY_KEY[(table, c)].default)}
        if held:
            self._overlay.setdefault(key, {}).update(held)
        return [c for c in cols if (table, c) not in self._missing]

    def _merge(self, table: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
        self._init_schema_state()
        if row is None or not self._missing:
            return row
        out = dict(row)
        for t, c in self._missing:
            if t == table:
                out.setdefault(c, _BOOT_BY_KEY[(t, c)].default)
        out.update(self._overlay.get((table, row[TABLE_KEYS[table]]), {}))
        return out

    async def _degraded_write(self, fn, *args):
        """Every projection write goes through here. Healthy: straight through. Degraded: under
        _heal_lock (see _init_schema_state), re-checked inside it, since a flush may have healed us
        while this write waited."""
        if not self._missing:
            return await fn(*args)
        async with self._heal_lock:
            return await fn(*args)

    async def upsert_lease(self, row):
        await self._degraded_write(self._upsert_lease, row)

    async def _upsert_lease(self, row):
        cols = self._split("gpu_pool_leases", row)
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "lease_id")
        sql = (f"INSERT INTO gpu_pool_leases ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT (lease_id) DO UPDATE SET {sets}")
        async with self.pool.connection() as conn:
            await conn.execute(sql, [row[c] for c in cols])

    async def _one(self, sql, args):
        async with self.pool.connection() as conn:
            return await (await conn.execute(sql, args)).fetchone()

    async def lease(self, lease_id):
        return self._merge("gpu_pool_leases", await self._one("SELECT * FROM gpu_pool_leases WHERE lease_id=%s",
                                                              (lease_id,)))

    async def lease_by_request(self, request_id):
        return self._merge("gpu_pool_leases", await self._one(
            "SELECT * FROM gpu_pool_leases WHERE request_id=%s", (request_id,)))

    async def live_leases(self):
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT * FROM gpu_pool_leases WHERE status = ANY(%s)", (list(LIVE_STATUSES),))).fetchall()
        return [self._merge("gpu_pool_leases", r) for r in rows]

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
            rows = await (await conn.execute(sql, [*args, limit])).fetchall()
        return [self._merge("gpu_pool_leases", r) for r in rows]

    async def cards(self):
        async with self.pool.connection() as conn:
            rows = await (await conn.execute("SELECT * FROM gpu_pool_cards")).fetchall()
        return [self._merge("gpu_pool_cards", r) for r in rows]

    async def upsert_card(self, row):
        await self._degraded_write(self._upsert_card, row)

    async def _upsert_card(self, row):
        from psycopg.types.json import Jsonb

        cols = self._split("gpu_pool_cards", row)
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "card")
        sql = (f"INSERT INTO gpu_pool_cards ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT (card) DO UPDATE SET {sets}")
        values = [Jsonb(row[c]) if isinstance(row[c], dict) else row[c] for c in cols]
        async with self.pool.connection() as conn:
            await conn.execute(sql, values)

    async def set_actuation_paused(self, at, by, now, actor):
        """The emergency stop (stage 5.7), all card rows in one statement: every row agrees, or none.
        Degraded (v3 columns missing): the stop still holds for this process via the overlay, and
        is written through when the columns arrive -- it is never refused."""
        await self._degraded_write(self._set_actuation_paused, at, by, now, actor)

    async def _set_actuation_paused(self, at, by, now, actor):
        if ("gpu_pool_cards", "actuation_paused_at") in self._missing:
            async with self.pool.connection() as conn:
                cards = [r["card"] for r in await (await conn.execute("SELECT card FROM gpu_pool_cards")).fetchall()]
                await conn.execute("UPDATE gpu_pool_cards SET updated_at=%s, updated_by=%s", [now, actor])
            for card in cards:
                self._overlay.setdefault(("gpu_pool_cards", card), {}).update(
                    actuation_paused_at=at, actuation_paused_by=by)
            logger.critical("gpu_pool_actuation_pause_not_persisted paused=%s by=%s -- schema degraded; held in "
                            "memory until the v3 columns exist", at is not None, by)
            return
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE gpu_pool_cards SET actuation_paused_at=%s, actuation_paused_by=%s, "
                               "updated_at=%s, updated_by=%s", [at, by, now, actor])
