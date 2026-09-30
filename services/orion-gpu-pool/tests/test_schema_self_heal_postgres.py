"""Boot schema self-heal on a real Postgres (the 2026-09-26 / 2026-09-30 outages, reproduced).

A pool deployed before its additive migration used to refuse to boot, and every LLM call leases
through the pool. Now: it adds the columns itself after taking the leader lock; if a lock holder
(pg_dump, a long transaction) keeps it from doing so in time, it SERVES degraded -- missing
columns held in memory, loud in /health -- and heals once the lock is released.

Skipped unless GPU_POOL_TEST_POSTGRES_URI points at a scratch database (CI provides one)."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

URI = os.environ.get("GPU_POOL_TEST_POSTGRES_URI")
pytestmark = pytest.mark.skipif(not URI, reason="GPU_POOL_TEST_POSTGRES_URI not set")
SQL_DB = Path(__file__).resolve().parents[3] / "services/orion-sql-db"
V1, V2, V3 = (SQL_DB / f"manual_migration_gpu_pool_{n}.sql" for n in ("v1", "v2_holds", "v3_actuation_pause"))
CARD_BOOT_COLUMNS = {"swap_role", "swap_generation", "swap_action", "residency_until", "loaded_at", "seen_ctx",
                     "actuation_paused_at", "actuation_paused_by"}


async def _apply(conn, path: Path) -> None:
    sql = "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("--"))
    for statement in (s.strip() for s in sql.split(";")):
        if statement:
            await conn.execute(statement)


async def _connect():
    import psycopg
    from psycopg.rows import dict_row

    return await psycopg.AsyncConnection.connect(URI, autocommit=True, row_factory=dict_row)


async def _v1_only(*, with_rows: bool = True) -> None:
    """The live shape before stage 4.3: v1 tables with rows, no v2/v3 columns."""
    async with await _connect() as conn:
        await conn.execute("DROP SCHEMA IF EXISTS gpu_pool CASCADE")
        await conn.execute("DROP TABLE IF EXISTS public.checkpoints, public.checkpoint_blobs, "
                           "public.checkpoint_writes, public.checkpoint_migrations, gpu_pool_leases, gpu_pool_cards")
        await _apply(conn, V1)
        if with_rows:
            await conn.execute("INSERT INTO gpu_pool_leases (lease_id, request_id, holder, work_class, priority, kind, "
                               "status, created_at) VALUES ('old', 'r-old', 'h', 'fast', 'system', 'request', 'granted', now())")
            await conn.execute("INSERT INTO gpu_pool_cards (card, swapped_in) VALUES ('gpu2', '{agent-gpu2}')")


async def _pool():
    from psycopg_pool import AsyncConnectionPool

    from app.store import ensure_checkpoint_schema, pool_kwargs

    await ensure_checkpoint_schema(URI)
    pool = AsyncConnectionPool(conninfo=URI, min_size=1, max_size=6, open=False, kwargs=pool_kwargs())
    await pool.open()
    return pool


async def _columns(table: str) -> dict[str, tuple]:
    async with await _connect() as conn:
        rows = await (await conn.execute(
            "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s", (table,))).fetchall()
    return {r["column_name"]: (r["data_type"], r["is_nullable"], r["column_default"]) for r in rows}


async def _hold_dump_lock(*tables: str):
    """What pg_dump does: ACCESS SHARE for the whole dump. ADD COLUMN needs ACCESS EXCLUSIVE."""
    conn = await _connect()
    await conn.execute("BEGIN")
    await conn.execute(f"LOCK TABLE {', '.join(tables)} IN ACCESS SHARE MODE")
    return conn


def test_boot_on_a_db_missing_v2_and_v3_self_heals_to_the_operator_migration_shape():
    async def go():
        from app.store import PostgresStore

        await _v1_only()
        pool = await _pool()
        store = PostgresStore(pool, conninfo=URI)
        try:
            await store.leader()                           # single writer first, as main.py does
            assert len(await store.check_schema()) == 9    # hold_lease_id + 8 card columns
            status = await store.heal_schema()
            assert status["state"] == "ok" and status["missing"] == [] and len(status["applied"]) == 9
            assert not store.schema_degraded
            again = await store.heal_schema()              # idempotent: nothing left to apply
            assert again["state"] == "ok" and again["applied"] == status["applied"]
            assert (await store.lease("old"))["status"] == "granted" and (await store.lease("old"))["hold_lease_id"] is None
            card = {r["card"]: r for r in await store.cards()}["gpu2"]
            assert card["swapped_in"] == ["agent-gpu2"] and card["swap_generation"] == 0
            assert card["actuation_paused_at"] is None     # additive: healing pauses nothing
        finally:
            await store.close()
            await pool.close()
        healed = (await _columns("gpu_pool_leases"), await _columns("gpu_pool_cards"))

        # the operator path yields the identical schema, and re-running it over a healed DB is a no-op
        async with await _connect() as conn:
            await _apply(conn, V2)
            await _apply(conn, V3)
        assert (await _columns("gpu_pool_leases"), await _columns("gpu_pool_cards")) == healed
        await _v1_only()
        async with await _connect() as conn:
            await _apply(conn, V2)
            await _apply(conn, V3)
        assert (await _columns("gpu_pool_leases"), await _columns("gpu_pool_cards")) == healed
    asyncio.run(go())


def test_a_missing_table_still_refuses_to_boot():
    """Creating tables is the operator's job: only additive columns are healed at boot."""
    async def go():
        from app.store import PostgresStore, SchemaNotHealable

        async with await _connect() as conn:
            await conn.execute("DROP TABLE IF EXISTS gpu_pool_leases, gpu_pool_cards")
        pool = await _pool()
        try:
            with pytest.raises(SchemaNotHealable, match="manual_migration_gpu_pool_v1"):
                await PostgresStore(pool).heal_schema()
        finally:
            await pool.close()
    asyncio.run(go())


def test_a_missing_v1_column_is_not_healed_and_refuses_to_boot():
    async def go():
        from app.store import PostgresStore, SchemaNotHealable

        await _v1_only()
        async with await _connect() as conn:
            await conn.execute("ALTER TABLE gpu_pool_cards DROP COLUMN cooldown_until")
        pool = await _pool()
        try:
            with pytest.raises(SchemaNotHealable, match="cooldown_until"):
                await PostgresStore(pool).heal_schema()
        finally:
            await pool.close()
    asyncio.run(go())


def test_lock_contention_serves_degraded_loudly_then_heals_and_writes_through():
    """A dump holds both projection tables: boot must not exit or hang. The runtime starts, grants a hold
    and a child, attach stays idempotent, the emergency stop holds -- all with the card columns
    missing -- and /health's schema block says degraded. Releasing the lock heals it, and the
    in-memory values land in Postgres, so a restarted pool sees the child link and the pause."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from app.runtime import PoolRuntime
    from app.store import PostgresStore
    from orion.gpu_pool.config import load_pool_config
    from orion.gpu_pool.discovery import Probe, load_profiles
    from orion.gpu_pool.lease_graph import build_lease_graph
    from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1

    class Bus:
        async def publish(self, channel, env):
            pass

        def record_hop_success(self, *a):
            pass

        def record_hop_timeout(self, *a):
            pass

    async def go():
        cfg = load_pool_config()
        await _v1_only()
        pool = await _pool()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()

        async def prober(role, url, kind, health):
            return Probe(kind == "service")

        def runtime(store):
            return PoolRuntime(cfg=cfg, profiles=load_profiles(), store=store,
                               graph=build_lease_graph(lambda: cfg, saver), prober=prober, bus=Bus())

        store = PostgresStore(pool, conninfo=URI)
        dump = await _hold_dump_lock("gpu_pool_leases", "gpu_pool_cards")
        try:
            await store.leader()
            t = time.monotonic()
            status = await store.heal_schema(lock_timeout_ms=300)      # no raise, no hang
            assert time.monotonic() - t < 10
            assert status["state"] == "degraded" and status["degraded_since"]
            assert set(status["missing"]) == {f"gpu_pool_cards.{c}" for c in CARD_BOOT_COLUMNS} | \
                {"gpu_pool_leases.hold_lease_id"}
            assert status["applied"] == []
            assert "lock timeout" in status["last_error"]
            assert store.schema_degraded

            rt = runtime(store)
            await rt.start()
            await rt.tick()
            h = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="world", holder="durable-runs:r1",
                                                   request_id="r1:1", kind="hold", retryable=True))
            assert h.status == "granted"
            req = GpuLeaseRequestV1(verb="attach", work_class="world", holder="gw", request_id="c1",
                                    hold_lease_id=h.lease_id, hold_generation=h.grant.generation)
            c = await rt.attach(req)
            assert c.status == "granted"
            retry = await rt.attach(req)                   # idempotent while degraded, not request_id_conflict
            assert retry.status == "granted" and retry.lease_id == c.lease_id
            assert (await store.lease(c.lease_id))["hold_lease_id"] == h.lease_id
            rt._ctx_seen["agent-gpu2"] = 131072            # a card column written through upsert_card
            await rt._save_seen_ctx()
            assert {r["card"]: r for r in await store.cards()}["gpu2"]["seen_ctx"]["agent-gpu2"] == 131072
            paused = await rt.control(GpuPoolControlV1(verb="pause_actuation", actor="juniper"))
            assert paused.ok
            assert all(r["actuation_paused_by"] == "juniper" for r in await store.cards())
            assert store.schema_status()["overlay_rows"] > 0

            stop = asyncio.Event()
            healer = asyncio.create_task(store.heal_forever(stop, backoff=(0.2,)))
            await asyncio.sleep(1.0)
            assert store.schema_status()["state"] == "degraded" and store.schema_status()["attempts"] >= 2
            await dump.execute("COMMIT")                    # the dump finishes
            await asyncio.wait_for(healer, timeout=15)
            status = store.schema_status()
            assert status["state"] == "ok" and status["missing"] == [] and status["in_memory"] == []
            assert status["overlay_rows"] == 0 and status["healed_at"]

            async with await _connect() as conn:            # written through, not just in memory
                child = await (await conn.execute("SELECT hold_lease_id FROM gpu_pool_leases WHERE lease_id=%s",
                                                  (c.lease_id,))).fetchone()
                assert child["hold_lease_id"] == h.lease_id
                cards = await (await conn.execute("SELECT * FROM gpu_pool_cards")).fetchall()
                assert {r["card"]: r for r in cards}["gpu2"]["seen_ctx"]["agent-gpu2"] == 131072
                assert cards and all(r["actuation_paused_at"] is not None and r["actuation_paused_by"] == "juniper"
                                     for r in cards)

            rt2 = runtime(PostgresStore(pool))             # a restart after the heal
            await rt2.start()
            assert rt2.paused is not None and rt2.paused["by"] == "juniper"
            assert (await rt2.attach(req)).lease_id == c.lease_id
        finally:
            if not dump.closed:
                await dump.close()
            await store.close()
            await pool.close()
    asyncio.run(go())


def test_heal_waits_for_a_degraded_write_in_flight_so_a_new_rows_value_is_not_lost():
    """Review finding (reproduced before the fix): a write had put hold_lease_id in the overlay and
    was awaiting its connection; the heal's flush ran in that gap, UPDATEd a row that did not exist
    yet (0 rows), cleared the overlay -- then the INSERT ran without the column. The child's hold
    link was gone from both Postgres and memory. The heal must wait for the write."""
    from datetime import datetime, timezone

    from app.store import PostgresStore

    async def go():
        await _v1_only()
        pool = await _pool()
        store = PostgresStore(pool, conninfo=URI)
        dump = await _hold_dump_lock("gpu_pool_leases", "gpu_pool_cards")
        try:
            await store.leader()
            assert (await store.heal_schema(lock_timeout_ms=200))["state"] == "degraded"
            await dump.execute("COMMIT")
            entered, gate = asyncio.Event(), asyncio.Event()

            async def split_then_wait_then_insert(row):   # _upsert_lease's shape, with the gap held open
                cols = store._split("gpu_pool_leases", row)
                entered.set()
                await gate.wait()
                async with pool.connection() as conn:
                    await conn.execute(f"INSERT INTO gpu_pool_leases ({', '.join(cols)}) "
                                       f"VALUES ({', '.join(['%s'] * len(cols))})", [row[c] for c in cols])

            store._upsert_lease = split_then_wait_then_insert
            now = datetime.now(timezone.utc)
            write = asyncio.create_task(store.upsert_lease({
                "lease_id": "child", "request_id": "c1", "holder": "h", "work_class": "fast", "priority": "system",
                "kind": "request", "status": "granted", "created_at": now, "hold_lease_id": "HOLD"}))
            await entered.wait()
            heal = asyncio.create_task(store.heal_schema())
            await asyncio.sleep(0.5)
            assert not heal.done()                         # the columns exist, but the flush waits
            gate.set()
            await write
            assert (await heal)["state"] == "ok"
            async with await _connect() as conn:
                row = await (await conn.execute("SELECT hold_lease_id FROM gpu_pool_leases WHERE lease_id='child'")).fetchone()
            assert row["hold_lease_id"] == "HOLD"
        finally:
            if not dump.closed:
                await dump.close()
            await store.close()
            await pool.close()
    asyncio.run(go())


def test_a_background_retry_under_a_dump_stalls_lease_writes_briefly_not_per_column():
    """While serving, a waiting ALTER queues every write to its table behind it. The retry uses
    one ALTER per table, stops at the first lock failure, and a short lock_timeout: a lease write
    racing it is delayed well under a second, not ~9 x lock_timeout."""
    from datetime import datetime, timezone

    from app.store import HEAL_RETRY_LOCK_TIMEOUT_MS, PostgresStore

    async def go():
        await _v1_only()
        pool = await _pool()
        store = PostgresStore(pool, conninfo=URI)
        dump = await _hold_dump_lock("gpu_pool_leases", "gpu_pool_cards")
        try:
            await store.leader()
            await store.heal_schema(lock_timeout_ms=200)
            now = datetime.now(timezone.utc)
            heal = asyncio.create_task(store.heal_schema(lock_timeout_ms=HEAL_RETRY_LOCK_TIMEOUT_MS))
            await asyncio.sleep(0.05)                      # the ALTER is now queued on gpu_pool_leases
            t = time.monotonic()
            await store.upsert_lease({"lease_id": "x", "request_id": "rx", "holder": "h", "work_class": "fast",
                                      "priority": "system", "kind": "request", "status": "queued", "created_at": now})
            waited = time.monotonic() - t
            status = await heal
            assert status["state"] == "degraded" and status["last_error"].count("LockNotAvailable") == 1
            assert waited < 1.0, waited
        finally:
            await dump.close()
            await store.close()
            await pool.close()
    asyncio.run(go())
