"""PostgresStore + the real migration + the real LangGraph Postgres saver, end to end.

Skipped unless GPU_POOL_TEST_POSTGRES_URI points at a scratch database (CI provides one)."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

URI = os.environ.get("GPU_POOL_TEST_POSTGRES_URI")
pytestmark = pytest.mark.skipif(not URI, reason="GPU_POOL_TEST_POSTGRES_URI not set")
MIGRATION = Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_gpu_pool_v1.sql"


async def _pool():
    """The production connection setup (app.store.pool_kwargs + ensure_checkpoint_schema), on a
    scratch database reset to: no pool schema, no shared checkpoint tables, fresh projection."""
    import psycopg
    from psycopg_pool import AsyncConnectionPool

    from app.store import ensure_checkpoint_schema, pool_kwargs

    async with await psycopg.AsyncConnection.connect(URI, autocommit=True) as conn:
        await conn.execute("DROP SCHEMA IF EXISTS gpu_pool CASCADE")
        await conn.execute("DROP TABLE IF EXISTS public.checkpoints, public.checkpoint_blobs, "
                           "public.checkpoint_writes, public.checkpoint_migrations")
    await ensure_checkpoint_schema(URI)
    pool = AsyncConnectionPool(conninfo=URI, min_size=1, max_size=6, open=False, kwargs=pool_kwargs())
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS gpu_pool_leases, gpu_pool_cards")
            sql = "\n".join(l for l in MIGRATION.read_text().splitlines() if not l.strip().startswith("--"))
            for statement in (s.strip() for s in sql.split(";")):
                if statement:
                    await conn.execute(statement)
    except BaseException:
        await pool.close()
        raise
    return pool


def test_projection_roundtrip_and_single_writer_lock():
    async def go():
        from app.store import PostgresStore

        pool = await _pool()
        try:
            await _roundtrip(pool)
        finally:
            await pool.close()

    async def _roundtrip(pool):
        from app.store import PostgresStore

        store = PostgresStore(pool)
        await store.check_schema()
        now = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
        row = {"lease_id": "a", "request_id": "ra", "holder": "h", "work_class": "fast", "priority": "system",
               "kind": "request", "status": "queued", "created_at": now, "updated_at": now}
        await store.upsert_lease(row)
        await store.upsert_lease({**row, "status": "granted", "role": "fast", "generation": 1,
                                  "granted_at": now, "expires_at": now + timedelta(seconds=30)})
        got = await store.lease("a")
        assert got["status"] == "granted" and got["generation"] == 1 and got["role"] == "fast"
        assert (await store.lease_by_request("ra"))["lease_id"] == "a"
        assert [r["lease_id"] for r in await store.live_leases()] == ["a"]
        await store.upsert_lease({**row, "status": "released"})
        assert await store.live_leases() == []
        assert len(await store.find_leases(work_class="fast", holder=None, status="released",
                                           since=now - timedelta(hours=1), until=None, limit=10)) == 1
        await store.upsert_card({"card": "gpu2", "swapped_in": ["agent-gpu2"], "lent": False,
                                 "updated_at": now, "updated_by": "t"})
        assert (await store.cards())[0]["swapped_in"] == ["agent-gpu2"]

        await store.leader()
        assert await store.leader_alive()
        second = PostgresStore(pool)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(second.leader(), timeout=1.0)
        await store.close()
        assert not await store.leader_alive()
        await asyncio.wait_for(second.leader(), timeout=10.0)
        assert await second.leader_alive()
        await second.close()
    asyncio.run(go())


def test_runtime_on_postgres_survives_restart():
    async def go():
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from app.runtime import PoolRuntime
        from app.store import PostgresStore
        from orion.gpu_pool.config import load_pool_config
        from orion.gpu_pool.discovery import Probe, load_profiles
        from orion.gpu_pool.lease_graph import build_lease_graph
        from orion.schemas.gpu_pool import GpuLeaseRequestV1

        cfg = load_pool_config()
        pool = await _pool()
        try:
            await _restart(pool, cfg, AsyncPostgresSaver, PoolRuntime, PostgresStore, Probe, load_profiles,
                           build_lease_graph, GpuLeaseRequestV1)
        finally:
            await pool.close()

    async def _restart(pool, cfg, AsyncPostgresSaver, PoolRuntime, PostgresStore, Probe, load_profiles,
                       build_lease_graph, GpuLeaseRequestV1):
        saver = AsyncPostgresSaver(pool)
        await saver.setup()

        async def prober(role, url, kind, health):
            return Probe(kind == "service")

        def runtime():
            return PoolRuntime(cfg=cfg, profiles=load_profiles(), store=PostgresStore(pool),
                               graph=build_lease_graph(lambda: cfg, saver), prober=prober)

        rt = runtime()
        await rt.start()
        await rt.tick()
        a = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="pg1"))
        b = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="pg2"))
        assert a.status == "granted" and b.status == "queued"

        rt2 = runtime()                           # restart on the same database
        await rt2.start()
        await rt2.tick()
        assert (await rt2.release(a.lease_id, "ok")).status == "ok"
        assert (await rt2.store.lease(b.lease_id))["status"] == "granted"
        path = [h["event"] for h in await rt2.history(b.lease_id)]
        assert path == ["admit", "grant"]
    asyncio.run(go())


def _run_with_runtime(body):
    """Build a real PoolRuntime on Postgres (saver in the pool schema) and run ``body(rt, pool)``."""
    async def go():
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from app.runtime import PoolRuntime
        from app.store import PostgresStore
        from orion.gpu_pool.config import load_pool_config
        from orion.gpu_pool.discovery import Probe, load_profiles
        from orion.gpu_pool.lease_graph import build_lease_graph

        cfg = load_pool_config()
        pool = await _pool()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()

            async def prober(role, url, kind, health):
                return Probe(kind == "service")

            rt = PoolRuntime(cfg=cfg, profiles=load_profiles(), store=PostgresStore(pool),
                             graph=build_lease_graph(lambda: cfg, saver), prober=prober)
            await rt.start()
            await rt.tick()
            await body(rt, pool)
        finally:
            await pool.close()
    asyncio.run(go())


async def _count(pool, sql, args=()):
    async with pool.connection() as conn:
        return (await (await conn.execute(sql, args)).fetchone())["n"]


def test_lease_threads_live_in_their_own_schema_not_the_shared_table():
    """durable-runs' resume sweep lists every row of public.checkpoints: pool threads there made
    that scan grow with every inference and stalled lease RPCs behind it (2026-09-25)."""
    from orion.schemas.gpu_pool import GpuLeaseRequestV1

    async def body(rt, pool):
        a = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="s1"))
        assert a.status == "granted"
        assert await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoints WHERE thread_id=%s",
                            (f"gpu_pool:{a.lease_id}",)) > 0
        assert await _count(pool, "SELECT count(*) AS n FROM pg_tables WHERE schemaname='public' "
                                  "AND tablename='checkpoints'") == 0
        assert [h["event"] for h in await rt.history(a.lease_id)] == ["admit", "grant"]
    _run_with_runtime(body)


def test_boot_adopts_lease_threads_left_in_the_shared_tables_and_leaves_others():
    from orion.schemas.gpu_pool import GpuLeaseRequestV1

    async def body(rt, pool):
        a = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="s2"))
        thread = f"gpu_pool:{a.lease_id}"
        async with pool.connection() as conn:
            # Recreate the pre-schema layout: this lease's thread (and a durable-runs thread) in public.
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                await conn.execute(f"CREATE TABLE public.{table} (LIKE gpu_pool.{table} INCLUDING ALL)")
                await conn.execute(f"INSERT INTO public.{table} SELECT * FROM gpu_pool.{table}")
                await conn.execute(f"DELETE FROM gpu_pool.{table}")
            await conn.execute("INSERT INTO public.checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
                               "metadata) VALUES ('durable-run-1', '', 'c1', '{}', '{}')")
        moved = await rt.store.adopt_public_checkpoints()
        assert moved > 0
        assert await rt.store.adopt_public_checkpoints() == 0          # idempotent
        assert await _count(pool, "SELECT count(*) AS n FROM public.checkpoints WHERE thread_id LIKE 'gpu_pool:%%'") == 0
        assert await _count(pool, "SELECT count(*) AS n FROM public.checkpoints WHERE thread_id='durable-run-1'") == 1
        assert await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoints WHERE thread_id=%s", (thread,)) > 0
        # the lease still resumes from its adopted history
        assert (await rt.release(a.lease_id, "ok")).status == "ok"
        assert [h["event"] for h in await rt.history(a.lease_id)][:2] == ["admit", "grant"]
    _run_with_runtime(body)


def test_prune_drops_only_old_released_threads():
    from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1

    async def body(rt, pool):
        old = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="p1"))
        await rt.release(old.lease_id, "ok")
        fresh = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="p2"))
        await rt.release(fresh.lease_id, "ok")
        live = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h", request_id="p3"))
        async with pool.connection() as conn:
            await conn.execute("UPDATE gpu_pool_leases SET updated_at = now() - interval '10 days' "
                               "WHERE lease_id = ANY(%s)", ([old.lease_id, live.lease_id],))
        pruned = await rt.store.prune_checkpoints(datetime.now(timezone.utc) - timedelta(days=7))
        assert pruned > 0

        async def threads(lease_id):
            return await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoints WHERE thread_id=%s",
                                (f"gpu_pool:{lease_id}",))
        assert await threads(old.lease_id) == 0           # released and old: gone
        assert await threads(fresh.lease_id) > 0          # released but recent: kept
        assert await threads(live.lease_id) > 0           # old but still granted: kept
        assert await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoint_writes WHERE thread_id=%s",
                            (f"gpu_pool:{old.lease_id}",)) == 0
        reply = await rt.control(GpuPoolControlV1(verb="backfill", backfill={
            "status": "released", "preview": False, "limit": 10}))
        assert old.lease_id in reply.detail["skipped_no_history"]
        assert reply.detail["replayed"] == 1              # the recent one still replays
    _run_with_runtime(body)
