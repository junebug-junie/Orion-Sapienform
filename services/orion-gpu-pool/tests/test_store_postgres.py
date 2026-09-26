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
SQL_DB = Path(__file__).resolve().parents[3] / "services/orion-sql-db"
MIGRATION = SQL_DB / "manual_migration_gpu_pool_v1.sql"
MIGRATION_V2 = SQL_DB / "manual_migration_gpu_pool_v2_holds.sql"   # stage 4.3


async def _apply(conn, path: Path) -> None:
    """As `psql -f` runs it: statement by statement, autocommit (CREATE INDEX CONCURRENTLY needs it)."""
    sql = "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("--"))
    for statement in (s.strip() for s in sql.split(";")):
        if statement:
            await conn.execute(statement)


async def _pool():
    """Production's layout, then the production connection setup (app.store.pool_kwargs +
    ensure_checkpoint_schema): the projection migration applied on a plain connection (public),
    and durable-runs' LangGraph tables already in public at the current migration version --
    so the pool's own saver.setup() must still create its tables in the gpu_pool schema."""
    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from app.store import ensure_checkpoint_schema, pool_kwargs

    async with await psycopg.AsyncConnection.connect(URI, autocommit=True, row_factory=dict_row) as conn:
        await conn.execute("DROP SCHEMA IF EXISTS gpu_pool CASCADE")
        await conn.execute("DROP TABLE IF EXISTS public.checkpoints, public.checkpoint_blobs, "
                           "public.checkpoint_writes, public.checkpoint_migrations, gpu_pool_leases, gpu_pool_cards")
        await _apply(conn, MIGRATION)
        await _apply(conn, MIGRATION_V2)
        await conn.execute("RESET lock_timeout")
        await AsyncPostgresSaver(conn).setup()          # durable-runs' tables, in public
    await ensure_checkpoint_schema(URI)
    pool = AsyncConnectionPool(conninfo=URI, min_size=1, max_size=6, open=False, kwargs=pool_kwargs())
    await pool.open()
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
        assert await _count(pool, "SELECT count(*) AS n FROM public.checkpoints") == 0   # durable-runs' table untouched
        assert await _count(pool, "SELECT count(*) AS n FROM public.gpu_pool_leases") > 0   # projection stays in public
        assert await _count(pool, "SELECT max(v) AS n FROM gpu_pool.checkpoint_migrations") == \
            await _count(pool, "SELECT max(v) AS n FROM public.checkpoint_migrations")
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
                await conn.execute(f"INSERT INTO public.{table} SELECT * FROM gpu_pool.{table}")
                await conn.execute(f"DELETE FROM gpu_pool.{table}")
            await conn.execute("INSERT INTO public.checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
                               "metadata) VALUES ('durable-run-1', '', 'c1', '{}', '{}')")
        moved = await rt.store.adopt_public_checkpoints()
        assert moved > 0
        for _ in range(6):   # no pooled connection keeps the adopt step's lock_timeout
            async with pool.connection() as conn:
                assert (await (await conn.execute("SHOW lock_timeout")).fetchone())["lock_timeout"] == "0"
        assert await rt.store.adopt_public_checkpoints() == 0          # idempotent
        assert await _count(pool, "SELECT count(*) AS n FROM public.checkpoints WHERE thread_id LIKE 'gpu_pool:%%'") == 0
        assert await _count(pool, "SELECT count(*) AS n FROM public.checkpoints WHERE thread_id='durable-run-1'") == 1
        assert await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoints WHERE thread_id=%s", (thread,)) > 0
        # the lease still resumes from its adopted history
        assert (await rt.release(a.lease_id, "ok")).status == "ok"
        assert [h["event"] for h in await rt.history(a.lease_id)][:2] == ["admit", "grant"]
    _run_with_runtime(body)


def test_prune_forgets_only_old_ended_leases_in_bounded_batches():
    from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1

    async def body(rt, pool):
        async def lease(rid):
            return (await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="diffusion", holder="h",
                                                       request_id=rid))).lease_id
        old = [await lease(f"o{i}") for i in range(3)]
        for lid in old:
            await rt.release(lid, "ok")
        fresh = await lease("f")
        await rt.release(fresh, "ok")
        gone_unavailable, dead, live = await lease("u"), await lease("d"), await lease("l")
        async with pool.connection() as conn:
            await conn.execute("UPDATE gpu_pool_leases SET status='unavailable' WHERE lease_id=%s", (gone_unavailable,))
            await conn.execute("UPDATE gpu_pool_leases SET status='dead_letter' WHERE lease_id=%s", (dead,))
            await conn.execute("UPDATE gpu_pool_leases SET updated_at = now() - interval '10 days' "
                               "WHERE lease_id = ANY(%s)", ([*old, gone_unavailable, dead, live],))
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        assert await rt.store.prune_checkpoints(cutoff, batch=2) == 4      # 3 released + 1 unavailable, 2 per batch
        assert await rt.store.prune_checkpoints(cutoff) == 0

        async def thread_rows(lid):
            return await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoints WHERE thread_id=%s",
                                (f"gpu_pool:{lid}",))

        async def lease_rows(lid):
            return await _count(pool, "SELECT count(*) AS n FROM gpu_pool_leases WHERE lease_id=%s", (lid,))
        for lid in [*old, gone_unavailable]:                  # ended and old: thread and row gone
            assert await thread_rows(lid) == 0 and await lease_rows(lid) == 0
        assert await _count(pool, "SELECT count(*) AS n FROM gpu_pool.checkpoint_writes WHERE thread_id = ANY(%s)",
                            ([f"gpu_pool:{lid}" for lid in old],)) == 0
        for lid in (fresh, dead, live):                       # recent / dead letter / still granted: kept
            assert await thread_rows(lid) > 0 and await lease_rows(lid) == 1
        reply = await rt.control(GpuPoolControlV1(verb="backfill", backfill={
            "status": "released", "preview": False, "limit": 10}))
        assert reply.detail["replayed"] == 1 and reply.detail["skipped_no_history"] == []
    _run_with_runtime(body)


def test_v2_migration_is_additive_idempotent_and_required_at_boot():
    """Applied to a live-shaped v1 table with rows: nothing lost, re-runnable, and a pool without it
    refuses to boot (check_schema) instead of failing on its first hold."""
    async def go():
        import psycopg
        from psycopg.rows import dict_row

        from app.store import PostgresStore

        async with await psycopg.AsyncConnection.connect(URI, autocommit=True, row_factory=dict_row) as conn:
            await conn.execute("DROP TABLE IF EXISTS gpu_pool_leases, gpu_pool_cards")
            await _apply(conn, MIGRATION)
            await conn.execute("INSERT INTO gpu_pool_leases (lease_id, request_id, holder, work_class, priority, kind, "
                               "status, created_at) VALUES ('old', 'r-old', 'h', 'fast', 'system', 'request', 'granted', now())")
            await conn.execute("INSERT INTO gpu_pool_cards (card, swapped_in) VALUES ('gpu2', '{agent-gpu2}')")
        pool = await _v1_only_pool()
        try:
            with pytest.raises(Exception):
                await PostgresStore(pool).check_schema()
        finally:
            await pool.close()
        async with await psycopg.AsyncConnection.connect(URI, autocommit=True, row_factory=dict_row) as conn:
            await _apply(conn, MIGRATION_V2)
            await _apply(conn, MIGRATION_V2)                      # idempotent
            row = await (await conn.execute("SELECT * FROM gpu_pool_leases WHERE lease_id='old'")).fetchone()
            assert row["status"] == "granted" and row["hold_lease_id"] is None
            card = await (await conn.execute("SELECT * FROM gpu_pool_cards WHERE card='gpu2'")).fetchone()
            assert card["swapped_in"] == ["agent-gpu2"] and card["swap_generation"] == 0 and card["swap_action"] is None
            idx = await (await conn.execute("SELECT indisvalid FROM pg_index WHERE indexrelid = "
                                            "'gpu_pool_leases_hold_idx'::regclass")).fetchone()
            assert idx["indisvalid"]
        pool = await _v1_only_pool()
        try:
            await PostgresStore(pool).check_schema()
        finally:
            await pool.close()
    asyncio.run(go())


async def _v1_only_pool():
    from psycopg_pool import AsyncConnectionPool

    from app.store import pool_kwargs
    pool = AsyncConnectionPool(conninfo=URI, min_size=1, max_size=2, open=False, kwargs=pool_kwargs())
    await pool.open()
    return pool


def test_holds_children_and_a_mid_load_card_survive_a_restart_on_postgres():
    """Acceptance check 6 on the real tables: the hold resumes by lease_id with its child's link,
    and a restart mid-load reconciles with `status` instead of a second transition."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from app.runtime import PoolRuntime
    from app.store import PostgresStore
    from orion.gpu_pool.config import load_pool_config
    from orion.gpu_pool.discovery import Probe, load_profiles
    from orion.gpu_pool.lease_graph import build_lease_graph
    from orion.schemas.gpu_pool import GpuActuateV1, GpuLeaseRequestV1

    class Bus:
        def __init__(self):
            self.published = []

        async def publish(self, channel, env):
            self.published.append((channel, env))

        def record_hop_success(self, *a):
            pass

        def record_hop_timeout(self, *a):
            pass

    async def go():
        cfg = load_pool_config()
        pool = await _pool()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()

            async def prober(role, url, kind, health):
                return Probe(kind == "service")

            def runtime(bus):
                rt = PoolRuntime(cfg=cfg, profiles=load_profiles(), store=PostgresStore(pool),
                                 graph=build_lease_graph(lambda: cfg, saver), prober=prober, bus=bus,
                                 actuate_roles=["agent-gpu2"])
                return rt

            rt = runtime(Bus())
            await rt.start()
            await rt.tick()
            # world is a 2-slot service: a hold there stands in for a run (no llm probe needed)
            h = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="world", holder="durable-runs:r1",
                                                   request_id="r1:1", kind="hold", retryable=True))
            assert h.status == "granted"
            c = await rt.attach(GpuLeaseRequestV1(verb="attach", work_class="world", holder="gw", request_id="c1",
                                                  hold_lease_id=h.lease_id, hold_generation=h.grant.generation))
            assert c.status == "granted"
            assert (await rt.store.lease(c.lease_id))["hold_lease_id"] == h.lease_id
            # an in-flight load, as the engine persists it
            from orion.gpu_pool.scheduler import SwapLoad
            await rt._begin_actuation(SwapLoad("agent-gpu2", "demand"))
            [(_, sent)] = [(ch, e) for ch, e in rt.bus.published if ch == "orion:gpu_pool:actuate:request"]
            load = GpuActuateV1.model_validate(sent.payload)

            bus2 = Bus()
            rt2 = runtime(bus2)
            await rt2.start()
            assert rt2.cards["gpu2"].swap_state == "loading" and rt2.cards["gpu2"].swap_generation == 1
            rt2._ctx_seen["agent-gpu2"] = 131072              # seen_ctx jsonb round trip
            await rt2._save_seen_ctx()
            row = {c["card"]: c for c in await rt2.store.cards()}["gpu2"]
            assert row["seen_ctx"]["agent-gpu2"] == 131072 and row["swap_state"] == "loading"
            msgs = [GpuActuateV1.model_validate(e.payload) for ch, e in bus2.published
                    if ch == "orion:gpu_pool:actuate:request"]
            assert [(m.action, m.generation) for m in msgs] == [("status", load.generation)]
            st = await rt2.status(h.lease_id)
            assert st.status == "granted" and st.grant.generation == h.grant.generation
            snap = await rt2.snapshot()
            assert {r.lease_id: r.hold_lease_id for r in snap.leases}[c.lease_id] == h.lease_id
            gpu2 = next(x for x in snap.cards if x.card == "gpu2")
            assert gpu2.swap_state == "loading" and gpu2.actuation["action_id"] == load.action_id
        finally:
            await pool.close()
    asyncio.run(go())
