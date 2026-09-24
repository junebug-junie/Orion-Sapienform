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
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(conninfo=URI, min_size=1, max_size=6, open=False,
                               kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
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
        second = PostgresStore(pool)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(second.leader(), timeout=1.0)
        await store.close()
        await asyncio.wait_for(second.leader(), timeout=10.0)
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
