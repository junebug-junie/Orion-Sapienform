"""Opt-in real PostgreSQL tests in a disposable local cluster; no production DSN.

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_reading_postgres.py -q
Models, DNS and bus publication are mocked. SQL, commits, locks and migrations are real.
"""
import asyncio
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from orion.schemas.reading import ReadingRequestedV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1, WorldPulseReadStage2ResultV1
from orion.world_pulse_read import queue
from orion.world_pulse_read.events import REQUESTED_CHANNEL
from orion.world_pulse_read.journal import journal_entry
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline
from orion.core.bus.bus_schemas import ServiceRef

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def local_pg(tmp_path_factory):
    if os.environ.get("RUN_READING_POSTGRES") != "1":
        pytest.skip("opt-in disposable PostgreSQL integration")
    candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin/initdb"), reverse=True)
    initdb = shutil.which("initdb") or (str(candidates[0]) if candidates else None)
    if not initdb:
        pytest.fail("initdb unavailable for requested integration")
    pg = Path(initdb).parent
    work = tmp_path_factory.mktemp("reading-pg")
    data, sock = work / "data", work / "socket"
    sock.mkdir()
    subprocess.run([str(pg / "initdb"), "-D", str(data), "-A", "trust", "-U", "reading_test", "--no-locale"], check=True, capture_output=True)
    subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-l", str(work / "postgres.log"), "-o", f"-k {sock} -h '' -p 16479", "-w", "start"], check=True, capture_output=True)
    try:
        yield {"host": str(sock), "port": 16479, "user": "reading_test", "database": "postgres"}
    finally:
        subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"], check=True, capture_output=True)


async def db(local_pg):
    conn = await asyncpg.connect(**local_pg)
    schema = "reading_" + uuid4().hex
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}"')
    await queue.ensure_seed_queue_schema(conn)
    await conn.execute("CREATE TABLE journal_entries (entry_id text PRIMARY KEY, source_ref text, body text)")
    return conn, schema


def request(url="https://example.org/article", **kwargs):
    return ReadingRequestedV1(url=url, requested_by="juniper", invocation_context="unified_chat", why_now="A new context", **kwargs)


def test_migration_is_additive_and_rerunnable(local_pg):
    async def run():
        conn = await asyncpg.connect(**local_pg)
        schema = "migration_" + uuid4().hex
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        for name in ("world_pulse_read_seed_queue", "world_pulse_read_stage2"):
            await conn.execute((ROOT / f"services/orion-sql-db/manual_migration_{name}_v1.sql").read_text())
        await conn.execute("INSERT INTO world_pulse_read_seed(seed_id,kind,run_id,url) VALUES('legacy','finding','r','https://example.org/old')")
        migration = (ROOT / "services/orion-sql-db/manual_migration_general_reading_v1.sql").read_text()
        await conn.execute(migration)
        await conn.execute(migration)
        legacy = await queue.claim_next_seed(conn)
        assert legacy.seed_id == "legacy"
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed") == 1
        await conn.close()
    asyncio.run(run())


def test_concurrent_submissions_alias_active_work_and_allow_later_reread(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        other = await asyncpg.connect(**local_pg)
        await other.execute(f'SET search_path TO "{schema}"')
        a, b = request(), request()
        receipts = await asyncio.gather(queue.enqueue_reading(conn, a), queue.enqueue_reading(other, b))
        assert all(r["status"] == "queued" for r in receipts)
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'") == 1
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE duplicate_of IS NOT NULL") == 1
        assert {r["request"]["request_id"] for r in receipts} == {str(a.request_id), str(b.request_id)}
        await queue.enqueue_reading(conn, a)
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed") == 2
        await conn.execute("UPDATE world_pulse_read_seed SET status='done', stage2_status='done' WHERE duplicate_of IS NULL")
        assert (await queue.enqueue_reading(conn, request()))["status"] == "queued"
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'") == 1
        await other.close()
        await conn.close()
    asyncio.run(run())


def test_event_sees_committed_request_from_another_connection(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        other = await asyncpg.connect(**local_pg)
        await other.execute(f'SET search_path TO "{schema}"')
        seen = []
        class Bus:
            async def publish(self, channel, envelope):
                assert channel == REQUESTED_CHANNEL
                seen.append(await other.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE request_id=$1", uuid_from(envelope.payload["request_id"])))
        assert (await queue.enqueue_reading(conn, request(), bus=Bus()))["status"] == "queued"
        assert seen == [1]
        async with conn.transaction():
            with pytest.raises(RuntimeError, match="own its commit"):
                await queue.enqueue_reading(conn, request("https://example.org/second"))
        await other.close()
        await conn.close()
    asyncio.run(run())


def uuid_from(value):
    from uuid import UUID
    return UUID(value)


def test_stage2_lineage_global_roundtrip_cap_and_durable_result(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        parent = request(parent_run_id="real-run", parent_trace_id="real-trace")
        await queue.enqueue_reading(conn, parent)
        seed = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(seed_ref=seed, what_i_learned="The source proposes a testable hypothesis.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        claim = await queue.claim_next_stage2_seed(conn)
        assert claim.seed.request == seed.request
        child = request("https://example.org/follow", parent_run_id=parent.parent_run_id, parent_trace_id=parent.parent_trace_id, root_request_id=parent.request_id, parent_request_id=parent.request_id)
        await queue.enqueue_reading(conn, child, max_round_trips=1)
        with pytest.raises(ValueError, match="round_trip_cap"):
            await queue.enqueue_reading(conn, request("https://example.org/third", root_request_id=parent.request_id, parent_request_id=child.request_id), max_round_trips=1)
        result = WorldPulseReadStage2ResultV1(summary="This remains a source-attributed candidate.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc), request=parent)
        await queue.mark_stage2_done(conn, seed.seed_id, stage2_trace_id=result.trace_id, result=result)
        status = await queue.reading_status(conn, parent.request_id)
        assert status["status"] == "landing_pending"
        assert status["summary"] == result.summary
        assert await queue.confirm_landings(conn) == []
        for entry in (journal_entry(handoff), journal_entry(handoff, result)):
            await conn.execute("INSERT INTO journal_entries VALUES($1,$2,$3)", entry.entry_id, entry.source_ref, entry.body)
        assert len(await queue.confirm_landings(conn)) == 1
        assert (await queue.reading_status(conn, parent.request_id))["status"] == "completed"
        await conn.close()
    asyncio.run(run())


def test_missing_journals_replay_saved_artifacts_without_model_or_wallet(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        req = request()
        await queue.enqueue_reading(conn, req)
        seed = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(seed_ref=seed, what_i_learned="Source-attributed learning.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        result = WorldPulseReadStage2ResultV1(summary="A candidate with a remaining question.", round_trips=2, trace_id=str(uuid4()), created_at=datetime.now(timezone.utc), request=req)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        await queue.mark_stage2_done(conn, seed.seed_id, stage2_trace_id=result.trace_id, result=result)
        class Bus:
            async def publish(self, channel, envelope):
                p = envelope.payload
                await conn.execute("INSERT INTO journal_entries VALUES($1,$2,$3) ON CONFLICT DO NOTHING", p["entry_id"], p["source_ref"], p["body"])
        pipe = object.__new__(WorldPulseReadStage2Pipeline)
        pipe._bus, pipe._source_ref = Bus(), ServiceRef(name="orion-hub")
        async def with_conn(fn):
            return await fn(conn)
        pipe._with_conn = with_conn
        assert (await queue.reading_status(conn, req.request_id))["status"] == "landing_pending"
        await pipe._repair_journal_landings()
        await pipe._repair_journal_landings()
        assert await conn.fetchval("SELECT count(*) FROM journal_entries") == 2
        stage2_body = await conn.fetchval("SELECT body FROM journal_entries WHERE source_ref LIKE 'world_pulse_read_stage2:%'")
        assert "round_trips=2" in stage2_body
        assert len(await queue.confirm_landings(conn)) == 1
        assert (await queue.reading_status(conn, req.request_id))["status"] == "completed"
        await conn.close()
    asyncio.run(run())
