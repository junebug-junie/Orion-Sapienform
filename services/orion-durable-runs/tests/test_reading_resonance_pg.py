"""Real Postgres checks for crash-safe requests, first-result wins and seed idempotency.

Set TEST_POSTGRES_DSN to a disposable database. Creates a unique schema per test.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pytest
import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
import asyncpg
from app.reading_resonance_store import persist_resonance
from orion.world_pulse_read.resonance_submit import bind_request
from orion.world_pulse_read.tests.test_resonance_runtime import receipt, packet
from orion.schemas.reading_resonance import ReadingResonanceRunBriefV1
from orion.substrate.resonance_seeds import UNSEEDED_RESONANCE_SQL, resonance_seeds


@pytest.fixture
def database():
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN needs a disposable database")
    schema = "resonance_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'SET search_path TO "{schema}"')
        conn.execute("CREATE TABLE world_pulse_read_seed(seed_id text PRIMARY KEY)")
        conn.execute((Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_reading_resonance.sql").read_text())
        conn.execute("CREATE TABLE substrate_endogenous_curiosity_candidates(generated_at timestamptz, candidates_json jsonb)")
    return dsn, schema


def test_request_and_terminal_receipt_survive_replayed_writes(database):
    async def run():
        dsn, schema = database
        conn = await asyncpg.connect(dsn, server_settings={"search_path": schema})
        try:
            await conn.execute("INSERT INTO world_pulse_read_seed VALUES('read-1')")
            brief = ReadingResonanceRunBriefV1(packet=packet())
            request = await bind_request(conn, brief)
            replay = await bind_request(conn, brief.model_copy(update={"max_tokens": 1800}))
            assert request == replay and replay.brief.max_tokens == 900
            async with AsyncConnectionPool(make_conninfo(dsn, options=f"-c search_path={schema}"),
                    kwargs={"autocommit": True, "row_factory": dict_row}, open=False) as pool:
                first = receipt()
                await persist_resonance(pool, first)
                other = first.model_copy(deep=True)
                other.raw = "A replay must not replace the answer"
                await persist_resonance(pool, other)
            stored = json.loads(await conn.fetchval("SELECT result_json FROM reading_resonance"))
            assert stored["raw"] == first.raw
            assert await conn.fetchval("SELECT count(*) FROM reading_resonance") == 1
        finally:
            await conn.close()
    asyncio.run(run())


def test_memory_only_receipts_do_not_starve_anchored_seeds_and_persisted_seed_is_not_reminted(database):
    async def run():
        dsn, schema = database
        conn = await asyncpg.connect(dsn, server_settings={"search_path": schema})
        try:
            now = datetime.now(timezone.utc)
            for i in range(14):
                r = receipt().model_copy(deep=True)
                r.seed_id = r.packet.seed_id = f"read-{i}"
                r.resonance_id = r.packet.packet_id = r.check.packet_id = f"packet-{i}"
                r.run_id = f"resonance-{i}"
                if i < 13:
                    r.judgment.citations = [r.judgment.citations[0]]
                await conn.execute("INSERT INTO world_pulse_read_seed VALUES($1)", r.seed_id)
                await conn.execute("INSERT INTO reading_resonance(seed_id,run_id,request_json,result_json,completed_at) "
                    "VALUES($1,$2,'{}',$3::jsonb,$4)", r.seed_id, r.run_id, r.model_dump_json(), now)
            sql = UNSEEDED_RESONANCE_SQL.replace(":lookback_hours", "$1").replace(":marker", "$2")
            marker = json.dumps([{"notes": ["source:reading_resonance"]}])
            rows = [dict(r) for r in await conn.fetch(sql, 24, marker)]
            for row in rows:
                row["result_json"] = json.loads(row["result_json"])
            assert len(rows) == 1 and rows[0]["result_json"]["resonance_id"] == "packet-13"
            seeds = resonance_seeds(rows)
            await conn.execute("INSERT INTO substrate_endogenous_curiosity_candidates VALUES($1,$2::jsonb)",
                now, json.dumps([s.model_dump(mode="json") for s in seeds]))
            assert await conn.fetch(sql, 24, marker) == []
        finally:
            await conn.close()
    asyncio.run(run())
