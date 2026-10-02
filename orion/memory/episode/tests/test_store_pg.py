"""Shadow episode-memory writes against a disposable Postgres.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server). Each test creates and
drops its own database. Proves the migration applies, a persist writes every table, a replayed
persist writes nothing twice, and rejected candidates are logged but never stored as memories.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

from orion.memory.episode.store import candidate_referent_keys, persist_episode
from orion.memory.episode.tests.test_validate import TURNS, _mem
from orion.memory.episode.validate import validate_distillation
from orion.schemas.memory_episode import EpisodeDistillationV1

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
MIGRATION = Path(__file__).resolve().parents[4] / "services" / "orion-sql-db" / "manual_migration_episode_memory_v1.sql"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")


async def _with_db(fn):
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    name = f"epmem_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=2, open=False,
                               kwargs={"autocommit": True, "row_factory": dict_row})
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute(MIGRATION.read_text())
        await fn(pool)
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _result():
    d = EpisodeDistillationV1.model_validate({
        "memories": [
            _mem(),
            _mem(voice="juniper_said", statement="Juniper hopes the trip to Austin goes smoothly for her.",
                 evidence=[{"turn": "t2", "field": "response", "quote": "I hope the trip goes smoothly"}]),
            _mem(statement="Juniper is flying to Denver tomorrow morning.",
                 evidence=[{"turn": "t2", "field": "prompt", "quote": "flying to Denver"}]),
        ],
        "questions": [{"text": "Is being away from home hard for Juniper?",
                       "evidence": [{"turn": "t1", "field": "prompt", "quote": "work travel"}]}],
    })
    return validate_distillation(d, TURNS, episode_id="ep-pg")


async def _counts(pool):
    out = {}
    async with pool.connection() as conn:
        for table in ("episode_memory", "episode_memory_evidence", "episode_memory_referent", "episode_memory_event",
                      "memory_tension_shadow", "episode_distill_run"):
            out[table] = (await (await conn.execute(f"SELECT count(*) AS n FROM {table}")).fetchone())["n"]
    return out


def test_persist_is_complete_and_replay_safe():
    async def body(pool):
        result = _result()
        kw = dict(episode_id="ep-pg", run_id="memdistill-ep-pg", result=result, model_route="memory_distill",
                  model="qwen-27b", prompt_version="memory_episode_distill.v1",
                  usage={"prompt_tokens": 900, "completion_tokens": 120}, llm_latency_ms=4200, hold_wait_ms=1000,
                  coverage=0.667)
        counts = await persist_episode(pool, **kw)
        assert counts == {"memories": 2, "rejections": 1, "questions": 1, "downgrades": 1}
        first = await _counts(pool)
        assert first["episode_memory"] == 2 and first["memory_tension_shadow"] == 1 and first["episode_distill_run"] == 1
        await persist_episode(pool, **kw)                       # crash-and-resume replay
        assert await _counts(pool) == first
        async with pool.connection() as conn:
            voices = [r["voice"] for r in await (await conn.execute(
                "SELECT voice FROM episode_memory ORDER BY statement")).fetchall()]
            ops = sorted(r["op"] for r in await (await conn.execute("SELECT op FROM episode_memory_event")).fetchall())
            rejected = await (await conn.execute(
                "SELECT memory_id, reason FROM episode_memory_event WHERE op = 'rejected_invalid'")).fetchall()
        assert voices == ["juniper_said", "orion_thought"]      # the downgrade was stored, not the claim
        assert ops.count("created") == 2 and "downgraded_voice" in ops
        assert [(r["memory_id"], r["reason"]) for r in rejected] == [(None, "no_verified_quote")]
        assert await candidate_referent_keys(pool) == ["event:austin-ai-ml-offsite-2026-09"]

    asyncio.run(_with_db(body))


def test_candidate_keys_fail_open_without_the_tables():
    async def body(pool):
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE episode_memory_referent")
        assert await candidate_referent_keys(pool) == []

    asyncio.run(_with_db(body))


def test_load_turns_sql_reads_full_text_in_time_order():
    from orion.memory.episode.distill import LOAD_TURNS_SQL, turns_from_rows

    async def body(pool):
        long_prompt = "y" * 500 + " end"
        async with pool.connection() as conn:
            await conn.execute("CREATE TABLE chat_history_log (correlation_id varchar, prompt text, response text, "
                               "created_at timestamp)")
            await conn.execute("INSERT INTO chat_history_log VALUES ('b', 'second', 'Workflow: X', '2026-09-28 09:44'),"
                               " ('a', %s, 'ok', '2026-09-28 09:06'), ('z', 'other', 'no', '2026-09-28 09:00')",
                               (long_prompt,))
            rows = await (await conn.execute(LOAD_TURNS_SQL, (["a", "b"],))).fetchall()
        turns = turns_from_rows([dict(r) for r in rows])
        assert [(t.correlation_id, t.is_command) for t in turns] == [("a", False), ("b", True)]
        assert turns[0].prompt == long_prompt

    asyncio.run(_with_db(body))
