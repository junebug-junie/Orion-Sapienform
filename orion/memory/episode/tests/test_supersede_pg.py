"""Supersede writes against a disposable Postgres (issue #3587). Synthetic rows only.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server); CI does, and fails on
a skip. Proves, with the real migration and the real reader SQL:

- a distilled correction supersedes its target at write time: all three marks (status, pointer,
  one ``superseded`` event) in the persist transaction;
- retrieval never serves the superseded row (``orion.memory.lived_memory.fetch_live_records``, the
  recall gate, both the full read and the id re-check), and the core-page fold input drops it;
- re-applying a handled pair is a no-op with no second event (idempotent pass);
- undo restores the row and pointer, and an undone pair is never re-applied;
- Juniper-confirmed rows and Orion's views are never superseded.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.memory.episode import supersede as s
from orion.memory.episode.store import persist_episode
from orion.memory.episode.tests.test_validate import TURNS, _mem
from orion.memory.episode.validate import validate_distillation
from orion.schemas.memory_episode import EpisodeDistillationV1

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
MIGRATION = Path(__file__).resolve().parents[4] / "services" / "orion-sql-db" / "manual_migration_episode_memory_v1.sql"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


async def _with_db(fn):
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    name = f"supersede_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=2, open=False,
                               kwargs={"autocommit": True, "row_factory": dict_row})
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute(MIGRATION.read_text())
        await fn(pool, dsn)
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


async def _old_row(pool, statement, *, purpose="about_juniper", state="auto"):
    mid = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO episode_memory (memory_id, episode_id, purpose, voice, channel, statement, stakes, "
            "confirmation_state, strength, last_reinforced_at, status, created_at, updated_at) "
            "VALUES (%s, 'ep-old', %s, 'juniper_said', 'chat', %s, 'low', %s, 0.5, %s, 'active', %s, %s)",
            (mid, purpose, statement, state, T0, T0, T0))
    return mid


def _correction():
    d = EpisodeDistillationV1.model_validate({"memories": [_mem()], "questions": []})
    result = validate_distillation(d, TURNS, episode_id="ep-new")
    assert result.memories, "fixture memory must validate"
    return result


async def _persist(pool, result, links):
    return await persist_episode(
        pool, episode_id="ep-new", run_id="memdistill-ep-new", result=result, model_route="memory_distill",
        model="m", prompt_version="v", usage={}, llm_latency_ms=1, hold_wait_ms=1, coverage=1.0,
        now=T0 + timedelta(days=3), supersede_links=links)


async def _one(pool, sql, params=()):
    async with pool.connection() as conn:
        return await (await conn.execute(sql, params)).fetchone()


async def _served_ids(dsn, ids=None):
    import asyncpg

    from orion.memory.lived_memory import KIND_EPISODE, fetch_live_records

    conn = await asyncpg.connect(dsn)
    try:
        by_kind = None if ids is None else {KIND_EPISODE: list(ids)}
        return {r.record_id for r in await fetch_live_records(conn, kinds=[KIND_EPISODE], ids_by_kind=by_kind)}
    finally:
        await conn.close()


async def _fold_ids(pool):
    from orion.memory.pages.juniper import EPISODE_MEMORY_SQL

    async with pool.connection() as conn:
        return {r["memory_id"] for r in await (await conn.execute(EPISODE_MEMORY_SQL)).fetchall()}


def test_write_time_supersede_sets_all_three_marks_and_readers_drop_the_old_row():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper told me her pet Biscuit is a cat.")
        assert old_id in await _served_ids(dsn) and old_id in await _fold_ids(pool)
        result = _correction()
        new_id = result.memories[0].memory_id
        link = s.Link(old_id=old_id, new_id=new_id, reason="B says the pet is a dog.", dense=0.7,
                      shared=("pet:biscuit",)).as_dict()
        counts = await _persist(pool, result, [link])
        assert counts["superseded"] == 1

        old = await _one(pool, "SELECT status, confirmation_state FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert (old["status"], old["confirmation_state"]) == ("superseded", "corrected")
        new = await _one(pool, "SELECT supersedes_memory_id::text AS p, status FROM episode_memory WHERE memory_id = %s",
                         (new_id,))
        assert (new["p"], new["status"]) == (old_id, "active")
        ev = await _one(pool, "SELECT event_id::text AS id, actor, reason, evidence FROM episode_memory_event "
                              "WHERE op = 'superseded' AND memory_id = %s", (old_id,))
        assert ev["id"] == s.superseded_event_id(old_id, new_id) and ev["actor"] == s.ACTOR_DISTILL
        assert ev["evidence"]["superseded_by"] == new_id and ev["evidence"]["prior_confirmation_state"] == "auto"

        # Retrieval never serves it, by full read or by the id re-check; the fold input drops it.
        assert old_id not in await _served_ids(dsn) and new_id in await _served_ids(dsn)
        assert await _served_ids(dsn, [old_id]) == set()
        assert old_id not in await _fold_ids(pool)
        # The pass's candidate corpus no longer holds it, so a second pass proposes nothing for it.
        async with pool.connection() as conn:
            assert old_id not in {r.memory_id for r in (await s.load_corpus(conn)).rows}

        # Idempotent: replaying the persist, or applying the pair again, writes nothing.
        assert (await _persist(pool, result, [link]))["superseded"] == 0
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link.from_dict(link), actor=s.ACTOR_PASS) is None
        n = await _one(pool, "SELECT count(*) AS n FROM episode_memory_event WHERE op = 'superseded'")
        assert n["n"] == 1

        # Undo restores the pair; the restored row is served again; the pair is never re-applied.
        async with pool.connection() as conn:
            async with conn.transaction():
                restored = await s.undo_link(conn, old_id=old_id, new_id=new_id, actor="operator", note="test")
        assert restored["restored_confirmation_state"] == "auto"
        old = await _one(pool, "SELECT status, confirmation_state FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert (old["status"], old["confirmation_state"]) == ("active", "auto")
        new = await _one(pool, "SELECT supersedes_memory_id FROM episode_memory WHERE memory_id = %s", (new_id,))
        assert new["supersedes_memory_id"] is None
        assert old_id in await _served_ids(dsn)
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link.from_dict(link), actor=s.ACTOR_PASS) is None
                assert await s.undo_link(conn, old_id=old_id, new_id=new_id, actor="operator") is None

    asyncio.run(_with_db(body))


def test_confirmed_rows_and_views_are_never_superseded_and_foreign_links_are_ignored():
    async def body(pool, dsn):
        confirmed = await _old_row(pool, "Juniper told me she lives in Millbrook.", state="confirmed")
        view = await _old_row(pool, "I think Juniper prefers mornings.", purpose="orion_view")
        other = await _old_row(pool, "Juniper said the office was loud.")
        result = _correction()
        new_id = result.memories[0].memory_id
        links = [s.Link(confirmed, new_id, "r").as_dict(), s.Link(view, new_id, "r").as_dict(),
                 s.Link(other, "not-this-episode", "r").as_dict()]
        assert (await _persist(pool, result, links))["superseded"] == 0
        for mid in (confirmed, view, other):
            row = await _one(pool, "SELECT status FROM episode_memory WHERE memory_id = %s", (mid,))
            assert row["status"] == "active"
        n = await _one(pool, "SELECT count(*) AS n FROM episode_memory_event WHERE op = 'superseded'")
        assert n["n"] == 0

    asyncio.run(_with_db(body))
