"""The page:lately fold's SQL against REAL Postgres (issue #3463).

The unit tests fake the pool; this one runs the page's SQL (orion/memory/pages/lately.py, built on the
lived-memory serving SQL in orion/memory/lived_memory.py) on real tables in a throwaway schema:

- dead episode rows never reach the fold prompt: superseded (status form), pointed at by an active
  correction (pointer form), rejected, expired, ``invalid_at`` set, and older than the 14-day window
  (unless an open follow-up);
- only Hub chat where Juniper spoke, inside an episode the distiller has not distilled, from the
  window, reaches it;
- the change signal moves for a new chat turn, a newly invalidated row, and a row aging out of the
  window, and stays still otherwise.

Rows are dated against the database clock (the SQL's window is ``now()``-relative). Set
ORION_ADMISSION_TEST_DSN to a disposable database (CI does).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

DSN = os.environ.get("ORION_ADMISSION_TEST_DSN") or os.environ.get("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

# 2026-11-15 01:30 Denver: inside the fold window; never the Denver date of a row written below.
NOW = datetime(2026, 11, 15, 8, 30, tzinfo=timezone.utc)

DDL = """
CREATE TABLE episode_memory (memory_id uuid PRIMARY KEY, purpose text NOT NULL, statement text NOT NULL,
  occurred_at timestamptz, confirmation_state text NOT NULL, status text NOT NULL DEFAULT 'active',
  stakes text NOT NULL DEFAULT 'low', strength numeric NOT NULL DEFAULT 0.5, supersedes_memory_id uuid,
  expires_at timestamptz, invalid_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE chat_history_log (id varchar PRIMARY KEY, correlation_id varchar, prompt text, response text,
  created_at timestamp without time zone DEFAULT now());
CREATE TABLE memory_episode_shadow (episode_id text PRIMARY KEY, turns jsonb NOT NULL DEFAULT '[]'::jsonb,
  last_turn_at timestamptz NOT NULL);
CREATE TABLE episode_distill_run (episode_id text PRIMARY KEY, run_id text NOT NULL,
  finished_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE self_concept_history (entry_id varchar PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
  concept_id varchar NOT NULL, version integer NOT NULL, content text NOT NULL, evidence_refs json,
  produced_by varchar NOT NULL);
"""
OLD_CLAIM = uuid.uuid4()   # active, but an active correction points at it (pointer form)


async def _setup(schema: str):
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(conninfo=DSN, min_size=1, max_size=2, open=False,
                               kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                                       "options": f"-c search_path={schema}"})
    await pool.open()
    try:
        await _fill(pool, schema)
    except BaseException:
        await pool.close()
        raise
    return pool


async def _db_now(conn) -> datetime:
    return (await (await conn.execute("SELECT now() AS t")).fetchone())["t"]


async def _episode(conn, purpose, statement, *, at, state="auto", status="active", supersedes=None,
                   expires=None, invalid=None, memory_id=None, stakes="low"):
    await conn.execute(
        "INSERT INTO episode_memory (memory_id, purpose, statement, occurred_at, confirmation_state, status, stakes, "
        "supersedes_memory_id, expires_at, invalid_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (memory_id or uuid.uuid4(), purpose, statement, at, state, status, stakes, supersedes, expires, invalid, at, at))


async def _chat(conn, episode_id, prompt, *, at, response="I heard you.", is_command=False):
    corr = str(uuid.uuid4())
    naive = at.astimezone(timezone.utc).replace(tzinfo=None)
    await conn.execute("INSERT INTO chat_history_log (id, correlation_id, prompt, response, created_at) "
                       "VALUES (%s, %s, %s, %s, %s)", (str(uuid.uuid4()), corr, prompt, response, naive))
    if episode_id is not None:
        await conn.execute(
            "INSERT INTO memory_episode_shadow (episode_id, turns, last_turn_at) VALUES (%s, %s::jsonb, %s) "
            "ON CONFLICT (episode_id) DO UPDATE SET turns = memory_episode_shadow.turns || excluded.turns, "
            "last_turn_at = greatest(memory_episode_shadow.last_turn_at, excluded.last_turn_at)",
            (episode_id, json.dumps([{"correlation_id": corr, "at": at.isoformat(), "is_command": is_command}]), at))


async def _fill(pool, schema: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for statement in (part.strip() for part in DDL.split(";")):
            if statement:
                await conn.execute(statement)
        now = await _db_now(conn)
        recent, old = now - timedelta(days=3), now - timedelta(days=20)
        await _episode(conn, "happened", "Juniper and I watched the first snow together.", at=recent)
        await _episode(conn, "about_juniper", "Juniper is flying to see her sister on Friday.", at=recent,
                       stakes="high")
        await _episode(conn, "follow_up", "I owe Juniper an answer about the garden lights.", at=old)
        await _episode(conn, "happened", "DEAD-OLD: a happened row from three weeks ago.", at=old)
        await _episode(conn, "happened", "DEAD-REJECTED: we moved to Mars.", at=recent, state="rejected",
                       status="rejected")
        await _episode(conn, "happened", "DEAD-SUPERSEDED: the snow was rain.", at=recent, state="corrected",
                       status="superseded")
        await _episode(conn, "happened", "DEAD-POINTER: the cat is called Biscuit.", at=recent, memory_id=OLD_CLAIM)
        await _episode(conn, "happened", "Juniper corrected me: the cat is called Pebble.", at=recent,
                       supersedes=OLD_CLAIM)
        await _episode(conn, "happened", "DEAD-EXPIRED: a reminder that ran out.", at=recent,
                       expires=recent + timedelta(hours=1))
        await _episode(conn, "happened", "DEAD-INVALID: a fact that stopped being true.", at=recent,
                       invalid=recent + timedelta(hours=2))
        await _episode(conn, "follow_up", "DEAD-FU-REJECTED: ask about the boat.", at=old, state="rejected",
                       status="rejected")
        await _episode(conn, "follow_up", "DEAD-FU-ANCIENT: a follow-up from two months ago.",
                       at=now - timedelta(days=60))
        # Hub chat: an undistilled episode (alive), a distilled one, outreach with no Juniper words, a
        # turn older than the window, and a turn in no episode at all.
        await _chat(conn, "ep-open", "the snow is finally here and I love it", at=now - timedelta(hours=5))
        await _chat(conn, "ep-open", "", at=now - timedelta(hours=4), response="DEAD-OUTREACH: I wondered about snow.")
        await _chat(conn, "ep-done", "DEAD-DISTILLED: we already talked about this.", at=now - timedelta(days=2))
        await conn.execute("INSERT INTO episode_distill_run (episode_id, run_id, finished_at) VALUES ('ep-done', 'r1', %s)",
                           (now - timedelta(days=1),))
        await _chat(conn, "ep-ancient", "DEAD-ANCIENT: an old open episode.", at=old)
        await _chat(conn, None, "DEAD-NO-EPISODE: a turn from somewhere else.", at=now - timedelta(hours=3))
        await _chat(conn, "ep-open", "DEAD-COMMAND: /journal", at=now - timedelta(hours=2), is_command=True)
        await conn.execute(
            "INSERT INTO self_concept_history VALUES ('l1', %s, 'page:lately', 1, 'Old page [episode_memory:x].', "
            "'[\"episode_memory:x\"]', 'core_page_fold')", (now,))


async def _teardown(pool, schema):
    async with pool.connection() as conn:
        await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await pool.close()


def _folder(pool, prompts):
    from app.core_page_fold import CorePageFolder
    from orion.memory.pages.lately import FOLD_SPEC

    async def call_llm(prompt):
        prompts.append(prompt)
        return {"text": "Nothing [S1].", "model": "t"}

    async def publish(*a):
        return True

    s = SimpleNamespace(core_page_fold_hour_local=1, core_page_fold_end_hour_local=3,
                        core_page_fold_poll_sec=600.0, core_page_fold_max_attempts_per_day=3)
    return CorePageFolder(settings=s, pool=pool, call_llm=call_llm, publish=publish, now=lambda: NOW, spec=FOLD_SPEC)


async def _page_written_now(pool):
    async with pool.connection() as conn:
        await conn.execute("UPDATE self_concept_history SET created_at = now() WHERE concept_id = 'page:lately'")


def test_dead_rows_never_reach_the_lately_fold_and_the_change_signal_sees_them():
    async def main():
        schema = f"core_page_lately_{uuid.uuid4().hex[:10]}"
        pool = await _setup(schema)
        try:
            prompts: list[str] = []
            # Nothing changed since v1 (written after every row).
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"
            assert prompts == []
            # The manual trigger folds anyway: read the prompt the fold would send.
            result = await _folder(pool, prompts).tick_once(dry_run=True, force=True)
            assert result.decision == "rejected"   # the fake page is too thin
            prompt = prompts[-1]
            for alive in ("watched the first snow together", "flying to see her sister",
                          "an answer about the garden lights", "the cat is called Pebble",
                          "Juniper said: the snow is finally here and I love it"):
                assert alive in prompt, alive
            for dead in ("DEAD-OLD", "DEAD-REJECTED", "DEAD-SUPERSEDED", "DEAD-POINTER", "DEAD-EXPIRED",
                         "DEAD-INVALID", "DEAD-FU-REJECTED", "DEAD-FU-ANCIENT", "DEAD-COMMAND", "DEAD-OUTREACH", "DEAD-DISTILLED", "DEAD-ANCIENT",
                         "DEAD-NO-EPISODE"):
                assert dead not in prompt, dead
            assert result.sources == 5   # 4 memories + 1 chat turn
            assert "my memory (about_juniper, high stakes)" in prompt

            # A new Hub chat turn where Juniper spoke refolds.
            async with pool.connection() as conn:
                await _chat(conn, "ep-open", "the snowman fell over", at=await _db_now(conn) + timedelta(seconds=5))
            before = len(prompts)
            assert (await _folder(pool, prompts).tick_once()).decision == "rejected"   # it folded
            assert len(prompts) == before + 1 and "the snowman fell over" in prompts[-1]
            await _page_written_now(pool)
            async with pool.connection() as conn:
                await conn.execute("UPDATE chat_history_log SET created_at = created_at - interval '1 minute'")
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"

            # Invalidating a row (it bumps updated_at) refolds, and the row is gone from the prompt.
            async with pool.connection() as conn:
                await conn.execute("UPDATE episode_memory SET invalid_at = now(), updated_at = now() + interval '5 seconds' "
                                   "WHERE statement LIKE 'Juniper and I watched%%'")
            assert (await _folder(pool, prompts).tick_once()).decision == "rejected"
            assert "watched the first snow together" not in prompts[-1]
            await _page_written_now(pool)
            async with pool.connection() as conn:
                await conn.execute("UPDATE episode_memory SET updated_at = created_at")
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"

            # A row aging out of the window changes the input with no timestamp of its own: it refolds.
            async with pool.connection() as conn:
                # Everything else is older than the page row will be.
                await conn.execute("UPDATE chat_history_log SET created_at = created_at - interval '1 hour'")
                edge = await _db_now(conn) - timedelta(days=14, minutes=1)
                await _episode(conn, "happened", "AGING: right at the edge of the window.", at=edge)
                await conn.execute("UPDATE self_concept_history SET created_at = now() - interval '2 minutes' "
                                   "WHERE concept_id = 'page:lately'")
            assert (await _folder(pool, prompts).tick_once()).decision == "rejected"
            assert "AGING" not in prompts[-1]
            # Written after it aged out: quiet again.
            await _page_written_now(pool)
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"

            # The distiller finishing the open episode takes its turns out of the chat source, even when
            # it keeps no memory: that refolds, and the turns are gone.
            async with pool.connection() as conn:
                await conn.execute("INSERT INTO episode_distill_run (episode_id, run_id, finished_at) "
                                   "VALUES ('ep-open', 'r2', now() + interval '5 seconds')")
            assert (await _folder(pool, prompts).tick_once()).decision == "rejected"
            assert "the snow is finally here" not in prompts[-1] and "the snowman fell over" not in prompts[-1]
        finally:
            await _teardown(pool, schema)

    asyncio.run(main())
