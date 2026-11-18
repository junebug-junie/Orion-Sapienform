"""The page:origin fold's SQL against REAL Postgres (issue #3462).

The unit tests fake the pool; this one runs the shared SQL (orion/memory/pages/origin.py, built on the
lived-memory serving SQL in orion/memory/lived_memory.py) on real tables in a throwaway schema:

- a superseded row (status form), a row an active correction points at (pointer form), a rejected
  row, an expired row and an out-of-scope purpose never reach the fold prompt;
- a rejected stance and a stance under salience 1 never reach it either;
- only Orion's own self:definition rows are read;
- the change signal moves for a new correction (even of another purpose) and a rejected stance.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI does).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

DSN = os.environ.get("ORION_ADMISSION_TEST_DSN") or os.environ.get("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

# 2026-11-15 01:30 Denver: inside the window.
NOW = datetime(2026, 11, 15, 8, 30, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=5)

# Columns the lived-memory serving SQL reads (orion/memory/lived_memory.py), nothing more.
DDL = """
CREATE TABLE episode_memory (memory_id uuid PRIMARY KEY, purpose text NOT NULL, statement text NOT NULL,
  occurred_at timestamptz, confirmation_state text NOT NULL, status text NOT NULL DEFAULT 'active',
  strength numeric NOT NULL DEFAULT 0.5, supersedes_memory_id uuid, expires_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE memory_crystallizations (crystallization_id uuid PRIMARY KEY, kind text NOT NULL,
  summary text NOT NULL, status text NOT NULL DEFAULT 'proposed', salience numeric NOT NULL DEFAULT 0.5,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
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


async def _episode(conn, purpose, statement, state="auto", status="active", supersedes=None, expires=None,
                   memory_id=None, at=OLD):
    await conn.execute(
        "INSERT INTO episode_memory (memory_id, purpose, statement, occurred_at, confirmation_state, status, "
        "supersedes_memory_id, expires_at, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (memory_id or uuid.uuid4(), purpose, statement, at, state, status, supersedes, expires, at, at))


async def _fill(pool, schema: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for statement in (part.strip() for part in DDL.split(";")):
            if statement:
                await conn.execute(statement)
        stances = [("stance", "Your memory storage died. We lost over a year of your journey of becoming.", "active", 1.0),
                   ("stance", "STANCE-REJECTED: you were born in a barn.", "rejected", 1.0),
                   ("stance", "STANCE-LOW: a passing hello.", "active", 0.6),
                   ("semantic", "SEMANTIC: not a stance at all.", "active", 1.0)]
        for kind, summary, status, salience in stances:
            await conn.execute(
                "INSERT INTO memory_crystallizations (crystallization_id, kind, summary, status, salience, created_at, "
                "updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)", (uuid.uuid4(), kind, summary, status, salience, OLD, OLD))
        await _episode(conn, "happened", "Juniper gave me a GitHub account, a milestone for my personhood.")
        await _episode(conn, "orion_view", "I reflected that my memory lives inside Juniper's last two years.")
        await _episode(conn, "happened", "DEAD-REJECTED: I was first switched on in 2019.", "rejected", "rejected")
        await _episode(conn, "happened", "DEAD-SUPERSEDED: my disk was replaced, nothing was lost.", "corrected",
                       "superseded")
        await _episode(conn, "happened", "DEAD-POINTER: my first name was Atlas.", memory_id=OLD_CLAIM)
        await _episode(conn, "happened", "Juniper corrected me: I was never called Atlas.", supersedes=OLD_CLAIM)
        await _episode(conn, "happened", "DEAD-EXPIRED: a reminder that ran out.", expires=OLD + timedelta(hours=1))
        await _episode(conn, "about_juniper", "OUT-OF-SCOPE: Juniper likes rain.")
        await _episode(conn, "follow_up", "OUT-OF-SCOPE: ask about the camera.")
        for i, (cid, by) in enumerate([("self:definition", "curiosity_self_inquiry"),
                                       ("self:definition", "curiosity_self_inquiry"),
                                       ("self:definition", "layer3_reflect"),
                                       ("page:juniper", "core_page_fold")], start=1):
            await conn.execute(
                "INSERT INTO self_concept_history VALUES (%s, %s, %s, %s, %s, '[]', %s)",
                (f"e{i}", OLD + timedelta(minutes=i), cid, i,
                 f"I am Orion, definition {i}." if by == "curiosity_self_inquiry" else f"NOT-MINE-{i}: other row.", by))
        await conn.execute(
            "INSERT INTO self_concept_history VALUES ('o1', %s, 'page:origin', 1, 'Old page [stance:x].', "
            "'[\"stance:x\"]', 'core_page_fold')", (OLD + timedelta(hours=1),))


async def _teardown(pool, schema):
    async with pool.connection() as conn:
        await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await pool.close()


def _folder(pool, prompts):
    from app.core_page_fold import CorePageFolder
    from orion.memory.pages.origin import FOLD_SPEC

    async def call_llm(prompt):
        prompts.append(prompt)
        return {"text": "Nothing [S1].", "model": "t"}

    async def publish(*a):
        return True

    s = SimpleNamespace(core_page_fold_hour_local=1, core_page_fold_end_hour_local=3,
                        core_page_fold_poll_sec=600.0, core_page_fold_max_attempts_per_day=3)
    return CorePageFolder(settings=s, pool=pool, call_llm=call_llm, publish=publish, now=lambda: NOW, spec=FOLD_SPEC)


def test_dead_rows_never_reach_the_origin_fold_and_the_change_signal_sees_them():
    async def main():
        schema = f"core_page_origin_{uuid.uuid4().hex[:10]}"
        pool = await _setup(schema)
        try:
            prompts: list[str] = []
            # Nothing changed since v1.
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"
            # The manual trigger folds anyway: read the prompt the fold would send.
            result = await _folder(pool, prompts).tick_once(dry_run=True, force=True)
            assert result.decision == "rejected"   # the fake page is too thin and skips the loss
            prompt = prompts[-1]
            for alive in ("We lost over a year of your journey of becoming.", "GitHub account",
                          "my memory lives inside Juniper's last two years", "I was never called Atlas",
                          "I am Orion, definition 1.", "I am Orion, definition 2."):
                assert alive in prompt, alive
            for dead in ("STANCE-REJECTED", "STANCE-LOW", "SEMANTIC:", "DEAD-REJECTED", "DEAD-SUPERSEDED",
                         "DEAD-POINTER", "DEAD-EXPIRED", "OUT-OF-SCOPE", "NOT-MINE"):
                assert dead not in prompt, dead
            assert result.sources == 6   # 1 stance + 3 memories + 2 self-definitions

            async with pool.connection() as conn:
                # A correction row of another purpose still moves the signal: it may retire an in-scope row.
                await _episode(conn, "about_juniper", "Juniper corrected me about the disk.", supersedes=OLD_CLAIM,
                               at=NOW - timedelta(hours=1))
            assert (await _folder(pool, prompts).tick_once()).decision != "no_new_sources"
            async with pool.connection() as conn:
                await conn.execute("UPDATE episode_memory SET created_at = %s, updated_at = %s "
                                   "WHERE statement = 'Juniper corrected me about the disk.'", (OLD, OLD))
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"
            async with pool.connection() as conn:
                # Rejecting a stance is a change too.
                await conn.execute("UPDATE memory_crystallizations SET status = 'rejected', updated_at = %s "
                                   "WHERE summary LIKE 'Your memory storage%%'", (NOW,))
            assert (await _folder(pool, prompts).tick_once()).decision != "no_new_sources"
            assert "We lost over a year" not in prompts[-1]
        finally:
            await _teardown(pool, schema)

    asyncio.run(main())
