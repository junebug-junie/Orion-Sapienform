"""The page:juniper fold's SQL against REAL Postgres (core memory pages Phase 1a).

The unit tests fake the pool; this one runs the shared SQL (orion/memory/pages/juniper.py) on real
tables in a throwaway schema, so the change signal is proven with the same filter the read uses:
an out-of-scope row never triggers a fold, an in-scope new row or a rejected row does, a removed
fact does, and removed / rejected / superseded rows never reach the prompt.

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

DDL = """
CREATE TABLE person (person_id text PRIMARY KEY, display_name text NOT NULL, role text NOT NULL DEFAULT 'other',
  status text NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(), merged_into text);
CREATE TABLE person_fact (fact_id text PRIMARY KEY, person_id text NOT NULL, category text NOT NULL,
  value_text text NOT NULL, voice text NOT NULL, source_ref text, created_at timestamptz NOT NULL DEFAULT now(),
  removed_at timestamptz);
CREATE TABLE episode_memory (memory_id uuid PRIMARY KEY, purpose text NOT NULL, statement text NOT NULL,
  occurred_at timestamptz, confirmation_state text NOT NULL, status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE self_concept_history (entry_id varchar PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
  concept_id varchar NOT NULL, version integer NOT NULL, content text NOT NULL, evidence_refs json,
  produced_by varchar NOT NULL);
"""
GOOD = "Juniper uses she/her [S1]. She lives in Millbrook [S2]. Biscuit is a dog [S3]."


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


async def _fill(pool, schema: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for statement in (part.strip() for part in DDL.split(";")):
            if statement:
                await conn.execute(statement)
        await conn.execute("INSERT INTO person (person_id, display_name, updated_at) VALUES ('p1', 'Juniper', %s)", (OLD,))
        await conn.execute("INSERT INTO person_fact VALUES ('f1', 'p1', 'pronouns', 'she/her', 'authored_profile', "
                           "NULL, %s, NULL), ('f-gone', 'p1', 'home', 'Boston', 'ask_answer', NULL, %s, %s)",
                           (OLD, OLD, OLD))
        rows = [("about_juniper", "Juniper corrected me that she lives in Millbrook, not Boston.", "confirmed", "active"),
                ("happened", "Juniper told me that Biscuit is a dog, not a cat.", "auto", "active"),
                ("happened", "Juniper said Biscuit is a cat.", "rejected", "rejected"),
                ("happened", "Juniper said she lives in Boston.", "corrected", "superseded"),
                ("happened", "The office was loud all evening.", "auto", "active")]
        for purpose, statement, state, status in rows:
            await conn.execute(
                "INSERT INTO episode_memory (memory_id, purpose, statement, occurred_at, confirmation_state, status, "
                "created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (uuid.uuid4(), purpose, statement, OLD, state, status, OLD, OLD))
        await conn.execute(
            "INSERT INTO self_concept_history VALUES ('e1', %s, 'page:juniper', 1, 'Old page [person_fact:f1].', "
            "'[\"person_fact:f1\"]', 'core_page_fold')", (OLD + timedelta(hours=1),))


async def _teardown(pool, schema):
    async with pool.connection() as conn:
        await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await pool.close()


def _folder(pool, prompts):
    from app.core_page_fold import CorePageFolder

    async def call_llm(prompt):
        prompts.append(prompt)
        return {"text": GOOD, "model": "t"}

    async def publish(*a):
        return True

    s = SimpleNamespace(core_page_fold_hour_local=1, core_page_fold_end_hour_local=3,
                        core_page_fold_poll_sec=600.0, core_page_fold_max_attempts_per_day=3)
    return CorePageFolder(settings=s, pool=pool, call_llm=call_llm, publish=publish, now=lambda: NOW)


def test_change_signal_uses_the_read_filter_and_the_prompt_drops_dead_rows():
    async def main():
        schema = f"core_page_{uuid.uuid4().hex[:10]}"
        pool = await _setup(schema)
        try:
            prompts: list[str] = []
            # Nothing changed since v1.
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"
            async with pool.connection() as conn:
                # An out-of-scope row (does not name her) changes nothing the page reads.
                await conn.execute(
                    "INSERT INTO episode_memory (memory_id, purpose, statement, confirmation_state, created_at, "
                    "updated_at) VALUES (%s, 'happened', 'The cabinet ran hot.', 'auto', %s, %s)",
                    (uuid.uuid4(), NOW, NOW))
            assert (await _folder(pool, prompts).tick_once()).decision == "no_new_sources"
            async with pool.connection() as conn:
                # Rejecting an in-scope row is a change.
                await conn.execute("UPDATE episode_memory SET status = 'rejected', confirmation_state = 'rejected', "
                                   "updated_at = %s WHERE statement LIKE 'Juniper told me that Biscuit%%'", (NOW,))
            result = await _folder(pool, prompts).tick_once()
            assert result.decision == "rejected"   # GOOD cites S3, which no longer exists: unresolvable
            assert "unresolvable_cite: S3" in result.errors
            prompt = prompts[-1]
            assert "Millbrook, not Boston" in prompt and "she/her" in prompt
            for dead in ("Biscuit is a cat", "she lives in Boston.", "Boston\n", "Biscuit is a dog",
                         "office was loud", "cabinet ran hot"):
                assert dead not in prompt, dead
            async with pool.connection() as conn:
                # A removed fact is a change too (fresh process: no attempts spent).
                await conn.execute("UPDATE episode_memory SET status = 'active', confirmation_state = 'auto', "
                                   "updated_at = %s WHERE statement LIKE 'Juniper told me that Biscuit%%'",
                                   (OLD,))
                await conn.execute("UPDATE self_concept_history SET created_at = %s", (NOW - timedelta(hours=9),))
                await conn.execute("UPDATE person_fact SET removed_at = %s WHERE fact_id = 'f1'", (NOW,))
            again = await _folder(pool, prompts).tick_once()
            assert again.decision != "no_new_sources"
        finally:
            await _teardown(pool, schema)

    asyncio.run(main())
