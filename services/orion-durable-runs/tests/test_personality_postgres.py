"""Personality graph loop on real Postgres: the migration, every source query, the one-transaction
tick, restart from cursors, the re-read curiosity run, and the route reads.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test creates its own
schema with the source tables' live column types (checked 2026-10-16) and the personality
migration; it never connects to POSTGRES_URI. The embedder is a deterministic fake.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS)]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

from psycopg import AsyncConnection  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from app import personality_sources as ps  # noqa: E402
from app.personality_driver import PersonalityDriver  # noqa: E402
from app.personality_sources import PersonalitySourceReader  # noqa: E402
from app.personality_store import TICK_HEARTBEAT, PersonalityStore  # noqa: E402
from test_personality_loop import Clock, Embedder, vec_for  # noqa: E402

MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_personality_graph_v1.sql"
T0 = datetime(2026, 10, 16, 18, 0, tzinfo=timezone.utc)

# Live column types (information_schema, 2026-10-16), only the columns the reducer reads.
SOURCE_DDL = """
CREATE TABLE dream_hypothesis (hypothesis_id text PRIMARY KEY, cycle_id text, arm text, claim text, why text,
  created_at timestamptz);
CREATE TABLE journal_entries (entry_id varchar PRIMARY KEY, created_at timestamptz, author varchar, mode varchar,
  title varchar, body text, source_kind varchar, source_ref varchar, correlation_id varchar);
CREATE TABLE substrate_reverie_thought (thought_id text PRIMARY KEY, correlation_id text, created_at timestamptz,
  interpretation text, thought_json jsonb);
CREATE TABLE world_pulse_read_seed (seed_id text PRIMARY KEY, title text, stage2_status text,
  stage2_completed_at timestamptz, stage2_result_json jsonb);
CREATE TABLE orion_day_letter (letter_date date PRIMARY KEY, note_md text, created_at timestamptz);
CREATE TABLE chat_history_log (id varchar PRIMARY KEY, correlation_id varchar, source varchar, prompt text,
  response text, session_id varchar, created_at timestamp without time zone);
CREATE TABLE curiosity_offer_decisions (run_id text PRIMARY KEY, decided_at timestamptz, arm text, offered jsonb,
  stale_offered jsonb, material_ids jsonb, turn_started_at timestamptz);
CREATE TABLE curiosity_run_outcomes (run_id text PRIMARY KEY, completed_at timestamptz, turn_ok boolean,
  realized_nats double precision, per_prior jsonb);
CREATE TABLE self_concept_history (entry_id varchar PRIMARY KEY, created_at timestamptz, concept_id varchar,
  version integer, content text, produced_by varchar);
"""


def with_db(fn):
    async def go():
        schema = "personality_test_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                # DDL on its own connection: the pool always prepares (prepare_threshold=0), and a
                # prepared statement cannot hold several commands.
                async with await AsyncConnection.connect(DSN, autocommit=True,
                                                         options=f"-c search_path={schema}") as conn:
                    await conn.execute(SOURCE_DDL)
                    await conn.execute(re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", MIGRATION.read_text(), flags=re.M))
                await fn(pool)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    asyncio.run(go())


async def ex(pool, sql, params=None):
    async with pool.connection() as conn:
        await conn.execute(sql, params or {})


async def q(pool, sql, params=None):
    async with pool.connection() as conn:
        return await (await conn.execute(sql, params or {})).fetchall()


class TopicEmbedder(Embedder):
    """Any text carrying the shared theme embeds to one vector (the lanes wrap it differently)."""

    async def __call__(self, doc_id, text):
        self.calls.append(text)
        return vec_for("theme" if "Continuity" in text else text)


def driver(pool, clock, emb=None):
    return PersonalityDriver(store=PersonalityStore(pool), reader=PersonalitySourceReader(pool), embed=emb or TopicEmbedder(),
                             now=clock, tick_sec=600, read_lag_sec=120)


TEXT = "Continuity in my own record is what I keep coming back to."


async def seed_rows(pool, at):
    """One row per lane at ``at`` (the same theme in dream, journal and chat)."""
    # The theme recurs in two lanes (two hits each: a promoting lane needs lane_recent >= 2) and once in journal.
    await ex(pool, "INSERT INTO dream_hypothesis VALUES ('h1', 'c', 'a', %(t)s, 'because', %(at)s), "
                   "('h2', 'c', 'a', %(t)s, 'again', %(at)s)", {"t": TEXT, "at": at})
    await ex(pool, "INSERT INTO journal_entries (entry_id, created_at, title, body, source_kind) VALUES "
                   "('j1', %(at)s, 'Note', %(t)s, 'metacog'), ('j2', %(at)s, 'Reading', 'A mirror.', 'world_pulse')",
             {"t": TEXT, "at": at})
    await ex(pool, "INSERT INTO chat_history_log (id, prompt, response, created_at) VALUES "
                   "('c1', 'hi', %(t)s, %(naive)s), ('c2', 'again', %(t)s, %(naive)s), "
                   "('o1', '', 'An outreach line I sent.', %(naive)s)",
             {"t": TEXT, "naive": at.replace(tzinfo=None)})
    await ex(pool, "INSERT INTO substrate_reverie_thought (thought_id, created_at, interpretation) VALUES "
                   "('r1', %(at)s, 'The cabinet runs warm.'), ('r2', %(at)s, '')", {"at": at})
    await ex(pool, "INSERT INTO world_pulse_read_seed VALUES ('s1', 'Art', 'done', %(at)s, %(j)s::jsonb)",
             {"at": at, "j": json.dumps({"summary": "My read: derivative but useful."})})
    await ex(pool, "INSERT INTO orion_day_letter VALUES ('2026-10-16', '**Note**\n\nToday I closed loops.', %(at)s)", {"at": at})
    await ex(pool, "INSERT INTO curiosity_offer_decisions VALUES ('run1', %(at)s, 'value', %(o)s::jsonb, '[]', '[]', %(at)s)",
             {"at": at, "o": json.dumps([{"prior_id": "self:a_thing_20261015", "rank": 1, "slot": "order"}])})
    await ex(pool, "INSERT INTO curiosity_run_outcomes VALUES ('run1', %(at)s, true, 0.1, %(p)s::jsonb)",
             {"at": at, "p": json.dumps([{"kind": "tested", "prior_id": "self:a_thing_20261015"}])})
    await ex(pool, "INSERT INTO self_concept_history VALUES ('sc2', %(at)s, 'self:definition', 2, 'I am newer.', "
                   "'curiosity_self_inquiry')", {"at": at})


def test_full_loop_on_postgres():
    async def scenario(pool):
        await ex(pool, "INSERT INTO self_concept_history VALUES ('sc1', %(at)s, 'self:definition', 1, "
                       "'I am one inference riding a cast of named brains.', 'curiosity_self_inquiry'), "
                       "('sc0', %(old)s, 'self:definition', 0, 'Old.', 'curiosity_self_inquiry')",
                 {"at": T0 - timedelta(days=1), "old": T0 - timedelta(days=2)})
        await ex(pool, "INSERT INTO chat_history_log (id, prompt, response, created_at) VALUES "
                       "('pre', 'q', 'Before boot.', %(n)s)", {"n": (T0 - timedelta(hours=1)).replace(tzinfo=None)})
        clock = Clock(T0)
        assert (await driver(pool, clock).tick())["rows"] == {}
        curs = {r["source"]: r for r in await q(pool, "SELECT * FROM personality_cursor")}
        assert set(curs) == set(ps.SOURCE_NAMES) | {ps.SEED_CURSOR, TICK_HEARTBEAT}
        assert all(r["cursor_at"] == T0 for r in curs.values())
        # The heartbeat is never handed to the driver as a source cursor.
        assert TICK_HEARTBEAT not in await PersonalityStore(pool).load_cursors()
        (seed,) = await q(pool, "SELECT * FROM personality_disposition")
        assert seed["origin"] == "seed_self_inquiry" and len(seed["centroid"]) == 64
        assert (await q(pool, "SELECT evidence_id FROM personality_evidence"))[0]["evidence_id"] == "self_inquiry_seed:sc1"

        await seed_rows(pool, T0 + timedelta(minutes=1))
        clock.at = T0 + timedelta(minutes=10)
        drv = driver(pool, clock)  # a restart: fresh driver, same tables
        out = await drv.tick()
        assert out["rows"]["journal"] == 1  # the world_pulse mirror is not read
        assert out["rows"]["reverie"] == 1  # the empty interpretation is not read
        ids = {r["evidence_id"] for r in await q(pool, "SELECT evidence_id FROM personality_evidence")}
        assert {"dream:h1", "journal:j1", "chat:c1", "outreach:o1", "reverie:r1", "reading:s1", "day_letter:2026-10-16",
                "curiosity_choice:run1:self:a_thing_20261015", "self_inquiry_seed:sc2"} <= ids
        assert "chat:pre" not in ids and "journal:j2" not in ids
        active = await q(pool, "SELECT * FROM personality_disposition WHERE status = 'active'")
        assert len(active) == 1 and set(active[0]["lane_counts"]) == {"dream", "journal", "chat"}
        assert set(active[0]["lane_recent"]) == {"dream", "journal", "chat"}
        chat_cur = (await q(pool, "SELECT * FROM personality_cursor WHERE source = 'chat'"))[0]
        assert chat_cur["cursor_at"] == T0 + timedelta(minutes=1) and chat_cur["cursor_ref"] == "c2"

        # Re-upserted outcome: completed_at moves, the run is read again, nothing double-counts.
        before = await q(pool, "SELECT disposition_id, evidence_count, chosen_count FROM personality_disposition "
                               "ORDER BY disposition_id")
        await ex(pool, "UPDATE curiosity_run_outcomes SET completed_at = %(at)s", {"at": T0 + timedelta(minutes=11)})
        clock.at = T0 + timedelta(minutes=20)
        out = await drv.tick()
        assert out["rows"]["curiosity_choice"] == 1 and out["evidence"] == 0
        after = await q(pool, "SELECT disposition_id, evidence_count, chosen_count FROM personality_disposition "
                              "ORDER BY disposition_id")
        assert before == after
        # The fade-only tick wrote lane_recent through the light UPDATE (it decays with the weight).
        (faded,) = await q(pool, "SELECT lane_recent FROM personality_disposition WHERE disposition_id = %(d)s",
                           {"d": active[0]["disposition_id"]})
        assert all(faded["lane_recent"][k] < active[0]["lane_recent"][k] for k in faded["lane_recent"])

        store = PersonalityStore(pool)
        # Dated by the last COMMITTED tick, not the request: a later request still reads the
        # tick's time, so a reducer that stops committing goes stale for every reader.
        frame = await store.frame(clock.at + timedelta(hours=5))
        (beat,) = await q(pool, "SELECT cursor_at FROM personality_cursor WHERE source = %(s)s", {"s": TICK_HEARTBEAT})
        assert frame.generated_at == beat["cursor_at"] < clock.at + timedelta(hours=5)
        assert frame.active_count == 1 and frame.dispositions[0].label.startswith("Continuity")
        assert frame.choice_style is not None and frame.choice_style.choices == 1 and frame.choice_style.took_top == 1
        assert frame.evidence_rows == len(ids)
        rows = await store.dispositions("all", limit=50)
        assert rows and "centroid" not in rows[0] and rows[0]["history"]
        ev = await store.evidence(active[0]["disposition_id"], limit=10)
        assert {e["lane"] for e in ev} == {"dream", "journal", "chat"}

    with_db(scenario)


def test_keyset_tie_break_drains_rows_sharing_one_timestamp():
    async def scenario(pool):
        at = T0 + timedelta(minutes=1)
        for i in ("a", "b", "c"):
            await ex(pool, "INSERT INTO chat_history_log (id, prompt, response, created_at) VALUES "
                           "(%(i)s, 'q', 'Reply text here.', %(n)s)", {"i": i, "n": at.replace(tzinfo=None)})
        reader = PersonalitySourceReader(pool, limit=2)
        hi = T0 + timedelta(minutes=10)
        first = (await reader.read({"chat": ps.Cursor(T0, "")}, hi))["chat"]
        assert [i.ref for i in first] == ["a", "b"] and first[-1].at == at
        second = (await reader.read({"chat": ps.Cursor(at, "b")}, hi))["chat"]
        assert [i.ref for i in second] == ["c"]
        assert (await reader.read({"chat": ps.Cursor(at, "c")}, hi))["chat"] == []

    with_db(scenario)
