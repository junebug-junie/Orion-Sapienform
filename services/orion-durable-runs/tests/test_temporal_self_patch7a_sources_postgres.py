"""Integration patch 7a (#3429) on real Postgres: the three new source reads (intention_close, day_letter,
personality_choice) run their SQL against the live column shapes, and only rows available inside the window come
back, each through its adapter.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test makes its own schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

TZ = "America/Denver"
KINDS = ("intention_close", "day_letter", "personality_choice")

# Live column types (\d, 2026-11-25), only the columns the reads use plus NOT NULLs.
SOURCE_TABLES = """
CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, trace_id varchar NOT NULL, level varchar NOT NULL,
  actor varchar NOT NULL, verb varchar NOT NULL, object_ref text, correlation_id varchar,
  opened_at timestamptz, outcome_value double precision, outcome_note text, surprise_nats double precision,
  closed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE orion_day_letter (letter_date date PRIMARY KEY, run_id text NOT NULL, window_start timestamptz NOT NULL,
  window_end timestamptz NOT NULL, note_md text NOT NULL, carry_forward_md text NOT NULL, material jsonb NOT NULL,
  sources jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE personality_disposition (disposition_id text PRIMARY KEY, origin text NOT NULL, status text NOT NULL,
  label text NOT NULL, centroid real[] NOT NULL, first_seen_at timestamptz NOT NULL,
  last_reinforced_at timestamptz NOT NULL, name text);
CREATE TABLE personality_evidence (evidence_id text PRIMARY KEY, lane text NOT NULL, source_ref text NOT NULL,
  occurred_at timestamptz NOT NULL, excerpt text NOT NULL, disposition_id text, similarity double precision,
  weight_delta double precision NOT NULL DEFAULT 0, meta jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now())
"""


def with_db(fn):
    from uuid import uuid4

    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    async def go():
        schema = "ts_patch7a_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=2, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    for stmt in SOURCE_TABLES.split(";"):
                        if stmt.strip():
                            await conn.execute(stmt, prepare=False)
                await fn(pool)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    asyncio.run(go())


LO = datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)
HI = LO + timedelta(hours=12)


async def seed(pool) -> None:
    acts = [
        # carried: opened the day before the window (inside the slack), closed in it, with its carry run
        ("act_take1", "prime", "make", "intention:aaaaaaaaaaaaaaaa", LO - timedelta(hours=3), LO + timedelta(hours=1),
         1.0, "carried: the reply is in the chat thread t1 (608 chars)", 0.11),
        ("act_run1", "durable.intention", "make", "durable_run:intention-aaaaaaaaaaaaaaaa", LO - timedelta(hours=2),
         LO + timedelta(minutes=50), 1.0, "completed", 0.2),
        # refused: no carry run
        ("act_take2", "prime", "make", "intention:bbbbbbbbbbbbbbbb", LO + timedelta(hours=2),
         LO + timedelta(hours=2, seconds=12), 0.0, "Hub refused the order: no thread to answer in", 0.08),
        # still open: not an intention close
        ("act_take3", "prime", "make", "intention:cccccccccccccccc", LO + timedelta(hours=3), None, None, None, None),
        # closed after the window: read by the next window, not this one
        ("act_take4", "prime", "make", "intention:dddddddddddddddd", LO + timedelta(hours=4), HI + timedelta(minutes=1),
         1.0, "carried", 0.01),
        # another Prime make act: not an intention
        ("act_other", "prime", "make", "concern:x", LO + timedelta(hours=1), LO + timedelta(hours=1, seconds=3),
         1.0, "made", 0.01),
    ]
    async with pool.connection() as conn:
        for a in acts:
            await conn.execute(
                "INSERT INTO grammar_acts (act_id, trace_id, level, actor, verb, object_ref, opened_at, closed_at,"
                " outcome_value, outcome_note, surprise_nats) VALUES (%s, 'tr-' || %s, 'act', %s, %s, %s, %s, %s, %s, %s, %s)",
                (a[0], a[0], *a[1:]))
        for d, at, md in [(date(2026, 11, 24), LO + timedelta(hours=7), "- first thread\n- second thread"),
                          (date(2026, 11, 23), LO - timedelta(hours=20), "- yesterday's thread")]:
            await conn.execute(
                "INSERT INTO orion_day_letter (letter_date, run_id, window_start, window_end, note_md, carry_forward_md,"
                " material, created_at) VALUES (%s, %s, %s, %s, 'Dear day.', %s, '{}'::jsonb, %s)",
                (d, f"orion-day-{d}-1", at, at, md, at))
        await conn.execute(
            "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, first_seen_at,"
            " last_reinforced_at, name) VALUES ('disp-wait', 'emergent', 'active', 'Prime waits, 68 of 68', '{0.1}',"
            " %s, %s, 'patience')", (LO, LO))
        ev = [("ev-prime", "prime", "act_p1", LO + timedelta(hours=5), LO + timedelta(hours=5, minutes=16),
               "Prime waits instead of acting", "disp-wait",
               {"act_id": "act_p1", "choice_id": "prime:c1", "alternatives": 3, "declined_all": True}),
              ("ev-reading", "reading", "reading_stage2:x", LO + timedelta(hours=5), LO + timedelta(hours=5, minutes=1),
               "arXiv 2611.31228", None, {"choice_id": "reading:x", "alternatives": 4}),
              # chosen just before the window, folded inside it: read by fold time
              ("ev-prime-early", "prime", "act_p0", LO - timedelta(minutes=10), LO + timedelta(minutes=6),
               "a thread from my letter over a promise I made", None,
               {"act_id": "act_p0", "choice_id": "prime:c0", "alternatives": 2, "declined_all": False})]
        for e in ev:
            await conn.execute(
                "INSERT INTO personality_evidence (evidence_id, lane, source_ref, occurred_at, created_at, excerpt,"
                " disposition_id, similarity, meta) VALUES (%s, %s, %s, %s, %s, %s, %s, 0.9, %s::jsonb)",
                (*e[:7], json.dumps(e[7])))


def test_patch7a_reads_return_only_window_rows_through_their_adapters():
    from app.temporal_self_sources import SourceReader

    async def scenario(pool):
        await seed(pool)
        _, events = await SourceReader(pool, TZ).read(LO, HI, ticks=False, kinds=KINDS)
        by_id = {e.event_id: e for e in events}
        assert set(by_id) == {"intention_close:act_take1", "intention_close:act_take2", "day_letter:2026-11-24",
                              "personality_choice:ev-prime", "personality_choice:ev-prime-early"}
        carried = by_id["intention_close:act_take1"]
        assert carried.verdict == "produced" and carried.related_refs == [
            "act:act_take1", "intention:aaaaaaaaaaaaaaaa", "trace:tr-act_take1",
            "durable_run:intention-aaaaaaaaaaaaaaaa", "act:act_run1"]
        assert carried.payload["run_outcome"] == "completed" and carried.payload["surprise_nats"] == 0.11
        refused = by_id["intention_close:act_take2"]
        assert refused.verdict == "not_produced" and not any(r.startswith("durable_run:") for r in refused.related_refs)
        letter = by_id["day_letter:2026-11-24"]
        assert letter.label == "wrote the letter for 2026-11-24, carrying 2 threads; first: first thread"
        assert letter.related_refs == ["durable_run:orion-day-2026-11-24-1", "orion_day_letter:2026-11-24#thread0",
                                       "orion_day_letter:2026-11-24#thread1"]
        assert letter.payload["note_chars"] == len("Dear day.")
        waits = by_id["personality_choice:ev-prime"]
        assert waits.verdict == "declined_all"
        assert waits.label == 'Prime waits instead of acting (declined all 3 options); folded into my leaning "patience"'
        assert waits.payload["disposition_status"] == "active"

    with_db(scenario)
