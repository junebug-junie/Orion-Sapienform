"""Game company patch 3 (spec 2026-11-16 rev 4, section 5 "Who typed"), on real Postgres.

A guest's message from the game page is written to chat_history_log on Juniper's own session, with a
prompt, and with source 'retro_guest'. The three readers that mean "Juniper spoke" must not match it:
the chronicle's chat_turn read (speaker rule in orion/temporal_self/sources.py), the attention count of
her turns (orion/attention/world_first.py ``_CHAT_TURN_WHERE``) and the day letter's chat gather
(orion/orion_day/gather.py ``JUNIPER_CHAT_SQL``). Each SQL runs here exactly as written: the asyncpg
``$n`` and SQLAlchemy ``:since`` forms through PREPARE / EXECUTE.

A flip mid-sitting: rows sent before the page was told "Juniper" keep 'retro_guest', later ones are
'hub_orion'. Nothing is rewritten, so only the later rows are hers.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one).
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from app.temporal_self_sources import QUERIES  # noqa: E402
from orion.attention.world_first import CHAT_TURN_TIMES_SQL  # noqa: E402
from orion.orion_day.gather import JUNIPER_CHAT_SOURCES, JUNIPER_CHAT_SQL  # noqa: E402
from orion.temporal_self.sources import ADAPTERS  # noqa: E402

T0 = datetime(2026, 11, 18, 18, 0, tzinfo=timezone.utc)
SESSION = "sess-juniper"

TABLE = """
CREATE TABLE chat_history_log (
  id text PRIMARY KEY, correlation_id text, session_id text, source text, created_at timestamp without time zone,
  prompt text, response text, client_meta jsonb)
"""


def _rows() -> list[dict]:
    """20 minutes of a guest typing, then the flip: Juniper's own turns. A line Orion said in between."""
    rows = [{"id": f"guest-{i}", "source": "retro_guest", "prompt": f"what is that room? {i}", "meta": {}}
            for i in range(3)]
    rows.append({"id": "line-1", "source": "hub_orion", "prompt": "", "meta": {
        "unsolicited": True, "source": "retro_commentary"}})
    rows += [{"id": f"juniper-{i}", "source": "hub_orion", "prompt": f"it is me now {i}", "meta": {"origin": "x"}}
             for i in range(3)]
    for n, r in enumerate(rows):
        r["created_at"] = (T0 + timedelta(minutes=7 * n)).replace(tzinfo=None)
    return rows


def _scenario(check):
    schema = "retro_guest_test_" + uuid4().hex[:10]
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            conn.execute(f'SET search_path = "{schema}"')
            conn.execute(TABLE)
            for r in _rows():
                conn.execute(
                    "INSERT INTO chat_history_log VALUES (%s, %s, %s, %s, %s, %s, 'reply', %s)",
                    (r["id"], "c-" + r["id"], SESSION, r["source"], r["created_at"], r["prompt"], Jsonb(r["meta"])))
            check(conn, schema)
        finally:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _prepared(conn, name: str, types: str, sql: str, args: str) -> list[dict]:
    conn.execute(f"PREPARE {name}({types}) AS {sql}")
    with psycopg.ClientCursor(conn, row_factory=dict_row) as cur:
        cur.execute(f"EXECUTE {name}{args}")
        return list(cur.fetchall())


def test_the_day_letter_reads_only_juniper_rows_never_a_guests_nor_a_game_line():
    def check(conn, _schema):
        rows = _prepared(conn, "juniper_chat", "timestamptz, timestamptz, text[]", JUNIPER_CHAT_SQL,
                         f"('{T0.isoformat()}', '{(T0 + timedelta(hours=2)).isoformat()}', "
                         f"ARRAY{list(JUNIPER_CHAT_SOURCES)!r}::text[])")
        # A game line is said to whoever holds the controller: not in the letter's chat with Juniper.
        assert [r["id"] for r in rows] == ["juniper-0", "juniper-1", "juniper-2"]

    _scenario(check)


def test_the_attention_count_of_her_turns_skips_the_guest():
    def check(conn, _schema):
        sql = CHAT_TURN_TIMES_SQL.replace(":since", "$1")
        rows = _prepared(conn, "chat_turns", "timestamp", sql, f"('{T0.replace(tzinfo=None).isoformat()}')")
        assert len(rows) == 3  # juniper-0..2; not the 3 guest rows, not Orion's line

    _scenario(check)


def test_the_chronicle_reads_guest_rows_as_guest_and_the_flip_keeps_each_rows_source():
    (q,) = [q for q in QUERIES if q.kind == "chat_turn"]

    def check(conn, schema):
        async def read():
            async with await psycopg.AsyncConnection.connect(
                    DSN, autocommit=True, row_factory=dict_row, options=f"-c search_path={schema}") as aconn:
                cur = await aconn.execute(q.sql, {"lo": T0, "hi": T0 + timedelta(hours=2)})
                return list(await cur.fetchall())

        rows = sorted(asyncio.run(read()), key=lambda r: r["created_at"])
        events = [ADAPTERS["chat_turn"](r, "America/Denver") for r in rows]
        assert [(e.source_ref, e.payload["speaker"]) for e in events] == [
            ("guest-0", "guest"), ("guest-1", "guest"), ("guest-2", "guest"), ("line-1", "orion_outreach"),
            ("juniper-0", "juniper"), ("juniper-1", "juniper"), ("juniper-2", "juniper")]
        assert [e.source_ref for e in events if e.subject_ref == SESSION] == ["juniper-0", "juniper-1", "juniper-2"]

    _scenario(check)
