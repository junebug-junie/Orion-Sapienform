"""Hub Personality tab reads and Juniper's yes / no (rev 4 move 4) on real Postgres.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test makes its own
schema with the personality migrations (+ the verdict migration) and a want_ledger table.
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

from app.personality_driver import PersonalityDriver  # noqa: E402
from app.personality_sources import PersonalitySourceReader  # noqa: E402
from app.personality_store import PersonalityStore, TendencyRetired  # noqa: E402
from test_personality_loop import Clock, Embedder, choice  # noqa: E402

SQL = REPO_ROOT / "services/orion-sql-db"
MIGRATIONS = [SQL / f"manual_migration_personality_{n}_v1.sql" for n in ("graph", "naming", "choice")]
VERDICT_MIGRATION = SQL / "manual_migration_personality_juniper_verdict_v1.sql"
WANT_LEDGER = SQL / "manual_migration_want_ledger_v1.sql"
T0 = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
SOURCE_DDL = """
CREATE TABLE curiosity_offer_decisions (run_id text PRIMARY KEY, decided_at timestamptz, arm text, offered jsonb,
  stale_offered jsonb, material_ids jsonb, turn_started_at timestamptz);
CREATE TABLE curiosity_run_outcomes (run_id text PRIMARY KEY, completed_at timestamptz, turn_ok boolean,
  realized_nats double precision, per_prior jsonb);
CREATE TABLE temporal_self_event (occurred_at timestamptz, source_kind text, label text);
CREATE TABLE temporal_self_arc (kind text, began_at timestamptz, ended_at timestamptz);
CREATE TABLE temporal_self_state (watermark timestamptz);
"""


def _strip_tx(text: str) -> str:
    return re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", text, flags=re.M)


def with_db(fn, *, verdict_table: bool = True):
    async def go():
        schema = "personality_tab_test_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                async with await AsyncConnection.connect(DSN, autocommit=True,
                                                         options=f"-c search_path={schema}") as conn:
                    await conn.execute(SOURCE_DDL)
                    for mig in [*MIGRATIONS, WANT_LEDGER, *([VERDICT_MIGRATION] if verdict_table else [])]:
                        await conn.execute(_strip_tx(mig.read_text()))
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


def driver(pool, clock):
    return PersonalityDriver(store=PersonalityStore(pool), reader=PersonalitySourceReader(pool), embed=Embedder(),
                             now=clock, tick_sec=600, read_lag_sec=120)


async def seed(pool, clock):
    """Three prime choices the same way (one node), one outreach decline, folded by a real tick."""
    store = PersonalityStore(pool)
    cs = [choice(f"prime:t{i}", T0 + timedelta(minutes=i), rank=3, declined=("pipeline", "budget")) for i in range(3)]
    cs.append(choice("outreach:o1", T0 + timedelta(minutes=5), lane="outreach", declined_all=True,
                     declined=("the gap", "a walk")))
    for c in cs:
        assert await store.insert_choices([c], written_by="bus") == 1
        await ex(pool, "UPDATE personality_choice SET created_at = %(t)s WHERE choice_id = %(c)s",
                 {"t": c.occurred_at, "c": c.choice_id})
    clock.at = T0 + timedelta(minutes=30)
    drv = driver(pool, clock)
    await drv.tick()
    return store, drv


def test_recent_choices_show_picked_passed_over_and_declines():
    async def scenario(pool):
        store, _ = await seed(pool, Clock(T0))
        rows = await store.recent_choices(limit=30)
        assert [r["choice_id"] for r in rows] == ["outreach:o1", "prime:t2", "prime:t1", "prime:t0"]
        dec = rows[0]
        assert dec["declined_all"] is True and dec["chosen"] is None and dec["lane"] == "outreach"
        assert [a["text"] for a in dec["passed_over"]] == ["the gap", "a walk"]
        pick = rows[1]
        assert pick["chosen"]["text"] == "people" and pick["chosen_market_rank"] == 3
        # Passed over in the market's order, each with its rank.
        assert [(a["text"], a["market_rank"]) for a in pick["passed_over"]] == [("pipeline", 1), ("budget", 2)]
        assert pick["tendency"]["label"].startswith("people over pipeline")
        # Prose nodes never show as the tendency a choice counted toward.
        assert all(r["tendency"] is None or r["tendency"]["disposition_id"].startswith("disp-") for r in rows)
        assert len(await store.recent_choices(limit=2)) == 2

    with_db(scenario)


def test_tendencies_carry_n_of_m_examples_and_no_prose_nodes():
    async def scenario(pool):
        await ex(pool, "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                       "centroid_n, weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                       "('old', 'recurrence', 'active', 'The perception error is high', ARRAY[1,0]::real[], 1, 99.0, "
                       "9, %(at)s, %(at)s, %(at)s)", {"at": T0})
        store, _ = await seed(pool, Clock(T0))
        body = await store.tendencies(limit=20, examples=3)
        assert body["warnings"] == []
        ids = [t["disposition_id"] for t in body["tendencies"]]
        assert "old" not in ids and ids
        people = next(t for t in body["tendencies"] if t["label"].startswith("people over pipeline"))
        assert people["n"] == 3 and people["m"] == 3 and people["juniper"] is None
        assert [e["choice_id"] for e in people["examples"]] == ["prime:t2", "prime:t1", "prime:t0"]
        assert people["examples"][0]["passed_over"][0]["text"] == "pipeline"
        assert len((await store.tendencies(limit=20, examples=1))["tendencies"][0]["examples"]) == 1

    with_db(scenario)


def test_no_retires_the_tendency_and_the_next_tick_keeps_it_retired():
    async def scenario(pool):
        clock = Clock(T0)
        store, drv = await seed(pool, clock)
        people = next(t for t in (await store.tendencies())["tendencies"] if t["label"].startswith("people over"))
        did = people["disposition_id"]

        yes = await drv.juniper_verdict(did, "yes")
        assert yes["retired"] is False and yes["status"] == people["status"]
        listed = next(t for t in (await store.tendencies())["tendencies"] if t["disposition_id"] == did)
        assert listed["juniper"]["verdict"] == "yes"

        clock.at += timedelta(minutes=1)
        no = await drv.juniper_verdict(did, "no")
        assert no["retired"] is True and no["status"] == "retired"
        (node,) = await q(pool, "SELECT status, retired_at FROM personality_disposition WHERE disposition_id = %(d)s",
                          {"d": did})
        assert node["status"] == "retired" and node["retired_at"] == clock.at
        rows = await q(pool, "SELECT verdict, retired, status_before, n_at_verdict, m_at_verdict, label_at_verdict "
                             "FROM personality_juniper_verdict WHERE disposition_id = %(d)s ORDER BY created_at",
                       {"d": did})
        assert [(r["verdict"], r["retired"]) for r in rows] == [("yes", False), ("no", True)]
        assert rows[1]["n_at_verdict"] == 3 and rows[1]["label_at_verdict"] == people["label"]
        hist = await q(pool, "SELECT status FROM personality_weight_history WHERE disposition_id = %(d)s "
                             "ORDER BY at DESC LIMIT 1", {"d": did})
        assert hist[0]["status"] == "retired"

        # Gone from the list, the frame (Prime tending, cue) and the reducer's working set.
        assert did not in [t["disposition_id"] for t in (await store.tendencies())["tendencies"]]
        frame = await store.frame(clock.at)
        assert did not in [d.disposition_id for d in frame.dispositions]
        assert did not in [n.disposition_id for n in await store.load_nodes()]
        # A later tick does not bring it back.
        clock.at += timedelta(minutes=20)
        await drv.tick()
        (node,) = await q(pool, "SELECT status FROM personality_disposition WHERE disposition_id = %(d)s", {"d": did})
        assert node["status"] == "retired"

        with pytest.raises(TendencyRetired):
            await drv.juniper_verdict(did, "yes")
        assert await drv.juniper_verdict("disp-nope", "yes") is None
        with pytest.raises(ValueError):
            await store.record_juniper_verdict(did, "maybe", clock.at)

    with_db(scenario)


def test_prose_nodes_cannot_be_answered():
    async def scenario(pool):
        await ex(pool, "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                       "centroid_n, weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                       "('old', 'recurrence', 'active', 'x', ARRAY[1,0]::real[], 1, 9.0, 9, %(at)s, %(at)s, %(at)s)",
                 {"at": T0})
        assert await PersonalityStore(pool).record_juniper_verdict("old", "no", T0) is None
        (old,) = await q(pool, "SELECT status FROM personality_disposition WHERE disposition_id = 'old'")
        assert old["status"] == "active"

    with_db(scenario)


def test_tendencies_list_without_the_verdict_table_warns():
    async def scenario(pool):
        store, _ = await seed(pool, Clock(T0))
        body = await store.tendencies()
        assert body["tendencies"] and all(t["juniper"] is None for t in body["tendencies"])
        assert body["warnings"][0].startswith("answers_table_missing")

    with_db(scenario, verdict_table=False)


def test_arm_counts_read_personality_trace_lines_by_outcome():
    async def scenario(pool):
        store = PersonalityStore(pool)
        assert (await store.arm_counts(T0 - timedelta(days=7)))["total"] == 0  # not live yet

        async def decision(i, outcome, warnings, at=T0):
            await ex(pool, "INSERT INTO want_ledger (decision_id, want_id, family, producer, verb, tier, decided_at, "
                           "outcome, reason, shadow, decision) VALUES (%(i)s, %(i)s, 'curiosity', 'p', 'v', 1, %(at)s, "
                           "%(o)s, 'r', true, %(d)s::jsonb)",
                     {"i": i, "at": at, "o": outcome, "d": json.dumps({"warnings": warnings})})

        await decision("a", "funded", ["personality:arm=care", "personality:care=0.4"])
        await decision("b", "deferred", ["personality:arm=care"])
        await decision("c", "funded", ["personality:arm=holdback"])
        await decision("d", "funded", ["something_else"])
        await decision("e", "funded", ["personality:arm=care"], at=T0 - timedelta(days=30))  # outside the window
        await ex(pool, "INSERT INTO want_ledger (decision_id, want_id, family, producer, verb, tier, decided_at, "
                       "outcome, reason, shadow, decision) VALUES ('f', 'f', 'x', 'p', 'v', 1, %(at)s, 'funded', 'r', "
                       "true, '{\"warnings\": null}'::jsonb)", {"at": T0})
        body = await store.arm_counts(T0 - timedelta(days=7))
        assert body["arms"] == {"care": {"funded": 1, "deferred": 1}, "holdback": {"funded": 1}}
        assert body["total"] == 3

    with_db(scenario)
