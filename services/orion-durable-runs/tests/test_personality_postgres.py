"""Personality graph loop (rev 4: choices only) on real Postgres: the migrations, the curiosity sync
and its backfill, bus intake, the one-transaction tick, restart from cursors, the route reads.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test creates its own
schema with the curiosity tables' live column types and the personality migrations; it never
connects to POSTGRES_URI. The embedder is a deterministic fake.
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
from test_personality_loop import Clock, Embedder, choice, vec_for  # noqa: E402

MIGRATIONS = [REPO_ROOT / f"services/orion-sql-db/manual_migration_personality_{n}_v1.sql"
              for n in ("graph", "naming", "choice")]
T0 = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)

# Live column types (information_schema), only the columns the curiosity sync reads.
SOURCE_DDL = """
CREATE TABLE curiosity_offer_decisions (run_id text PRIMARY KEY, decided_at timestamptz, arm text, offered jsonb,
  stale_offered jsonb, material_ids jsonb, turn_started_at timestamptz);
CREATE TABLE curiosity_run_outcomes (run_id text PRIMARY KEY, completed_at timestamptz, turn_ok boolean,
  realized_nats double precision, per_prior jsonb);
CREATE TABLE temporal_self_event (occurred_at timestamptz, source_kind text, label text);
CREATE TABLE temporal_self_arc (kind text, began_at timestamptz, ended_at timestamptz);
CREATE TABLE temporal_self_state (watermark timestamptz);
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
                    for mig in MIGRATIONS:
                        await conn.execute(re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", mig.read_text(), flags=re.M))
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


def driver(pool, clock, emb=None):
    return PersonalityDriver(store=PersonalityStore(pool), reader=PersonalitySourceReader(pool), embed=emb or Embedder(),
                             now=clock, tick_sec=600, read_lag_sec=120)


async def curiosity_run(pool, run_id, at, tested, arm="value_order"):
    offered = [{"prior_id": "self:a_thing_20261015", "rank": 1, "slot": "order"},
               {"prior_id": "self:b_thing", "rank": 2, "slot": "order"},
               {"prior_id": "self:c_thing", "rank": 3, "slot": "coverage"}]
    await ex(pool, "INSERT INTO curiosity_offer_decisions VALUES (%(r)s, %(at)s, %(arm)s, %(o)s::jsonb, '[]', '[]', %(at)s)",
             {"r": run_id, "at": at, "arm": arm, "o": json.dumps(offered if arm != "self_inquiry" else [])})
    await ex(pool, "INSERT INTO curiosity_run_outcomes VALUES (%(r)s, %(at)s, true, 0.1, %(p)s::jsonb)",
             {"r": run_id, "at": at, "p": json.dumps([{"kind": "tested", "prior_id": p} for p in tested])})


async def insert_at(pool, store, c, created_at):
    """A bus choice, with its created_at pinned (the column defaults to now(); tests need a clock)."""
    assert await store.insert_choices([c], written_by="bus") == 1
    await ex(pool, "UPDATE personality_choice SET created_at = %(t)s WHERE choice_id = %(c)s",
             {"t": created_at, "c": c.choice_id})


def test_full_loop_on_postgres():
    # The sync's rows get created_at = the DB's now(); a clock far past it makes them readable at once.
    t0 = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)

    async def scenario(pool):
        store = PersonalityStore(pool)
        await curiosity_run(pool, "run1", t0 - timedelta(days=2), ["self:b_thing"])
        await curiosity_run(pool, "run2", t0 - timedelta(days=1), ["self:x"], arm="self_inquiry")
        clock = Clock(t0)
        out = await driver(pool, clock).tick()
        # The backfill: run1 -> one choice, folded in the same tick; the self-inquiry run is never a choice.
        assert out["curiosity_synced"] == 1 and out["evidence"] == 1
        (row,) = await q(pool, "SELECT * FROM personality_choice")
        assert row["choice_id"] == "curiosity:run1:self:b_thing" and row["written_by"] == "curiosity_sync"
        assert row["chosen"]["text"] == "b thing" and row["chosen_market_rank"] == 2
        assert [a["key"] for a in row["alternatives"]] == ["self:a_thing_20261015", "self:c_thing"]
        (cur,) = await q(pool, "SELECT * FROM personality_cursor WHERE source = 'curiosity_sync'")
        assert cur["cursor_ref"] == "run1"
        assert TICK_HEARTBEAT not in await store.load_cursors()

        # A bus choice (created_at pinned to the test clock), and a re-delivery that is a no-op.
        await insert_at(pool, store, choice("prime:t1", t0 + timedelta(minutes=1), rank=3), t0 + timedelta(minutes=1))
        assert await store.insert_choices([choice("prime:t1", t0)], written_by="bus") == 0
        clock.at = t0 + timedelta(minutes=10)
        drv = driver(pool, clock)  # a restart: fresh driver, same tables
        out = await drv.tick()
        assert out["evidence"] == 1 and out["choices_read"] == 1
        ev = {r["evidence_id"]: r for r in await q(pool, "SELECT * FROM personality_evidence")}
        assert set(ev) == {"choice:curiosity:run1:self:b_thing", "choice:prime:t1"}
        assert ev["choice:prime:t1"]["lane"] == "prime" and ev["choice:prime:t1"]["excerpt"] == "people over pipeline"
        nodes = await q(pool, "SELECT * FROM personality_disposition")
        assert {n["origin"] for n in nodes} == {"choice"} and all(n["opposed_count"] == 0 for n in nodes)
        assert any(n["label"] == "people over pipeline, 1 of 1" for n in nodes)
        (cur,) = await q(pool, "SELECT * FROM personality_cursor WHERE source = 'choice'")
        assert cur["cursor_ref"] == "prime:t1"

        # Re-upserted curiosity outcome: completed_at moves, the run is read again, nothing new.
        # Hub replaces per_prior on a re-upsert: a different tested set must not add a second decision.
        await ex(pool, "UPDATE curiosity_run_outcomes SET completed_at = %(at)s, per_prior = %(p)s::jsonb "
                       "WHERE run_id = 'run1'",
                 {"at": t0 + timedelta(minutes=11),
                  "p": json.dumps([{"kind": "tested", "prior_id": "self:a_thing_20261015"}])})
        clock.at = t0 + timedelta(minutes=20)
        out = await drv.tick()
        assert out["curiosity_synced"] == 0 and out["evidence"] == 0
        assert len(await q(pool, "SELECT 1 FROM personality_choice")) == 2

        frame = await store.frame(clock.at + timedelta(hours=5))
        (beat,) = await q(pool, "SELECT cursor_at FROM personality_cursor WHERE source = %(s)s", {"s": TICK_HEARTBEAT})
        assert frame.generated_at == beat["cursor_at"]
        assert frame.candidate_count == 2 and frame.active_count == 0
        # Two decisions in the window: run1 (rank 2) and prime:t1 (rank 3) -> both own way.
        assert frame.choice_style.choices == 2 and frame.choice_style.took_top == 0
        assert frame.evidence_rows == 2
        rows = await store.dispositions("all", limit=50)
        assert len(rows) == 2 and "centroid" not in rows[0] and "opposed_count" in rows[0]

    with_db(scenario)


def test_prose_nodes_are_invisible_to_the_loop_and_routes():
    async def scenario(pool):
        await ex(pool, "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                       "centroid_n, weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                       "('old', 'recurrence', 'active', 'The perception error is high', ARRAY[1,0]::real[], 1, 9.0, 9, "
                       "%(at)s, %(at)s, %(at)s)", {"at": T0})
        store = PersonalityStore(pool)
        assert await store.load_nodes() == [] and await store.status_counts() == {}
        assert await store.dispositions("all") == []
        frame = await store.frame(T0)
        assert frame.dispositions == [] and frame.active_count == 0
        await driver(pool, Clock(T0)).tick()
        (old,) = await q(pool, "SELECT status, weight FROM personality_disposition WHERE disposition_id = 'old'")
        assert old == {"status": "active", "weight": 9.0}  # untouched until the retire job runs

    with_db(scenario)


def test_keyset_tie_break_drains_choices_sharing_one_created_at():
    async def scenario(pool):
        store = PersonalityStore(pool)
        at = T0 + timedelta(minutes=1)
        for i in ("a", "b", "c"):
            await insert_at(pool, store, choice(f"prime:{i}", at), at)
        reader = PersonalitySourceReader(pool, limit=2)
        hi = T0 + timedelta(minutes=10)
        first = await reader.read_choices(None, hi)
        assert [r["choice_id"] for r in first] == ["prime:a", "prime:b"]
        second = await reader.read_choices(ps.Cursor(at, "prime:b"), hi)
        assert [r["choice_id"] for r in second] == ["prime:c"]
        assert await reader.read_choices(ps.Cursor(at, "prime:c"), hi) == []
        assert ps.choice_from_row(first[0]) == choice("prime:a", at)

    with_db(scenario)


def test_situation_mix_groups_stamped_evidence_per_disposition():
    async def scenario(pool):
        await ex(pool, "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                       "centroid_n, weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                       "('d1', 'choice', 'active', 'one', ARRAY[1,0]::real[], 1, 2.0, 4, %(at)s, %(at)s, %(at)s), "
                       "('d2', 'choice', 'active', 'two', ARRAY[0,1]::real[], 1, 1.0, 1, %(at)s, %(at)s, %(at)s)",
                 {"at": T0})

        def sit(arousal, phase, company):
            return json.dumps({"situation": {"arousal": arousal, "day_phase": phase, "company": company}})

        rows = [("e1", "d1", sit("idle", "night", False)), ("e2", "d1", sit("idle", "night", True)),
                ("e3", "d1", sit("engaged", "morning", False)),
                ("e4", "d1", json.dumps({"other": 1})),
                ("e5", "d2", "{}")]
        for eid, did, meta in rows:
            await ex(pool, "INSERT INTO personality_evidence (evidence_id, lane, source_ref, occurred_at, excerpt, "
                           "disposition_id, meta) VALUES (%(e)s, 'prime', %(e)s, %(at)s, 'x', %(d)s, %(m)s::jsonb)",
                     {"e": eid, "d": did, "m": meta, "at": T0})
        out = {r["disposition_id"]: r["situation_mix"] for r in await PersonalityStore(pool).dispositions("all")}
        assert out["d1"] == {"n": 3, "arousal": {"idle": 2, "engaged": 1}, "day_phase": {"night": 2, "morning": 1},
                             "company": {"true": 1, "false": 2}}
        assert out["d2"] == {"n": 0, "arousal": {}, "day_phase": {}, "company": {"true": 0, "false": 0}}

    with_db(scenario)


# --- self-naming (manual_migration_personality_naming_v1.sql) ----------------------------------

N_NAME, N_SENTENCE = "The Long Watch", "I keep watching the field at night."


def _nnode(did, text, **kw):
    from orion.personality.graph import DEFAULT_PARAMS, Node, _derived_weight

    lr = kw.pop("lane_recent", {"curiosity": 3.0, "prime": 3.0})
    base = dict(disposition_id=did, origin="choice", status="active", label=f"label {did}", label_similarity=0.9,
                centroid=vec_for(text), centroid_n=5, weight=_derived_weight(lr, DEFAULT_PARAMS), evidence_count=10,
                lane_counts={"curiosity": 5, "prime": 5}, chosen_count=0, first_seen_at=T0 - timedelta(days=3),
                last_reinforced_at=T0 - timedelta(hours=1), retired_at=None, updated_at=T0, lane_recent=lr,
                last_folded_at=T0)
    base.update(kw)
    return Node(**base)


async def _naming_row(pool, nid, did, verdict, packet, created, **kw):
    await ex(pool, "INSERT INTO orion_disposition_naming (naming_id, run_id, packet_id, disposition_id, verdict, name, "
                   "sentence, merge_with, evidence_refs, shown, hub_check, written_at, created_at) VALUES (%(n)s, 'run', "
                   "%(p)s, %(d)s, %(v)s, %(name)s, %(s)s, %(m)s, '[]', %(shown)s::jsonb, 'accepted', %(w)s, %(c)s)",
             {"n": nid, "p": packet.packet_id, "d": did, "v": verdict, "name": kw.get("name"), "s": kw.get("sentence"),
              "m": kw.get("merge_with"), "shown": packet.model_dump_json(), "w": created, "c": created})


def test_naming_source_columns_routes_and_unmerge_on_postgres():
    from orion.personality.graph import EvidenceRow, HistoryRow
    from app.personality_store import NAMING_CURSOR, TickWrite
    import importlib.util

    async def scenario(pool):
        store = PersonalityStore(pool)
        clock = Clock(T0)
        drv = driver(pool, clock)
        await drv.tick()  # first boot: source cursors at T0
        d1 = _nnode("d1", f"{N_NAME} {N_SENTENCE}")
        d2 = _nnode("d2", "something else", lane_recent={"curiosity": 2.0, "prime": 2.0})
        evs = [EvidenceRow(f"choice:c{i}", "chat" if i % 2 else "prime", "s", T0 - timedelta(hours=i), f"excerpt {i}",
                           "d1", 0.9 - i / 100, 0.0, {"situation": {"arousal": "strained", "company": True}} if i < 3 else {})
               for i in range(6)]
        evs.append(EvidenceRow("choice:x", "chat", "s", T0, "other", "d2", 0.9, 0.0, {}))
        await store.commit_tick(TickWrite(now=T0, evidence=evs, nodes=[d1, d2]))

        packet = await store.naming_next(T0 + timedelta(minutes=1))
        assert packet.disposition_id == "d1" and packet.evidence[0].lane in ("chat", "prime")
        assert {e.lane for e in packet.evidence[:2]} == {"chat", "prime"} and len(packet.evidence) == 5
        assert packet.situation_mix["arousal"] == {"strained": 3} and packet.situation_coverage == 0.5
        assert packet.neighbors[0].disposition_id == "d2"

        await _naming_row(pool, "n1", "d1", "name", packet, T0 + timedelta(minutes=2), name=N_NAME, sentence=N_SENTENCE)
        clock.at = T0 + timedelta(minutes=10)
        await drv.tick()
        (r,) = await q(pool, "SELECT * FROM personality_disposition WHERE disposition_id = 'd1'")
        assert r["name"] == N_NAME and r["name_sentence"] == N_SENTENCE and r["orion_verdict"] == "named"
        assert r["naming_id"] == "n1" and r["shown_count"] == 1 and r["verdict_evidence_count"] == 10
        assert r["name_grounding_rank"] == 1 and r["name_grounding_margin"] > 0 and r["last_shown_at"] is not None
        (c,) = await q(pool, "SELECT cursor_ref FROM personality_cursor WHERE source = %(s)s", {"s": NAMING_CURSOR})
        assert c["cursor_ref"] == "n1"

        # d1 now has a verdict; d2 is next. Reading the packet marks nothing as shown.
        nxt = await store.naming_next(clock.at)
        assert nxt.disposition_id == "d2"
        (r2,) = await q(pool, "SELECT shown_count FROM personality_disposition WHERE disposition_id = 'd2'")
        assert r2["shown_count"] == 0

        frame = await store.frame(clock.at)
        assert {d.disposition_id: d.name for d in frame.dispositions} == {"d1": N_NAME, "d2": None}
        rows = await store.dispositions(status="all")
        by = {x["disposition_id"]: x for x in rows}
        assert by["d1"]["name"] == N_NAME and by["d1"]["name_grounding_rank"] == 1 and "merged_into" in by["d1"]
        log = await store.namings("d1")
        assert [x["naming_id"] for x in log] == ["n1"] and log[0]["shown"]["packet_id"] == packet.packet_id

        # Reject d2: it leaves the frame, stays in the table.
        await _naming_row(pool, "n2", "d2", "reject", nxt, clock.at - timedelta(minutes=5),
                          sentence="These are machine status lines, not me.")
        clock.at = T0 + timedelta(minutes=20)
        await drv.tick()
        frame = await store.frame(clock.at)
        assert [d.disposition_id for d in frame.dispositions] == ["d1"]

        # MERGE d3 into d1 (cosine >= 0.85), then roll it back with the unmerge script.
        near = [x + 0.1 * y for x, y in zip(vec_for(f"{N_NAME} {N_SENTENCE}"), vec_for("noise"))]
        d3 = _nnode("d3", "unused", centroid=near, lane_recent={"chat": 2.0}, lane_counts={"chat": 4},
                    evidence_count=4, centroid_n=2, updated_at=clock.at)
        await store.commit_tick(TickWrite(now=clock.at, nodes=[d3], history=[HistoryRow("d3", clock.at, d3.weight, "active", 4)], evidence=[
            EvidenceRow("choice:d3", "chat", "s", T0, "near words", "d3", 0.95, 0.0, {})]))
        pkt3 = await store.naming_next(clock.at)
        assert pkt3.disposition_id == "d3"
        await _naming_row(pool, "n3", "d3", "merge", pkt3, clock.at - timedelta(minutes=5), merge_with="d1")
        clock.at = T0 + timedelta(minutes=30)
        await drv.tick()
        rows = {x["disposition_id"]: x for x in await q(pool, "SELECT * FROM personality_disposition")}
        assert rows["d3"]["status"] == "retired" and rows["d3"]["merged_into"] == "d1"
        assert rows["d1"]["evidence_count"] == 14 and rows["d1"]["lane_counts"]["chat"] == 4

        spec = importlib.util.spec_from_file_location("unmerge", SERVICE_ROOT / "scripts/personality_unmerge.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dsn = DSN + ("&" if "?" in DSN else "?") + f"options=-csearch_path%3D{(await q(pool, 'SELECT current_schema() AS s'))[0]['s']}"
        dry = await asyncio.to_thread(mod.run, dsn, "d3", False, clock.at)
        assert dry["after"]["b"]["status"] == "active" and dry["after"]["a"]["evidence_count"] == 10
        (still,) = await q(pool, "SELECT status FROM personality_disposition WHERE disposition_id = 'd3'")
        assert still["status"] == "retired"  # dry run wrote nothing
        await asyncio.to_thread(mod.run, dsn, "d3", True, clock.at + timedelta(seconds=1))
        rows = {x["disposition_id"]: x for x in await q(pool, "SELECT * FROM personality_disposition")}
        assert rows["d3"]["status"] == "active" and rows["d3"]["merged_into"] is None
        assert rows["d1"]["evidence_count"] == 10 and "chat" not in rows["d1"]["lane_counts"]

    with_db(scenario)
