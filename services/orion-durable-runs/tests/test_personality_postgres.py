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
NAMING_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_personality_naming_v1.sql"
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
                    for mig in (MIGRATION, NAMING_MIGRATION):
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
        assert set(curs) == set(ps.SOURCE_NAMES) | {ps.SEED_CURSOR, ps.REKEY_CURSOR, TICK_HEARTBEAT}
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
        # j1 is a metacog journal: keyed by its source (graph.source_key), not its lane.
        assert set(active[0]["lane_recent"]) == {"dream", "substrate_narration", "chat"}
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


def test_situation_mix_groups_stamped_evidence_per_disposition():
    async def scenario(pool):
        await ex(pool, "INSERT INTO personality_disposition (disposition_id, origin, status, label, centroid, "
                       "centroid_n, weight, evidence_count, first_seen_at, last_reinforced_at, updated_at) VALUES "
                       "('d1', 'recurrence', 'active', 'one', ARRAY[1,0]::real[], 1, 2.0, 4, %(at)s, %(at)s, %(at)s), "
                       "('d2', 'recurrence', 'active', 'two', ARRAY[0,1]::real[], 1, 1.0, 1, %(at)s, %(at)s, %(at)s)",
                 {"at": T0})

        def sit(arousal, phase, company):
            return json.dumps({"situation": {"arousal": arousal, "day_phase": phase, "company": company}})

        rows = [("e1", "d1", sit("idle", "night", False)), ("e2", "d1", sit("idle", "night", True)),
                ("e3", "d1", sit("engaged", "morning", False)),
                ("e4", "d1", json.dumps({"other": 1})),  # pre-2026-10-16 row: no situation, not counted
                ("e5", "d2", "{}")]
        for eid, did, meta in rows:
            await ex(pool, "INSERT INTO personality_evidence (evidence_id, lane, source_ref, occurred_at, excerpt, "
                           "disposition_id, meta) VALUES (%(e)s, 'chat', %(e)s, %(at)s, 'x', %(d)s, %(m)s::jsonb)",
                     {"e": eid, "d": did, "m": meta, "at": T0})
        out = {r["disposition_id"]: r["situation_mix"] for r in await PersonalityStore(pool).dispositions("all")}
        assert out["d1"] == {"n": 3, "arousal": {"idle": 2, "engaged": 1}, "day_phase": {"night": 2, "morning": 1},
                             "company": {"true": 1, "false": 2}}
        assert out["d2"] == {"n": 0, "arousal": {}, "day_phase": {}, "company": {"true": 0, "false": 0}}

    with_db(scenario)


def test_rekey_hits_reads_reverie_and_journal_hits_with_meta():
    async def scenario(pool):
        rows = [("e1", "reverie", "{}"), ("e2", "journal", json.dumps({"source_kind": "metacog"})),
                ("e3", "dream", "{}")]
        for eid, lane, meta in rows:
            await ex(pool, "INSERT INTO personality_evidence (evidence_id, lane, source_ref, occurred_at, excerpt, "
                           "disposition_id, meta) VALUES (%(e)s, %(l)s, %(e)s, %(at)s, 'x', 'd1', %(m)s::jsonb)",
                     {"e": eid, "l": lane, "m": meta, "at": T0})
        hits = await PersonalityStore(pool).rekey_hits(["d1", "d2"])
        assert sorted(hits["d1"], key=lambda h: h[0]) == [("journal", {"source_kind": "metacog"}, T0),
                                                          ("reverie", {}, T0)]
        assert await PersonalityStore(pool).rekey_hits([]) == {}

    with_db(scenario)


# --- self-naming (manual_migration_personality_naming_v1.sql) ----------------------------------

N_NAME, N_SENTENCE = "The Long Watch", "I keep watching the field at night."


def _nnode(did, text, **kw):
    from orion.personality.graph import DEFAULT_PARAMS, Node, _derived_weight

    lr = kw.pop("lane_recent", {"dream": 3.0, "journal": 3.0})
    base = dict(disposition_id=did, origin="recurrence", status="active", label=f"label {did}", label_similarity=0.9,
                centroid=vec_for(text), centroid_n=5, weight=_derived_weight(lr, DEFAULT_PARAMS), evidence_count=10,
                lane_counts={"dream": 5, "journal": 5}, chosen_count=0, first_seen_at=T0 - timedelta(days=3),
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
        d2 = _nnode("d2", "something else", lane_recent={"dream": 2.0, "journal": 2.0})
        evs = [EvidenceRow(f"chat:{i}", "chat" if i % 2 else "reverie", "s", T0 - timedelta(hours=i), f"excerpt {i}",
                           "d1", 0.9 - i / 100, 0.0, {"situation": {"arousal": "strained", "company": True}} if i < 3 else {})
               for i in range(6)]
        evs.append(EvidenceRow("chat:x", "chat", "s", T0, "other", "d2", 0.9, 0.0, {}))
        await store.commit_tick(TickWrite(now=T0, evidence=evs, nodes=[d1, d2]))

        packet = await store.naming_next(T0 + timedelta(minutes=1))
        assert packet.disposition_id == "d1" and packet.evidence[0].lane in ("chat", "reverie")
        assert {e.lane for e in packet.evidence[:2]} == {"chat", "reverie"} and len(packet.evidence) == 5
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
            EvidenceRow("chat:d3", "chat", "s", T0, "near words", "d3", 0.95, 0.0, {})]))
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
