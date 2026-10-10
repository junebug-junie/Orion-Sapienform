"""Temporal Self patch 3 on real Postgres: the migration, the source SQL, the one-transaction
commit, restart identity, late rows, retention and the tolerant loader.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test creates its own
schema with the source tables' live column types (evals/fixtures/temporal_self_source_tables.sql)
and both temporal_self migrations; it never connects to POSTGRES_URI.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS), str(SERVICE_ROOT / "evals")]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

from psycopg import AsyncConnection  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

import temporal_self_fixture_db as fdb  # noqa: E402
from app import regulation_store  # noqa: E402
from app.temporal_self_chronicle import Chronicler  # noqa: E402
from app.temporal_self_sources import SourceReader  # noqa: E402
from app.temporal_self_store import ChronicleStore, StaleWriterError, WindowWrite  # noqa: E402
from orion.schemas.regulation import ArousalReadingV1  # noqa: E402
from orion.temporal_self import build_frame, initial_state  # noqa: E402
from temporal_self_world import TZ, FakeReader, FakeStore, World  # noqa: E402
from test_temporal_self_chronicle import STEP, T0, cfg, chat, scenario  # noqa: E402


def with_db(scenario_fn):
    async def go():
        schema = "ts_chronicle_test_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    await fdb.create_schema(conn)
                await scenario_fn(pool)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    asyncio.run(go())


async def load_world(pool, w: World, kinds_by_visible=None) -> None:
    """Insert the world's rows that are visible now (adapter-shaped rows -> table rows)."""
    rows: dict = {}
    for r in w.rows:
        if r.visible_at <= w.now and not getattr(r, "_loaded", False):
            rows.setdefault(r.kind, []).append(r.row)
            r._loaded = True
    rows["body_cluster"] = [] if getattr(w, "_body_loaded", False) else list(w.body["cluster"])
    w._body_loaded = True
    async with pool.connection() as conn:
        await fdb.load(conn, rows)


async def run_pg(pool, w: World, until, *, restart_at=None, **kw) -> Chronicler:
    def build():
        return Chronicler(store=ChronicleStore(pool), reader=SourceReader(pool, TZ), cfg=cfg(**kw), now=lambda: w.now)

    ch = build()
    while w.now < until:
        w.now += STEP
        await load_world(pool, w)
        out = await ch.step()
        assert out["error"] is None, out["error"]
        if restart_at is not None and w.now >= restart_at:
            ch, restart_at = build(), None
    return ch


def canon(arc_json) -> dict:
    d = arc_json if isinstance(arc_json, dict) else json.loads(arc_json)
    return json.loads(json.dumps(d, sort_keys=True))


def test_migration_is_idempotent_and_keeps_the_regulate_insert_working():
    async def scenario_fn(pool):
        async with pool.connection() as conn:
            await fdb.create_schema(conn)                       # a second apply of both migrations is a no-op
            reading = ArousalReadingV1(arousal_level="idle", since=T0, observed_at=T0)
            await conn.execute(regulation_store.INSERT_EVENT_SQL, regulation_store.transition_row(None, reading, "2026-10-10"))
            row = await (await conn.execute("SELECT late_unfolded FROM temporal_self_event")).fetchone()
        assert row["late_unfolded"] is False

    with_db(scenario_fn)


def test_source_sql_reads_exactly_what_the_adapters_read():
    async def scenario_fn(pool):
        w = scenario()
        w.now = T0 + timedelta(hours=1)
        # In-flight visual attempt: not a deferral yet.
        w.add("visual_deferral", {"attempt_id": "att-1", "started_at": T0 + timedelta(minutes=7), "outcome": "active",
                                  "result_json": {"reason": None, "detail": {"state": "hot"}}})
        await load_world(pool, w)
        lo, hi = T0 - timedelta(minutes=5), T0 + timedelta(minutes=50)
        ticks, events = await SourceReader(pool, TZ).read(lo, hi)
        fticks, fevents = await FakeReader(w).read(lo, hi)
        fevents = [e for e in fevents if e.event_id != "visual_deferral:att-1"]
        assert sorted(t.log_id for t in ticks) == sorted(t.log_id for t in fticks) and len(ticks) == 39
        assert sorted(e.model_dump_json() for e in events) == sorted(e.model_dump_json() for e in fevents)
        assert {e.source_kind for e in events} >= {"chat_turn", "field_dominance_run", "reverie_chain", "metacog_observation"}
        async with pool.connection() as conn:
            await conn.execute("UPDATE reverie_visual_attempt SET outcome = 'deferred_thermal' WHERE attempt_id = 'att-1'")
        _, events = await SourceReader(pool, TZ).read(lo, hi)
        assert "visual_deferral:att-1" in {e.event_id for e in events}

    with_db(scenario_fn)


def test_cabinet_text_timestamps_with_either_separator_are_read():
    async def scenario_fn(pool):
        async with pool.connection() as conn:
            await fdb.insert(conn, "orion_biometrics_summary", [
                {"node": "athena", "timestamp": "2026-10-10 15:01:00.5+00", "measurements": {"cabinet_temp_c": 25.0}},
                {"node": "athena", "timestamp": "2026-10-10T15:02:00+00:00", "measurements": {"cabinet_temp_c": 26.0}},
                {"node": "athena", "timestamp": "2026-10-10 16:30:00+00", "measurements": {"cabinet_temp_c": 99.0}},
                {"node": "circe", "timestamp": "2026-10-10 15:01:30+00", "measurements": {"cabinet_temp_c": 50.0}},
            ])
        body = await SourceReader(pool, TZ).read_body(T0, T0 + timedelta(minutes=10))
        assert sorted(r["cabinet_temp_c"] for r in body["cabinet"]) == [25.0, 26.0]

    with_db(scenario_fn)


def test_live_postgres_run_matches_in_memory_and_survives_a_restart():
    w_mem, mem = scenario(), FakeStore()
    ch_mem = Chronicler(store=mem, reader=FakeReader(w_mem), cfg=cfg(), now=lambda: w_mem.now)
    while w_mem.now < T0 + timedelta(hours=2):
        w_mem.now += STEP
        asyncio.run(ch_mem.step())

    async def scenario_fn(pool):
        w = scenario()
        ch = await run_pg(pool, w, T0 + timedelta(hours=2), restart_at=T0 + timedelta(minutes=13))
        store = ChronicleStore(pool)
        arcs = await store.arcs("2026-10-10")
        assert {a["arc_id"]: canon(a) for a in arcs} == {k: canon(v) for k, v in mem.arcs.items()}
        frame = await store.frame()
        assert canon(frame) == canon(mem.frame) and frame["skipped_at_or_before_watermark"] == 0
        cursors = {c["source_kind"]: c for c in await store.cursors()}
        assert cursors["read_watermark"]["last_occurred_at"] == ch.watermark
        assert cursors["broadcast_tick"]["last_source_ref"] == "bc-00038"
        assert await store.day("2026-10-10") is None
        loaded = await store.load_state()
        assert loaded.state == ch.state and loaded.origin.isoformat() == "2026-10-10T06:00:00+00:00"

    with_db(scenario_fn)


def test_late_row_is_stored_unfolded_and_counted_once():
    async def scenario_fn(pool):
        w = scenario()
        w.add("chat_turn", chat(9, T0 + timedelta(minutes=40), session="s2"), delay=timedelta(seconds=200))
        await run_pg(pool, w, T0 + timedelta(hours=1, minutes=30), read_lag_sec=60.0)
        store = ChronicleStore(pool)
        assert await store.late_counts(T0 - timedelta(days=1)) == {"chat_turn": 1}
        assert (await store.frame())["skipped_at_or_before_watermark"] == 1
        assert not [a for a in await store.arcs("2026-10-10") if a["subject_ref"] == "s2"]

    with_db(scenario_fn)


def test_commit_refuses_a_moved_watermark_and_writes_nothing():
    async def scenario_fn(pool):
        store = ChronicleStore(pool)
        s = initial_state()
        frame = build_frame(s, T0)
        await store.commit_window(WindowWrite(state=s, watermark=T0, frame=frame, now=T0))
        with pytest.raises(StaleWriterError):
            await store.commit_window(WindowWrite(state=s, watermark=T0 + STEP, frame=frame, now=T0,
                                                  expected_prev=T0 - STEP))
        assert (await store.load_state()).watermark == T0

    with_db(scenario_fn)


def test_tolerant_loader_reports_a_stale_state_row():
    async def scenario_fn(pool):
        async with pool.connection() as conn:
            await conn.execute("INSERT INTO temporal_self_state (state_id, watermark, origin, reducer_version, state_gz) "
                               "VALUES ('orion', %s, %s, 'old', %s)", (T0, T0, b"\x1f\x8bnot really gzip"))
        loaded = await ChronicleStore(pool).load_state()
        assert loaded.state is None and loaded.watermark == T0 and loaded.error

    with_db(scenario_fn)


def test_retention_covers_chronicle_rows_and_arousal_transitions():
    async def scenario_fn(pool):
        now = T0
        old = now - timedelta(days=31)
        async with pool.connection() as conn:
            reading = ArousalReadingV1(arousal_level="idle", since=old, observed_at=old)
            await conn.execute(regulation_store.INSERT_EVENT_SQL, regulation_store.transition_row(None, reading, "2026-09-09"))
            reading = ArousalReadingV1(arousal_level="engaged", since=now, observed_at=now)
            await conn.execute(regulation_store.INSERT_EVENT_SQL, regulation_store.transition_row(None, reading, "2026-10-10"))
            await conn.execute("INSERT INTO temporal_self_arc (arc_id, day_id, kind, subject_ref, began_at, ended_at, status, arc_json) "
                               "VALUES ('old', '2026-07-01', 'attention', 'x', %s, %s, 'closed', '{}'), "
                               "('open', '2026-07-01', 'concern', 'y', %s, NULL, 'open', '{}')",
                               (now - timedelta(days=100), now - timedelta(days=100), now - timedelta(days=100)))
            await conn.execute("INSERT INTO temporal_self_day (day_id, closed_at, day_json) VALUES ('2025-01-01', %s, '{}')",
                               (now - timedelta(days=400),))
        out = await ChronicleStore(pool).retention(now, event_days=30, arc_days=90, day_days=365)
        assert out == {"event": 1, "arc": 1, "day": 1}
        async with pool.connection() as conn:
            left = await (await conn.execute("SELECT source_kind, label FROM temporal_self_event")).fetchall()
            arcs = await (await conn.execute("SELECT arc_id FROM temporal_self_arc")).fetchall()
        assert [r["label"] for r in left] == ["engaged"] and [r["arc_id"] for r in arcs] == ["open"]

    with_db(scenario_fn)
