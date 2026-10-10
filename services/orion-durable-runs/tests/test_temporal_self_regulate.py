"""temporal_self.update / regulate node (Temporal Self rev 4, order 3 "R2/R3 core").

The graph and driver run on an in-memory saver with fake reads; the store's SQL runs against an
isolated Postgres schema when ORION_ADMISSION_TEST_DSN is set (CI provides one).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.runner import SELF_DRIVEN_WORKFLOWS  # noqa: E402
from app.temporal_self_driver import TemporalSelfDriver, coalesce, day_id_for, event_from_chat_turn, thread_for  # noqa: E402
from app.temporal_self_graph import TemporalSelfDeps  # noqa: E402
from orion.schemas.drive_reading import DriveReadingV1  # noqa: E402
from orion.schemas.regulation import TEMPORAL_SELF_WORKFLOW, ArousalInputsV1, RegulationStateV1  # noqa: E402

T0 = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)   # 12:00 America/Denver


class World:
    """Fake reads with a movable clock."""

    def __init__(self):
        self.now = T0
        self.minutes = 300.0            # Juniper quiet for 5 h (DB view)
        self.turn_ok = True
        self.reflex, self.thermal = None, "normal"
        self.depth, self.sustained, self.gpu_age = 0, 0.0, 2.0
        self.projected: list[RegulationStateV1] = []
        self.transitions: list[tuple] = []
        self.fail_project = False
        self.fail_record = False
        self.drives: list = []
        self.seen_last_turn: list = []

    async def read_inputs(self, now, prev, last_turn_at):
        self.seen_last_turn.append(last_turn_at)
        minutes = self.minutes
        if last_turn_at is not None:
            minutes = min(minutes, (now - last_turn_at).total_seconds() / 60.0)
        return ArousalInputsV1(
            observed_at=now, juniper_turn_read_ok=self.turn_ok, minutes_since_juniper_turn=minutes,
            cabinet_read_ok=True, cabinet_reflex=self.reflex, cabinet_thermal_state=self.thermal,
            gpu_state_age_sec=self.gpu_age, gpu_queue_depth=self.depth, gpu_queue_sustained_sec=self.sustained)

    async def read_drives(self, now):
        return list(self.drives), []

    async def project(self, model):
        if self.fail_project:
            raise ConnectionError("redis down")
        self.projected.append(model)

    async def record(self, prev, new, day_id):
        if self.fail_record:
            return False
        self.transitions.append((prev.arousal_level if prev else None, new.arousal_level, day_id))
        return True

    def deps(self, **kw):
        return TemporalSelfDeps(read_inputs=self.read_inputs, read_drives=self.read_drives, project=self.project,
                                record_transition=self.record, now=lambda: self.now, **kw)


def driver(world, saver=None, **kw):
    return TemporalSelfDriver(checkpointer=saver or InMemorySaver(), deps=world.deps(**kw), tick_sec=120.0)


def tick(d):
    return asyncio.run(d.step({"event_id": f"tick:{d._deps.now().isoformat()}", "kind": "tick"}))


def test_workflow_is_self_driven():
    assert TEMPORAL_SELF_WORKFLOW in SELF_DRIVEN_WORKFLOWS


def test_thread_is_bucketed_by_local_date():
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Denver")
    late = datetime(2026, 10, 11, 3, 0, tzinfo=timezone.utc)   # 21:00 on 10-10 in Denver
    assert day_id_for(late, tz) == "2026-10-10"
    assert thread_for("2026-10-10") == "temporal_self:orion:2026-10-10"


def test_idle_step_projects_and_records_one_transition():
    w = World()
    d = driver(w)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "idle"
    assert w.transitions == [(None, "idle", "2026-10-10")]
    w.now += timedelta(seconds=120)
    tick(d)
    assert len(w.transitions) == 1                                # quiet step: no row
    assert w.projected[-1].arousal.since == T0
    assert d.health()["arousal_level"] == "idle"


def test_juniper_turn_wakes_engaged_within_one_step_and_outreach_does_not():
    w = World()
    d = driver(w)
    tick(d)
    assert event_from_chat_turn({"prompt": "", "client_meta": {"unsolicited": True}}, w.now) is None
    w.now += timedelta(seconds=7)
    ev = event_from_chat_turn({"prompt": "hey Orion", "source": "hub_orion", "correlation_id": "c1"}, w.now)
    asyncio.run(d.step(coalesce([{"kind": "tick", "event_id": "t"}, ev])))
    assert w.projected[-1].arousal.arousal_level == "engaged"
    assert w.transitions[-1][:2] == ("idle", "engaged")
    # The bus turn is remembered across steps even before sql-writer's row lands (DB says 300 min).
    w.now += timedelta(minutes=10)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "engaged"
    w.now += timedelta(minutes=40)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "idle"


def test_strain_holds_through_a_restart_and_since_survives():
    w = World()
    saver = InMemorySaver()
    d = driver(w, saver)
    w.depth, w.sustained = 4, 320.0
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "strained"
    w.depth, w.sustained = 0, 0.0
    w.now += timedelta(seconds=120)
    tick(d)
    # Restart: a new driver on the same saver resumes the day thread's checkpoint.
    d2 = driver(w, saver)
    w.now += timedelta(seconds=120)
    tick(d2)
    a = w.projected[-1].arousal
    assert a.arousal_level == "strained" and a.since == T0 and a.strain_clear_since == T0 + timedelta(seconds=120)
    for _ in range(4):
        w.now += timedelta(seconds=120)
        tick(d2)
    assert w.projected[-1].arousal.arousal_level == "idle"
    assert [t[:2] for t in w.transitions] == [(None, "strained"), ("strained", "idle")]


def test_new_local_day_seeds_from_yesterday():
    w = World()
    saver = InMemorySaver()
    d = driver(w, saver)
    w.now = datetime(2026, 10, 11, 5, 58, tzinfo=timezone.utc)   # 23:58 Denver, 10-10
    tick(d)
    w.now += timedelta(minutes=4)                                  # 00:02 Denver, 10-11
    tick(d)
    a = w.projected[-1].arousal
    assert a.arousal_level == "idle" and a.since == datetime(2026, 10, 11, 5, 58, tzinfo=timezone.utc)
    assert len(w.transitions) == 1


def test_redis_down_and_history_write_failure_do_not_stop_the_step():
    w = World()
    w.fail_project, w.fail_record = True, True
    d = driver(w)
    out = tick(d)
    state = RegulationStateV1.model_validate(out["regulation"])
    assert state.arousal.arousal_level == "idle"
    assert "project_failed" in state.warnings and "transition_write_failed" in state.warnings


def test_disabled_flag_reads_unknown():
    w = World()
    d = driver(w, arousal_enabled=False)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "unknown"
    assert w.projected[-1].arousal.reasons == ["disabled"]


def test_stale_input_reads_unknown_not_idle():
    w = World()
    w.turn_ok = False
    d = driver(w)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "unknown"


def test_drive_reading_is_embedded_verbatim():
    w = World()
    w.drives = [DriveReadingV1(observed_at=T0, level=1.2, threshold=3.0, state="building", source_ref="dp-1")]
    d = driver(w)
    tick(d)
    assert w.projected[-1].drives == w.drives


def test_old_cabinet_seed_is_dropped_after_an_outage():
    """Review finding: a stored hot/critical cabinet state must not seed the reflex after a gap."""
    w = World()
    seen = []

    async def read_inputs(now, prev, last_turn_at):
        seen.append(prev)
        return await World.read_inputs(w, now, prev, last_turn_at)

    w.read_inputs = read_inputs
    w.reflex, w.thermal = "cabinet_hot", "hot"
    d = driver(w)
    tick(d)
    w.now += timedelta(seconds=120)
    tick(d)
    assert seen[-1] is not None and seen[-1].cabinet_thermal_state == "hot"     # short gap: seeded
    w.now += timedelta(hours=1)
    tick(d)
    assert seen[-1] is None                                                     # outage: not seeded


def test_stale_gpu_host_does_not_outrank_a_fresh_one():
    from app import regulation_store

    class Conn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, sql, params=None):
            now = T0
            rows = [{"host": "old", "generated_at": now - timedelta(seconds=60 + 5 * i), "queue_depth": {"agent": 9}}
                    for i in range(60)]
            rows += [{"host": "circe", "generated_at": now - timedelta(seconds=5 * i), "queue_depth": {}}
                     for i in range(5)]

            class R:
                async def fetchall(self_inner):
                    return rows
            return R()

    class Pool:
        def connection(self):
            return Conn()

    out = asyncio.run(regulation_store._gpu(Pool(), T0, floor=2, sustain_sec=300.0))
    assert out["gpu_queue_depth"] == 0 and out["gpu_state_age_sec"] == 0.0


# --- store SQL on real Postgres -------------------------------------------------------------

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_temporal_self_event_v1.sql"
HISTORY_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_regulation_history.sql"


async def _with_schema(scenario):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = "regulation_test_" + uuid4().hex
    async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False,
                                       kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                                               "options": f"-c search_path={schema}"}) as pool:
            await scenario(pool)
    finally:
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_store_reads_and_transition_write_on_postgres():
    from app import regulation_store

    async def scenario(pool):
        async with pool.connection() as conn:
            for sql in (MIGRATION.read_text(), HISTORY_MIGRATION.read_text()):
                await conn.execute(sql, prepare=False)
            await conn.execute("""CREATE TABLE chat_history_log (id varchar PRIMARY KEY, source varchar, prompt text,
                                  client_meta jsonb, created_at timestamp DEFAULT now())""")
            await conn.execute("""CREATE TABLE orion_biometrics_summary (node varchar, timestamp varchar,
                                  measurements jsonb)""")
            await conn.execute("""INSERT INTO chat_history_log (id, source, prompt, client_meta, created_at) VALUES
                ('j', 'hub_orion', 'hello', NULL, LOCALTIMESTAMP - interval '50 minutes'),
                ('o', NULL, NULL, '{"unsolicited": true}', LOCALTIMESTAMP - interval '1 minutes')""")
        now = datetime.now(timezone.utc)
        async with pool.connection() as conn:
            for i in range(10):
                ts = (now - timedelta(seconds=30 * i)).strftime("%Y-%m-%d %H:%M:%S.%f+00")
                await conn.execute("INSERT INTO orion_biometrics_summary VALUES ('athena', %s, %s)",
                                   (ts, json.dumps({"cabinet_temp_c": 34.5})))
            for i in range(70):
                await conn.execute(
                    """INSERT INTO gpu_pool_state_history (snapshot_id, generated_at, host, mode, config_digest,
                       backlog_depth, queue_depth) VALUES (%s, %s, 'circe', 'enforce', 'd', '{}', %s)""",
                    (f"s{i}", now - timedelta(seconds=5 * i), json.dumps({"agent": 3})))
        inputs = await regulation_store.read_arousal_inputs(pool, now, None, gpu_queue_floor=2, gpu_sustain_sec=300.0)
        assert inputs.juniper_turn_read_ok and 49.0 < inputs.minutes_since_juniper_turn < 52.0
        assert inputs.cabinet_read_ok and inputs.cabinet_reflex == "cabinet_hot" and inputs.cabinet_temp_c == 34.5
        assert inputs.gpu_queue_depth == 3 and inputs.gpu_queue_sustained_sec >= 300.0
        assert inputs.gpu_state_age_sec < 5.0

        from orion.regulation.arousal import classify_arousal

        reading = classify_arousal(None, inputs)
        assert reading.arousal_level == "strained"
        row = regulation_store.transition_row(None, reading, "2026-10-10")
        assert await regulation_store.record_transition(pool, row)
        assert await regulation_store.record_transition(pool, row)          # idempotent
        async with pool.connection() as conn:
            rows = await (await conn.execute("SELECT * FROM temporal_self_event")).fetchall()
        assert len(rows) == 1 and rows[0]["source_kind"] == "arousal_transition" and rows[0]["label"] == "strained"
        assert rows[0]["payload_json"]["to"] == "strained"

    asyncio.run(_with_schema(scenario))


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_store_missing_tables_read_as_stale_not_zero():
    from app import regulation_store

    async def scenario(pool):
        inputs = await regulation_store.read_arousal_inputs(pool, datetime.now(timezone.utc), None,
                                                            gpu_queue_floor=2, gpu_sustain_sec=300.0)
        assert not inputs.juniper_turn_read_ok and not inputs.cabinet_read_ok
        assert inputs.gpu_state_age_sec is None and inputs.gpu_queue_depth is None
        assert not await regulation_store.record_transition(pool, {
            "event_id": "x", "day_id": "d", "occurred_at": datetime.now(timezone.utc), "source_ref": "r",
            "label": "idle", "payload": "{}"})

    asyncio.run(_with_schema(scenario))
