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
        self.fault = None
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
            gpu_state_age_sec=self.gpu_age, gpu_queue_depth=self.depth, gpu_queue_sustained_sec=self.sustained,
            gpu_fault=self.fault)

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


class Causes:
    """Fake ``record_cause``: what the regulate node asked to write (strain loop part 1)."""

    def __init__(self):
        self.rows: list[tuple] = []
        self.fail = False

    async def __call__(self, reading, day_id, joined, seed):
        if self.fail:
            return False
        self.rows.append((list(joined), seed, reading.since))
        return True


def test_strain_cause_rows_record_each_joining_cause_once():
    w, causes = World(), Causes()
    d = driver(w, record_cause=causes)
    w.depth, w.sustained = 4, 320.0
    tick(d)                                                      # onset: the level row carries GPU
    assert w.transitions[-1][1] == "strained" and causes.rows == []
    w.now += timedelta(seconds=120)
    w.reflex, w.thermal = "cabinet_hot", "hot"
    tick(d)                                                      # the cabinet joins mid-episode
    assert [r[:2] for r in causes.rows] == [(["cabinet_hot"], False)]
    assert causes.rows[0][2] == T0                               # same episode
    for _ in range(3):                                           # unchanged, or a cause dropping out
        w.now += timedelta(seconds=120)
        w.depth, w.sustained = 0, 0.0
        tick(d)
    assert len(causes.rows) == 1 and len(w.transitions) == 1


def test_strain_seed_row_when_strained_with_nothing_recorded_then_retry_on_failure():
    w = World()
    saver = InMemorySaver()
    w.depth, w.sustained = 4, 320.0
    tick(driver(w, saver))                                       # flag off: onset written, no cause state
    causes = Causes()
    causes.fail = True
    d = driver(w, saver, record_cause=causes)                    # deploy with the flag on, mid-episode
    w.now += timedelta(seconds=120)
    tick(d)
    assert causes.rows == [] and "strain_cause_write_failed" in w.projected[-1].warnings
    causes.fail = False
    w.now += timedelta(seconds=120)
    tick(d)                                                      # retried next step
    assert [r[:2] for r in causes.rows] == [(["gpu_queue_sustained"], True)] and causes.rows[0][2] == T0
    w.now += timedelta(seconds=120)
    tick(d)
    assert len(causes.rows) == 1


def test_strain_cause_rows_off_by_flag_and_not_when_calm():
    w = World()
    d = driver(w)                                                # record_cause=None: the flag is off
    w.reflex, w.thermal = "cabinet_hot", "hot"
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "strained"
    w2, causes = World(), Causes()
    d2 = driver(w2, record_cause=causes)
    for _ in range(3):
        tick(d2)
        w2.now += timedelta(seconds=120)
    assert causes.rows == []                                     # idle: nothing to record


def test_strain_keepalive_every_30_min_while_strained_and_none_after():
    w, causes = World(), Causes()
    d = driver(w, record_cause=causes)
    w.depth, w.sustained = 4, 320.0
    tick(d)                                                      # onset at T0
    for _ in range(15):                                          # 30 min of 2-min steps
        w.now += timedelta(seconds=120)
        tick(d)
    assert [r[:2] for r in causes.rows] == [([], False)]         # one keepalive, at T0 + 30 min
    for _ in range(15):
        w.now += timedelta(seconds=120)
        tick(d)
    assert len(causes.rows) == 2
    w.depth, w.sustained = 0, 0.0                                # strain clears after 10 clean min
    for _ in range(20):
        w.now += timedelta(seconds=120)
        tick(d)
    assert w.projected[-1].arousal.arousal_level != "strained"
    n = len(causes.rows)
    for _ in range(20):
        w.now += timedelta(seconds=120)
        tick(d)
    assert len(causes.rows) == n                                 # not strained: no keepalives


def test_strain_row_when_a_cause_leaves_no_sooner_than_10_min():
    w, causes = World(), Causes()
    d = driver(w, record_cause=causes)
    w.depth, w.sustained = 4, 320.0
    w.reflex, w.thermal = "cabinet_hot", "hot"
    tick(d)                                                      # onset: cabinet + GPU
    w.now += timedelta(seconds=120)
    w.reflex, w.thermal = None, "normal"                         # the cabinet cools at T0+2 min
    tick(d)
    assert causes.rows == []                                     # only 2 min since the onset row
    for _ in range(4):                                           # T0+4 .. T0+10 min
        w.now += timedelta(seconds=120)
        tick(d)
    assert [r[:2] for r in causes.rows] == [([], False)]         # the leave row, at T0+10 min
    assert w.projected[-1].arousal.arousal_level == "strained"
    for _ in range(4):                                           # unchanged: no more rows
        w.now += timedelta(seconds=120)
        tick(d)
    assert len(causes.rows) == 1


def test_a_flapping_cause_costs_at_most_one_row_per_10_min():
    w, causes = World(), Causes()
    d = driver(w, record_cause=causes)
    w.depth, w.sustained = 4, 320.0
    tick(d)                                                      # onset: GPU
    for i in range(30):                                          # 60 min: cabinet on/off every step
        w.now += timedelta(seconds=120)
        w.reflex, w.thermal = ("cabinet_hot", "hot") if i % 2 == 0 else (None, "normal")
        tick(d)
    assert causes.rows[0][0] == ["cabinet_hot"]                  # the first join is immediate
    assert len(causes.rows) <= 1 + 60 // 10


def test_first_steps_after_deploy_record_causes_now_within_10_min():
    """A checkpoint from before this key existed: episode causes and last row known, causes present
    at that row not. A row lands once the rate limit allows, not only at the 30-min keepalive."""
    from app.temporal_self_graph import STRAIN_CAUSE_CHANGE_MIN_SEC, _strain_causes_step
    from orion.schemas.regulation import ArousalReadingV1

    causes = Causes()
    deps = World().deps(record_cause=causes)
    state = {"strain_causes_seen": ["cabinet_hot", "gpu_queue_sustained"],
             "strain_row_at": T0.isoformat(), "day_id": "2026-10-10"}

    def step(minutes):
        r = ArousalReadingV1(arousal_level="strained", since=T0 - timedelta(days=1),
                             observed_at=T0 + timedelta(minutes=minutes), reasons=["gpu_queue_sustained"])
        return asyncio.run(_strain_causes_step(deps, r, state, False, []))

    assert step(STRAIN_CAUSE_CHANGE_MIN_SEC / 60 - 1)[2] is None and causes.rows == []   # too soon
    seen, row_at, now = step(STRAIN_CAUSE_CHANGE_MIN_SEC / 60)
    assert [r[:2] for r in causes.rows] == [([], False)] and now == ["gpu_queue_sustained"]
    assert seen == ["cabinet_hot", "gpu_queue_sustained"]


def test_cause_row_shape_matches_the_level_row():
    from app import regulation_store
    from orion.schemas.regulation import ArousalReadingV1

    r = ArousalReadingV1(arousal_level="strained", since=T0, observed_at=T0 + timedelta(minutes=5),
                         reasons=["cabinet_hot"])
    row = regulation_store.cause_row(r, "2026-10-10", ["cabinet_hot"], False)
    assert row["event_id"] == f"arousal_transition:{T0.isoformat()}:strained:at:{(T0 + timedelta(minutes=5)).isoformat()}"
    assert row["occurred_at"] == T0 + timedelta(minutes=5) and row["label"] == "strained"
    payload = json.loads(row["payload"])
    assert payload["from"] == payload["to"] == "strained" and payload["joined"] == ["cabinet_hot"]
    assert row["event_id"] != regulation_store.transition_row(None, r, "2026-10-10")["event_id"]


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
            if "gpu_pool_events" in sql:
                rows = []

            class R:
                async def fetchall(self_inner):
                    return rows
            return R()

    class Pool:
        def connection(self):
            return Conn()

    out = asyncio.run(regulation_store._gpu(Pool(), T0, floor=2, sustain_sec=300.0))
    assert out["gpu_queue_depth"] == 0 and out["gpu_state_age_sec"] == 0.0
    assert out["gpu_fault"] is None


class _GpuPool:
    """Fake pool for ``regulation_store._gpu``: snapshots and pool events by SQL text."""

    def __init__(self, snaps, events, fail_events=False):
        self.snaps, self.events, self.fail_events = snaps, events, fail_events
        self.event_params = []

    def connection(self):
        outer = self

        class Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def execute(self, sql, params=None):
                if "gpu_pool_events" in sql:
                    if outer.fail_events:
                        raise RuntimeError("relation gpu_pool_events does not exist")
                    outer.event_params.append(params)
                    rows = outer.events
                else:
                    rows = outer.snaps

                class R:
                    async def fetchall(self_inner):
                        return rows
                return R()
        return Conn()


def _jam(now, n=70, depth=12):
    return [{"host": "circe", "generated_at": now - timedelta(seconds=5 * i), "queue_depth": {"agent": depth}}
            for i in range(n)]


def test_store_reads_a_controller_fault_from_pool_events():
    from app import regulation_store

    events = [{"generated_at": T0 - timedelta(seconds=601), "event": "actuate_refused",
               "reason": "config_unloadable:ValidationError", "role": "agent-gpu2"}]
    pool = _GpuPool(_jam(T0), events)
    out = asyncio.run(regulation_store._gpu(pool, T0, floor=2, sustain_sec=300.0))
    assert out["gpu_fault"] == "config_unreadable:agent-gpu2"
    assert out["gpu_queue_depth"] == 12 and out["gpu_queue_sustained_sec"] >= 300.0
    assert pool.event_params[0]["since"] == T0 - timedelta(seconds=1800)
    assert set(pool.event_params[0]["events"]) == {"actuate_refused", "swap_failed", "swapped"}


def test_store_reads_the_shed_held_queue_and_selects_its_columns():
    """2026-10-20: the shed columns (sql-writer, manual_migration_regulation_gpu_shed_held_v1)
    reach the reducer, so the live heat-held queue is not GPU load. A json column may arrive as text."""
    from app import regulation_store
    from orion.regulation.arousal import classify_arousal

    held = {"memory_distill": 2, "agent": 9}
    snaps = [{"host": "circe", "generated_at": T0 - timedelta(seconds=5 * i), "queue_depth": dict(held),
              "shed_held_depth": json.dumps(held) if i % 2 else dict(held), "shed_reason": "cabinet_hot"}
             for i in range(70)]
    out = asyncio.run(regulation_store._gpu(_GpuPool(snaps, []), T0, floor=2, sustain_sec=300.0))
    assert out["gpu_queue_depth"] == 11 and out["gpu_queue_sustained_sec"] == 0.0
    assert (out["gpu_queue_shed_held"], out["gpu_shed_reason"]) == (11, "cabinet_hot")
    assert "shed_held_depth" in regulation_store.GPU_SNAPSHOTS_SQL and "shed_reason" in regulation_store.GPU_SNAPSHOTS_SQL
    inputs = ArousalInputsV1(observed_at=T0, juniper_turn_read_ok=True, minutes_since_juniper_turn=300.0,
                             cabinet_read_ok=True, cabinet_reflex=None, cabinet_thermal_state="elevated", **out)
    assert classify_arousal(None, inputs).reasons == ["gpu_shed:cabinet_hot"]


def test_store_reads_rows_saved_before_the_shed_columns_as_nothing_held():
    from app import regulation_store

    out = asyncio.run(regulation_store._gpu(_GpuPool(_jam(T0), []), T0, floor=2, sustain_sec=300.0))
    assert out["gpu_queue_shed_held"] == 0 and out["gpu_shed_reason"] is None
    assert out["gpu_queue_sustained_sec"] >= 300.0


def test_store_fault_read_failure_makes_gpu_stale_not_load():
    from app import regulation_store

    out = asyncio.run(regulation_store._gpu(_GpuPool(_jam(T0), [], fail_events=True), T0, floor=2,
                                            sustain_sec=300.0))
    assert out == {}


def test_fault_jam_reads_unknown_through_a_restart_then_real_load_strains():
    """Lifecycle: the fault is re-read from pool events every step, so a durable-runs restart in
    the middle of a jam still reads unknown (never strained), and the queue counts as load again
    only once the controller swapped."""
    w = World()
    saver = InMemorySaver()
    w.depth, w.sustained = 12, 3600.0
    w.fault = "config_unreadable:agent-gpu2"
    d = driver(w, saver)
    tick(d)
    a = w.projected[-1].arousal
    assert a.arousal_level == "unknown" and a.reasons[0] == "gpu_fault:config_unreadable:agent-gpu2"
    assert not a.strain_latched
    # Restart mid-jam: a new driver on the same checkpoint.
    w.now += timedelta(seconds=120)
    d2 = driver(w, saver)
    tick(d2)
    assert w.projected[-1].arousal.arousal_level == "unknown"
    # Restart after an outage longer than the carry gap: still unknown, never strained.
    w.now += timedelta(hours=2)
    tick(driver(w, saver))
    assert w.projected[-1].arousal.arousal_level == "unknown"
    # The controller swapped: the backlog it leaves is real work being drained.
    w.fault = None
    w.now += timedelta(seconds=120)
    tick(d2)
    assert w.projected[-1].arousal.arousal_level == "strained"
    assert "gpu_queue_sustained" in w.projected[-1].arousal.reasons


# --- store SQL on real Postgres -------------------------------------------------------------

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_temporal_self_event_v1.sql"
HISTORY_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_regulation_history.sql"
SHED_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_regulation_gpu_shed_held_v1.sql"


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
            for sql in (MIGRATION.read_text(), HISTORY_MIGRATION.read_text(), SHED_MIGRATION.read_text()):
                await conn.execute(sql, prepare=False)
            await conn.execute("""CREATE TABLE chat_history_log (id varchar PRIMARY KEY, source varchar, prompt text,
                                  client_meta jsonb, created_at timestamp DEFAULT now())""")
            await conn.execute("""CREATE TABLE orion_biometrics_summary (node varchar, timestamp varchar,
                                  measurements jsonb)""")
            await conn.execute("""CREATE TABLE gpu_pool_events (event_id varchar PRIMARY KEY,
                                  generated_at timestamptz NOT NULL, event varchar NOT NULL, reason varchar,
                                  role varchar)""")
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

        assert inputs.gpu_fault is None

        from orion.regulation.arousal import classify_arousal

        reading = classify_arousal(None, inputs)
        assert reading.arousal_level == "strained"

        # The same jam with the lane controller refusing: a fault, not load.
        async with pool.connection() as conn:
            await conn.execute("""INSERT INTO gpu_pool_events (event_id, generated_at, event, reason, role) VALUES
                ('r1', %(a)s, 'actuate_refused', 'config_unloadable:ValidationError', 'agent-gpu2'),
                ('r0', %(b)s, 'actuate_refused', 'config_unloadable:ValidationError', 'agent-gpu2'),
                ('old', %(c)s, 'swapped', 'demand', 'agent-gpu2'),
                ('g', %(a)s, 'granted', NULL, 'agent')""",
                               {"a": now - timedelta(seconds=30), "b": now - timedelta(seconds=630),
                                "c": now - timedelta(seconds=4000)})
        faulted = await regulation_store.read_arousal_inputs(pool, now, None, gpu_queue_floor=2,
                                                             gpu_sustain_sec=300.0)
        assert faulted.gpu_fault == "config_unreadable:agent-gpu2"
        assert classify_arousal(None, faulted.model_copy(update={"cabinet_reflex": None})).arousal_level == "unknown"
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


# --- R4 runtime immune sweep: the `immune` node ----------------------------------------------

from orion.regulation.immune import ImmuneSample  # noqa: E402


class ImmuneWorld(World):
    """A World whose field history holds one constant zombie (1.0 all day, rest [0, 0.5])."""

    def __init__(self):
        super().__init__()
        self.reads: list = []
        self.recorded: list = []
        self.fail_read: Exception | None = None
        self.hang = False
        self.record_ok = True

    async def read_immune_series(self, now):
        self.reads.append(now)
        if self.hang:
            await asyncio.sleep(10)
        if self.fail_read is not None:
            raise self.fail_read
        ts = [now - timedelta(minutes=i) for i in range(24 * 60, 0, -1)]
        zombie = [ImmuneSample(t, 1.0, updated_at=t, source=f"slice:{i}") for i, t in enumerate(ts)]
        calm = [ImmuneSample(t, 0.0, updated_at=t) for t in ts]
        sem = {"value_kind": "level", "sparsity": "per_tick", "rest_range": (0.0, 0.5)}
        return ({("node:z", "gauge"): zombie, ("node:c", "gauge"): calm},
                {("node:z", "gauge"): sem, ("node:c", "gauge"): dict(sem, rest_range=(0.0, 0.0))})

    async def record_immune(self, findings, day_id, now):
        self.recorded.append(([f.metric_id + ":" + f.kind for f in findings], day_id, now))
        return self.record_ok

    def deps(self, **kw):
        kw.setdefault("read_immune_series", self.read_immune_series)
        kw.setdefault("record_immune", self.record_immune)
        return super().deps(**kw)


def test_immune_sweep_runs_when_idle_records_findings_and_sets_the_count():
    w = ImmuneWorld()
    d = driver(w)
    out = tick(d)
    assert w.projected[-1].arousal.arousal_level == "idle"
    assert len(w.reads) == 1
    assert w.recorded == [(["node:z.gauge:constant"], "2026-10-10", T0)]
    assert d.latest.immune_findings_open == 1
    assert w.projected[-1].immune_findings_open == 1          # re-projected after the sweep
    assert out["immune"]["open"] == 1 and out["immune"]["findings"] == ["node:z.gauge:constant"]
    assert out["immune"]["coverage"] == {"judged": 2, "rest_unjudged": 0, "skipped": 0}
    assert out["immune"]["error"] is None
    assert d.health()["immune"]["open"] == 1


def test_immune_sweep_skipped_when_not_idle():
    w = ImmuneWorld()
    w.minutes = 5.0                                            # Juniper just spoke: engaged
    d = driver(w)
    tick(d)
    assert w.projected[-1].arousal.arousal_level == "engaged"
    assert w.reads == [] and d.latest.immune_findings_open == 0
    assert d.health()["immune"] == {"skipped": "arousal engaged"}     # visible, not silent


def test_immune_sweep_never_runs_with_arousal_disabled_and_says_so():
    w = ImmuneWorld()
    d = driver(w, arousal_enabled=False)
    tick(d)
    assert w.reads == [] and d.health()["immune"]["skipped"] == "arousal unknown"


def test_immune_sweep_runs_at_most_once_per_interval_and_count_carries():
    w = ImmuneWorld()
    d = driver(w)
    tick(d)
    for _ in range(5):
        w.now += timedelta(minutes=30)
        tick(d)
    assert len(w.reads) == 1                                   # 2.5 h later: still one sweep
    assert d.latest.immune_findings_open == 1                  # regulate carries the open count
    w.now = T0 + timedelta(hours=6, seconds=1)
    tick(d)
    assert len(w.reads) == 2


def test_immune_last_run_carries_over_midnight():
    w = ImmuneWorld()
    w.now = datetime(2026, 10, 11, 5, 0, tzinfo=timezone.utc)   # 23:00 Denver, 10-10
    d = driver(w)
    tick(d)
    w.now += timedelta(hours=2)                                  # 01:00 Denver, 10-11: new thread
    tick(d)
    assert len(w.reads) == 1 and d.latest.immune_findings_open == 1


def test_immune_sweep_failure_does_not_break_regulation():
    w = ImmuneWorld()
    w.fail_read = RuntimeError("relation substrate_field_state does not exist")
    d = driver(w)
    out = tick(d)
    assert w.projected[-1].arousal.arousal_level == "idle"
    assert d.latest is not None and d.latest.immune_findings_open == 0
    assert "immune_failed" in out["warnings"]
    assert "immune_failed" in d.latest.warnings and "immune_failed" in w.projected[-1].warnings
    assert out["immune"]["error"].startswith("RuntimeError")
    w.now += timedelta(seconds=120)
    tick(d)
    assert len(w.reads) == 1                                   # a broken read is not retried every step


def test_immune_sweep_timeout_does_not_break_regulation():
    w = ImmuneWorld()
    w.hang = True
    d = driver(w, immune_timeout_sec=0.05)
    out = tick(d)
    assert w.projected[-1].arousal.arousal_level == "idle"
    assert out["immune"]["error"].startswith("timeout")


def test_immune_write_failure_is_a_warning_and_still_counts():
    w = ImmuneWorld()
    w.record_ok = False
    d = driver(w)
    out = tick(d)
    assert "immune_write_failed" in out["warnings"] and d.latest.immune_findings_open == 1
    assert "immune_write_failed" in d.latest.warnings


def test_immune_disabled_never_reads():
    w = ImmuneWorld()
    d = driver(w, read_immune_series=None)
    tick(d)
    assert w.reads == [] and d.latest.immune_findings_open == 0


def test_redis_projection_omits_immune_count_for_older_readers():
    """Consumer-first: RegulationStateV1 is extra="forbid", so a reader on older code would reject
    the new field and read arousal as unknown. The Redis body leaves it out until they rebuild."""
    from orion.schemas.regulation import parse_regulation_state, regulation_state_redis_json

    w = ImmuneWorld()
    d = driver(w)
    tick(d)
    body = json.loads(regulation_state_redis_json(d.latest))
    assert "immune_findings_open" not in body
    assert parse_regulation_state(json.dumps(body)) is not None


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_store_immune_read_and_idempotent_write_on_postgres():
    from app import regulation_store
    from orion.regulation.immune import ImmuneFinding

    async def scenario(pool):
        async with pool.connection() as conn:
            await conn.execute(MIGRATION.read_text(), prepare=False)
            await conn.execute("""CREATE TABLE substrate_field_state (tick_id text PRIMARY KEY,
                                  generated_at timestamptz NOT NULL, field_json jsonb NOT NULL,
                                  created_at timestamptz NOT NULL DEFAULT now())""")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        async with pool.connection() as conn:
            # Two ticks per minute for 10 minutes: the read keeps the latest tick of each minute.
            for i in range(20):
                at = now - timedelta(seconds=30 * (i + 1))
                field = {"node_vectors": {"node:substrate.route": {"prediction_error": 0.5 - i * 0.01, "other": 9.0}},
                         "node_vector_updated_at": {"node:substrate.route": {"prediction_error": at.isoformat()}}}
                await conn.execute("INSERT INTO substrate_field_state (tick_id, generated_at, field_json) "
                                   "VALUES (%s, %s, %s)", (f"t{i}", at, json.dumps(field)))
        series = await regulation_store.read_immune_series(pool, now, ["prediction_error"], window_sec=3600)
        assert list(series) == [("node:substrate.route", "prediction_error")]
        samples = series[("node:substrate.route", "prediction_error")]
        assert len(samples) in (10, 11)                         # one per minute bucket
        assert all(s.updated_at is not None for s in samples)
        assert [s.t for s in samples] == sorted(s.t for s in samples)

        f = ImmuneFinding("node:substrate.route.prediction_error", "node:substrate.route", "prediction_error",
                          "decayed_to_zero", now - timedelta(hours=6), now - timedelta(hours=1), {"ratio": 0.92})
        row = regulation_store.immune_finding_row(f, "2026-10-10", now)
        assert await regulation_store.record_immune_findings(pool, [row])
        later = regulation_store.immune_finding_row(
            ImmuneFinding(f.metric_id, f.node, f.channel, f.kind, f.window_start, now, {"ratio": 0.92}),
            "2026-10-10", now + timedelta(hours=6))
        assert await regulation_store.record_immune_findings(pool, [later])   # same day: one row, refreshed
        async with pool.connection() as conn:
            rows = await (await conn.execute("SELECT * FROM temporal_self_event")).fetchall()
        assert len(rows) == 1 and rows[0]["source_kind"] == "immune_finding"
        assert rows[0]["label"] == "decayed_to_zero" and rows[0]["verdict"] == "open"
        assert rows[0]["payload_json"]["window_end"] == now.isoformat()

    asyncio.run(_with_schema(scenario))
