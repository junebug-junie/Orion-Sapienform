"""temporal_self.update / self_monitor node (self-monitor organ patch 2: host and publish).

The graph and driver run on an in-memory saver with fake reads and writes. The store's SQL runs against
an isolated Postgres schema when ORION_ADMISSION_TEST_DSN is set (CI provides one).
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

from app.temporal_self_driver import CARRIED_KEYS, TemporalSelfDriver  # noqa: E402
from app.temporal_self_graph import NODES, TemporalSelfDeps  # noqa: E402
from orion.schemas.regulation import ArousalInputsV1  # noqa: E402
from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1  # noqa: E402
from orion.self_monitor.record import HOLD  # noqa: E402
from orion.self_monitor.view import parse_state  # noqa: E402
from orion.temporal_self.sources import faculty_status_changed  # noqa: E402

T0 = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)   # 11:00 America/Denver
HOUR = timedelta(hours=1)


def rec(faculty: str, candidate: str, *, now: datetime, sources=("grammar_acts",)) -> FacultyRecordV1:
    return FacultyRecordV1(
        faculty=faculty, sources=list(sources), gradeability="graded", window_start=now - timedelta(days=7),
        window_end=now, n_bets=20, n_graded=14, n_failed_to_execute=6, n_expired=0, weight_graded=10.0,
        candidate_status=candidate, status=candidate, candidate_held=1, reasons=[f"{candidate} because"],
    )


class World:
    """Fake reads and writes with a movable clock. ``candidates``: faculty -> candidate status."""

    def __init__(self):
        self.now = T0
        self.candidates: dict[str, str] = {"act.make.walkway_grade": "reliable"}
        self.source_errors: list[str] = []
        self.collect_calls = 0
        self.collect_delay = 0.0
        self.collect_error: Exception | None = None
        self.published: list[str] = []
        self.history: list[SelfMonitorStateV1] = []
        self.events: list[dict] = []
        self.fail_events = False
        self.fail_publish = False

    async def read_inputs(self, now, prev, last_turn_at):
        return ArousalInputsV1(observed_at=now, juniper_turn_read_ok=True, minutes_since_juniper_turn=300.0,
                               cabinet_read_ok=True, cabinet_thermal_state="normal", gpu_state_age_sec=2.0,
                               gpu_queue_depth=0, gpu_queue_sustained_sec=0.0)

    async def read_drives(self, now):
        return [], []

    async def project(self, model):
        return None

    async def record_transition(self, prev, new, day_id):
        return True

    async def collect(self, now):
        self.collect_calls += 1
        if self.collect_delay:
            await asyncio.sleep(self.collect_delay)
        if self.collect_error is not None:
            raise self.collect_error
        return SelfMonitorStateV1(computed_at=now, window_days=7.0,
                                  faculties=[rec(f, c, now=now) for f, c in sorted(self.candidates.items())],
                                  source_errors=list(self.source_errors))

    async def publish(self, state):
        if self.fail_publish:
            raise ConnectionError("redis down")
        self.published.append(state.model_dump_json())

    async def record_history(self, state):
        self.history.append(state)

    async def record_events(self, rows):
        if self.fail_events:
            return False
        self.events.extend(rows)
        return True

    def deps(self, **kw):
        base = dict(read_inputs=self.read_inputs, read_drives=self.read_drives, project=self.project,
                    record_transition=self.record_transition, now=lambda: self.now,
                    collect_self_monitor=self.collect, publish_self_monitor=self.publish,
                    record_self_monitor_history=self.record_history, record_faculty_events=self.record_events)
        base.update(kw)
        return TemporalSelfDeps(**base)


def driver(world: World, **kw) -> TemporalSelfDriver:
    return TemporalSelfDriver(checkpointer=InMemorySaver(), deps=world.deps(**kw), tick_sec=120.0)


def tick(d: TemporalSelfDriver) -> dict:
    return asyncio.run(d.step({"event_id": f"tick:{d._deps.now().isoformat()}", "kind": "tick"}))


def changes(world: World) -> list[dict]:
    return [json.loads(r["payload"]) for r in world.events if not json.loads(r["payload"])["keepalive"]]


def test_node_runs_after_immune_and_before_chronicle_and_is_carried_over_midnight():
    assert NODES.index("immune") < NODES.index("self_monitor") < NODES.index("chronicle")
    assert {"self_monitor", "self_monitor_at"} <= set(CARRIED_KEYS)


def test_interval_gates_the_evaluation_to_once_an_hour():
    w = World()
    d = driver(w)
    tick(d)
    w.now += timedelta(minutes=2)
    tick(d)
    w.now += timedelta(minutes=50)
    tick(d)
    assert w.collect_calls == 1 and len(w.published) == 1 and len(w.history) == 1
    w.now = T0 + HOUR
    tick(d)
    assert w.collect_calls == 2 and len(w.history) == 2


def test_published_payload_validates_and_history_gets_the_same_state():
    w = World()
    d = driver(w)
    out = tick(d)
    state = parse_state(w.published[0])
    assert state is not None and state.kind == "self_monitor.state.v1"
    assert [f.faculty for f in state.faculties] == ["act.make.walkway_grade"]
    assert state.faculties[0].status == "reliable" and state.faculties[0].status_since == T0
    assert w.history[0].model_dump_json() == w.published[0]
    summary = out["self_monitor"]["summary"]
    assert summary["by_status"] == {"reliable": 1} and summary["error"] is None
    assert d.health()["self_monitor"]["faculties"] == 1


def test_hysteresis_is_carried_across_steps_and_a_held_change_writes_exactly_one_event():
    w = World()
    d = driver(w)
    tick(d)                                  # first evaluation adopts reliable: not notable, no row
    assert w.events == []
    w.candidates["act.make.walkway_grade"] = "broken"
    for k in range(1, HOLD):
        w.now = T0 + k * HOUR
        tick(d)
        assert parse_state(w.published[-1]).faculties[0].status == "reliable"   # not held yet
    assert w.events == []
    w.now = T0 + HOLD * HOUR
    tick(d)
    held = parse_state(w.published[-1]).faculties[0]
    assert held.status == "broken" and held.status_since == w.now
    assert len(w.events) == 1
    row = w.events[0]
    payload = json.loads(row["payload"])
    assert payload["from"] == "reliable" and payload["to"] == "broken" and payload["n_failed_to_execute"] == 6
    assert row["day_id"] == "2026-11-03" and row["subject_ref"] == "faculty:act.make.walkway_grade"
    # No change: no row, for every evaluation inside the keepalive period.
    for k in range(1, 4):
        w.now = T0 + (HOLD + k) * HOUR
        tick(d)
    assert len(w.events) == 1
    # The row reads back through the chronology's own adapter.
    ev = faculty_status_changed({"event_id": row["event_id"], "occurred_at": row["occurred_at"],
                                 "payload_json": row["payload"]})
    assert ev is not None and ev.verdict == "broken" and ev.event_id == row["event_id"]


def test_a_notable_status_on_first_sight_writes_one_row_with_no_from():
    w = World()
    w.candidates = {"turn.self.steps": "miscalibrated", "act.ok": "reliable"}
    tick(driver(w))
    assert [(c["faculty"], c["from"], c["to"]) for c in changes(w)] == [("turn.self.steps", None, "miscalibrated")]


def test_a_failed_event_write_is_retried_next_evaluation_with_the_same_from():
    w = World()
    d = driver(w)
    tick(d)
    w.candidates["act.make.walkway_grade"] = "broken"
    w.fail_events = True
    for k in range(1, HOLD + 1):
        w.now = T0 + k * HOUR
        out = tick(d)
    assert w.events == [] and "self_monitor_event_write_failed" in out["warnings"]
    w.fail_events = False
    w.now += HOUR
    tick(d)
    assert [(c["from"], c["to"]) for c in changes(w)] == [("reliable", "broken")]
    # Stamped at the retry, past the chronicle's watermark, so it still folds (review finding).
    assert w.events[0]["occurred_at"] == w.now


def test_keepalive_every_12_hours_while_a_notable_status_holds():
    w = World()
    w.candidates = {"act.prime.predict": "unreliable"}
    d = driver(w)
    tick(d)
    for k in range(1, 13):
        w.now = T0 + k * HOUR
        tick(d)
    keepalives = [r for r in w.events if json.loads(r["payload"])["keepalive"]]
    assert len(changes(w)) == 1 and len(keepalives) == 1
    assert keepalives[0]["event_id"].endswith(f":at:{(T0 + 12 * HOUR).isoformat()}")


def test_a_dropped_notable_faculty_closes_with_one_row_unless_its_source_failed():
    w = World()
    w.candidates = {"act.prime.predict": "unreliable", "act.ok": "reliable"}
    d = driver(w)
    tick(d)
    w.candidates = {"act.ok": "reliable"}
    w.source_errors = ["grammar_acts: OperationalError: timeout"]
    w.now += HOUR
    tick(d)
    assert len(changes(w)) == 1                    # unread is not gone: nothing written
    w.source_errors = []
    w.now += HOUR
    out = tick(d)
    dropped = changes(w)[-1]
    assert dropped["to"] == "unknown" and dropped["dropped"] and dropped["from"] == "unreliable"
    assert "act.prime.predict" not in out["self_monitor"]["carry"]


def test_timeout_is_a_warning_and_leaves_regulation_and_the_carry_alone():
    w = World()
    d = driver(w, self_monitor_timeout_sec=0.05)
    tick(d)
    carry = d.health()["self_monitor"]
    w.collect_delay = 0.5
    w.now += HOUR
    out = tick(d)
    assert "self_monitor_failed" in out["warnings"]
    assert out["self_monitor"]["summary"]["error"].startswith("timeout")
    assert out["regulation"]["arousal"]["arousal_level"] == "idle"
    assert out["self_monitor"]["carry"]["act.make.walkway_grade"]["status"] == "reliable"
    assert out["self_monitor_at"] == w.now.isoformat()     # retried next interval, not next tick
    assert carry["faculties"] == 1
    w.collect_delay = 0.0
    w.now += timedelta(minutes=2)
    tick(d)
    assert w.collect_calls == 2


def test_a_failed_read_or_publish_never_crashes_the_step():
    w = World()
    w.collect_error = RuntimeError("pool closed")
    d = driver(w)
    out = tick(d)
    assert "self_monitor_failed" in out["warnings"] and out["self_monitor"]["summary"]["error"].startswith("RuntimeError")
    w.collect_error = None
    w.fail_publish = True
    w.now += HOUR
    out = tick(d)
    assert "self_monitor_publish_failed" in out["warnings"] and len(w.history) == 1


def test_status_since_is_byte_identical_while_a_status_is_held():
    """Outreach (PR #3061) dedupes on status_since: it must not move while the status holds."""
    w = World()
    d = driver(w)
    tick(d)
    first = json.loads(w.published[0])["faculties"][0]["status_since"]
    w.candidates["act.make.walkway_grade"] = "broken"   # a candidate that has not held yet
    for k in range(1, HOLD):
        w.now = T0 + k * HOUR
        tick(d)
    w.candidates["act.make.walkway_grade"] = "reliable"
    w.now += HOUR
    tick(d)
    assert [json.loads(p)["faculties"][0]["status_since"] for p in w.published] == [first] * len(w.published)


def test_the_hysteresis_carry_survives_a_restart():
    """A restarted durable-runs reads the carry from the checkpoint, so a new candidate still has to
    hold 3 evaluations: advance(None, ...) would adopt it at once (PR #3061's concern)."""
    w = World()
    saver = InMemorySaver()
    a = TemporalSelfDriver(checkpointer=saver, deps=w.deps(), tick_sec=120.0)
    tick(a)
    w.candidates["act.make.walkway_grade"] = "broken"
    w.now += HOUR
    tick(a)
    b = TemporalSelfDriver(checkpointer=saver, deps=w.deps(), tick_sec=120.0)   # the restart
    w.now += HOUR
    out = tick(b)
    held = parse_state(w.published[-1]).faculties[0]
    assert held.status == "reliable" and held.status_since == T0 and held.candidate_held == 2
    assert out["self_monitor"]["carry"]["act.make.walkway_grade"]["candidate_held"] == 2
    w.now += HOUR
    tick(b)
    assert parse_state(w.published[-1]).faculties[0].status == "broken" and len(changes(w)) == 1


def test_disabled_runs_nothing():
    w = World()
    out = tick(driver(w, collect_self_monitor=None))
    assert w.collect_calls == 0 and "self_monitor" not in out and "self_monitor_failed" not in out["warnings"]


def test_hysteresis_carry_crosses_midnight():
    w = World()
    w.now = datetime(2026, 11, 4, 6, 30, tzinfo=timezone.utc)   # 23:30 Denver, 11-03
    d = driver(w)
    tick(d)
    w.candidates["act.make.walkway_grade"] = "broken"
    w.now += HOUR                                               # 00:30, 11-04: a new day thread
    out = tick(d)
    assert out["day_id"] == "2026-11-04"
    assert out["self_monitor"]["carry"]["act.make.walkway_grade"]["candidate_held"] == 1
    assert parse_state(w.published[-1]).faculties[0].status == "reliable"


# --- store SQL on real Postgres -------------------------------------------------------------

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
EVENT_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_temporal_self_event_v1.sql"
HISTORY_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_self_monitor_history_v1.sql"


async def _with_schema(scenario):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = "self_monitor_test_" + uuid4().hex
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
def test_store_reads_sources_writes_history_and_events_on_postgres():
    from app import self_monitor_store
    from orion.self_monitor.collect import build_state
    from orion.self_monitor.host import FacultyChange, event_row

    async def scenario(pool):
        async with pool.connection() as conn:
            for sql in (EVENT_MIGRATION.read_text(), HISTORY_MIGRATION.read_text()):
                await conn.execute(sql, prepare=False)
            # One real source table; the other four are missing (named in source_errors, never silent).
            await conn.execute("""CREATE TABLE prediction_forecast (forecast_id text PRIMARY KEY, series_id text,
                                  role text, issued_at timestamptz, scored_at timestamptz, in_80 boolean)""")
            now = datetime.now(timezone.utc)
            for i in range(12):
                await conn.execute(
                    "INSERT INTO prediction_forecast VALUES (%s, 'athena.cabinet_temp_c', 'champion', %s, %s, %s)",
                    (f"f{i}", now - timedelta(hours=i + 1), now - timedelta(hours=i), i % 5 != 0))
        now = datetime.now(timezone.utc)
        results = await self_monitor_store.read_sources(pool, now, 7.0)
        state = build_state(results, now=now, window_days=7.0)
        faculties = {f.faculty: f for f in state.faculties}
        assert faculties["forecast.athena.cabinet_temp_c"].n_bets == 12
        assert len(state.source_errors) == 4 and all("UndefinedTable" in e for e in state.source_errors)

        old = state.model_copy(update={"computed_at": now - timedelta(days=40)})
        await self_monitor_store.record_history(pool, old, retention_days=30)
        await self_monitor_store.record_history(pool, state, retention_days=30)
        await self_monitor_store.record_history(pool, state, retention_days=30)   # same instant: one row
        change = FacultyChange("act.prime.predict", "reliable", "unreliable", now)
        row = event_row(change, day_id="2026-11-03", now=now)
        assert await self_monitor_store.record_events(pool, [row])
        assert await self_monitor_store.record_events(pool, [row])                # idempotent on its id
        async with pool.connection() as conn:
            hist = await (await conn.execute(
                "SELECT computed_at, state_json->>'kind' AS kind FROM substrate_self_monitor_history")).fetchall()
            events = await (await conn.execute(
                "SELECT event_id, source_kind, source_table, label, verdict, payload_json FROM temporal_self_event"
            )).fetchall()
        assert len(hist) == 1 and hist[0]["kind"] == "self_monitor.state.v1"     # the 40-day-old row was pruned
        assert len(events) == 1
        e = events[0]
        assert (e["source_kind"], e["source_table"], e["label"], e["verdict"]) == (
            "faculty_status_changed", "self_monitor", "unreliable", "unreliable")
        assert e["payload_json"]["faculty"] == "act.prime.predict"

    asyncio.run(_with_schema(scenario))
