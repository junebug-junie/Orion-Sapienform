"""Body economy Patch A3: the `physiology` node of temporal_self.update, and its store helpers.

The graph runs on an in-memory saver with fake reads (the regulate test's World). The store SQL
runs against an isolated Postgres schema when ORION_ADMISSION_TEST_DSN is set (CI provides one).
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app import physiology_store  # noqa: E402
from app.temporal_self_driver import TemporalSelfDriver  # noqa: E402
from app.temporal_self_graph import NODES, TemporalSelfDeps  # noqa: E402
from orion.physiology.body import BodyInputs, EnergyInput, JuniperInput, LaneInput  # noqa: E402
from orion.schemas.physiology import PhysiologyStateV1  # noqa: E402
from orion.schemas.regulation import ArousalInputsV1  # noqa: E402

T0 = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)


class World:
    def __init__(self):
        self.now = T0
        self.temp = 29.0
        self.cabinet_ok = True
        self.regulation_projected = 0
        self.states: list[PhysiologyStateV1] = []
        self.heat_seen: list = []
        self.fail_read = False
        self.hang = False

    async def read_inputs(self, now, prev, last_turn_at):
        return ArousalInputsV1(observed_at=now, juniper_turn_read_ok=True, minutes_since_juniper_turn=300.0,
                               cabinet_read_ok=self.cabinet_ok, cabinet_thermal_state="normal",
                               cabinet_temp_c=self.temp, gpu_state_age_sec=2.0, gpu_queue_depth=0,
                               gpu_queue_sustained_sec=0.0)

    async def read_drives(self, now):
        return [], []

    async def project(self, model):
        self.regulation_projected += 1

    async def record(self, prev, new, day_id):
        return True

    async def read_body(self, now, heat):
        self.heat_seen.append(heat)
        if self.fail_read:
            raise ConnectionError("db down")
        if self.hang:
            await asyncio.sleep(10)
        return BodyInputs(heat=heat, energy=EnergyInput(read_ok=False), lanes=LaneInput(True, [], {"fast": 4}, newest_event_at=now),
                          juniper=JuniperInput(True, 0, 0, now), claude=()), ["lane_capacity:test"]

    async def project_physiology(self, model):
        self.states.append(model)

    def driver(self, **kw):
        deps = TemporalSelfDeps(read_inputs=self.read_inputs, read_drives=self.read_drives, project=self.project,
                                record_transition=self.record, now=lambda: self.now,
                                read_body=self.read_body, project_physiology=self.project_physiology, **kw)
        return TemporalSelfDriver(checkpointer=InMemorySaver(), deps=deps, tick_sec=120.0)


def tick(d, kind="tick"):
    return asyncio.run(d.step({"event_id": f"{kind}:{d._deps.now().isoformat()}", "kind": kind}))


def test_node_sits_between_regulate_and_immune():
    assert NODES.index("regulate") < NODES.index("physiology") < NODES.index("immune")


def test_each_tick_writes_one_state_with_heat_from_the_regulate_inputs():
    w = World()
    d = w.driver()
    out = tick(d)
    assert len(w.states) == 1
    st = w.states[0]
    heat = next(c for c in st.constraints if c.constraint == "heat")
    assert heat.level == 29.0 and heat.no_reading is None
    assert st.warnings == ["lane_capacity:test"]
    energy = next(c for c in st.constraints if c.constraint == "energy")
    assert energy.level is None and energy.no_reading.startswith("read_failed:")
    assert out["physiology"]["tick_id"] == st.tick_id
    assert "energy:read_failed:orion_energy_hourly" in out["physiology"]["no_reading"]
    assert d.health()["physiology"]["tick_id"] == st.tick_id
    w.now += timedelta(seconds=120)
    tick(d)
    assert len(w.states) == 2 and w.states[1].generated_at == w.now


def test_steps_inside_the_min_interval_are_skipped():
    w = World()
    d = w.driver()
    tick(d)
    w.now += timedelta(seconds=30)
    asyncio.run(d.step({"event_id": "chat:1", "kind": "chat_turn", "at": w.now.isoformat(), "juniper": True,
                        "correlation_id": "1"}))
    assert len(w.states) == 1 and w.regulation_projected == 2   # regulation still stepped


def test_cabinet_read_failure_reaches_the_body_as_no_reading():
    w = World()
    w.cabinet_ok = False
    d = w.driver()
    tick(d)
    heat = next(c for c in w.states[0].constraints if c.constraint == "heat")
    assert heat.level is None and heat.no_reading == "read_failed:cabinet"


def test_failure_is_a_warning_and_regulation_still_projects_and_retries_next_tick():
    w = World()
    w.fail_read = True
    d = w.driver()
    out = tick(d)
    assert w.states == [] and w.regulation_projected == 1
    assert "physiology_failed" in out["warnings"] and out["physiology"]["error"].startswith("ConnectionError")
    w.fail_read = False
    w.now += timedelta(seconds=120)
    tick(d)
    assert len(w.states) == 1


def test_timeout_is_bounded():
    w = World()
    w.hang = True
    d = w.driver(physiology_timeout_sec=0.05)
    out = tick(d)
    assert w.states == [] and out["physiology"]["error"].startswith("timeout")


def test_disabled_writes_nothing():
    w = World()
    deps = TemporalSelfDeps(read_inputs=w.read_inputs, read_drives=w.read_drives, project=w.project,
                            record_transition=w.record, now=lambda: w.now)
    d = TemporalSelfDriver(checkpointer=InMemorySaver(), deps=deps, tick_sec=120.0)
    tick(d)
    assert w.states == [] and w.regulation_projected == 1


# --- store helpers ------------------------------------------------------------------------------

def test_capacity_from_pool_state():
    payload = {"roles": [{"role": "fast", "status": "confirmed", "slots": 4},
                         {"role": "diffusion", "status": "static", "slots": 1},
                         {"role": "agent-gpu2", "status": "down", "slots": 2}]}
    assert physiology_store.capacity_from_pool_state(payload) == {"fast": 4, "diffusion": 1, "agent-gpu2": 0}
    assert physiology_store.capacity_from_pool_state(None) is None


def test_month_start_is_local():
    tz = ZoneInfo("America/Denver")
    late = datetime(2026, 11, 1, 3, 0, tzinfo=timezone.utc)      # still Oct 31 in Denver
    assert physiology_store.month_start_utc(late, tz) == datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)


def test_claude_limit_cache_keeps_latest_per_kind_and_rejects_malformed():
    cache = physiology_store.ClaudeLimitCache()
    base = {"observed_at": T0.isoformat(), "kind": "session_limit", "window_hours": 5.0,
            "window_start": (T0 - timedelta(hours=5)).isoformat(), "window_end": T0.isoformat(),
            "state": "clear", "observed": True, "event_count": 0, "observed_message_count": 3,
            "scanned_file_count": 1}
    assert cache.offer(base, T0)
    assert cache.offer({**base, "state": "limited", "event_count": 2}, T0 + timedelta(seconds=5))
    assert cache.offer({**base, "kind": "weekly_limit"}, T0)
    assert not cache.offer({"junk": 1}, T0)
    snap = cache.snapshot()
    assert [(o.kind, o.state, o.event_count) for o in snap] == [("session_limit", "limited", 2),
                                                                ("weekly_limit", "clear", 0)]


# --- store SQL on real Postgres -------------------------------------------------------------------

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")


async def _with_schema(scenario):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = "physiology_test_" + uuid4().hex
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
def test_store_reads_on_postgres():
    tz = ZoneInfo("America/Denver")
    now = datetime.now(timezone.utc)

    async def scenario(pool):
        async with pool.connection() as conn:
            await conn.execute("""CREATE TABLE orion_energy_hourly (hour_start timestamptz, source varchar,
                                  kwh double precision, covered_sec double precision, sample_count int,
                                  updated_at timestamptz DEFAULT now())""")
            await conn.execute("""CREATE TABLE energy_cost_accrued (usage_point_id varchar,
                                  interval_start timestamptz, energy_kwh double precision,
                                  marginal_usd_per_kwh double precision, computed_at timestamptz)""")
            await conn.execute("""CREATE TABLE gpu_pool_events (event_id varchar PRIMARY KEY, generated_at timestamptz,
                                  event varchar, lease_id varchar, role varchar, held_ms double precision)""")
            await conn.execute("""CREATE TABLE gpu_pool_leases (lease_id text PRIMARY KEY, status text NOT NULL,
                                  role text, granted_at timestamptz, hold_lease_id text)""")
            await conn.execute("""CREATE TABLE endogenous_outreach_decisions (decision_id text PRIMARY KEY,
                                  decided_at timestamptz, outreach boolean, reason text, result_json jsonb)""")
            hour = now.replace(minute=0, second=0, microsecond=0)
            for src, kwh in (("athena", 0.3), ("cooling", 0.5)):
                await conn.execute("INSERT INTO orion_energy_hourly VALUES (%s, %s, %s, 3600, 10)", (hour, src, kwh))
            # Same interval priced by two tariff versions: the newest computed_at wins.
            await conn.execute("""INSERT INTO energy_cost_accrued VALUES
                ('01', %(h)s, 1.0, 0.50, %(old)s), ('01', %(h)s, 1.0, 0.10, %(new)s)""",
                               {"h": hour - timedelta(days=1), "old": now - timedelta(days=2), "new": now})
            await conn.execute("""INSERT INTO gpu_pool_events VALUES
                ('e1', %(a)s, 'released', 'L1', 'fast', 60000),
                ('e2', %(b)s, 'granted', 'L2', 'agent', NULL),
                ('e3', %(b)s, 'granted', 'L3', 'fast', NULL),
                ('e4', %(a)s, 'released', 'L3', 'fast', 1000),
                ('e5', %(a)s, 'released', 'C1', 'agent', 5000),
                ('e6', %(old)s, 'released', 'OLD', 'fast', 1000)""",
                               {"a": now - timedelta(minutes=5), "b": now - timedelta(minutes=10),
                                "old": now - timedelta(hours=3)})
            # The pool's own projection: L2 still open; LOST's close event never landed but the pool
            # says released (must NOT read as open); C1 is a child call inside hold L2.
            await conn.execute("""INSERT INTO gpu_pool_leases VALUES
                ('L1', 'released', 'fast', %(b)s, NULL), ('L2', 'granted', 'agent', %(b)s, NULL),
                ('L3', 'released', 'fast', %(b)s, NULL), ('LOST', 'released', 'metacog', %(b)s, NULL),
                ('C1', 'released', 'agent', %(b)s, 'L2')""", {"b": now - timedelta(minutes=10)})
            await conn.execute("""INSERT INTO endogenous_outreach_decisions VALUES
                ('d1', %(t)s, true, 'sent', '{"source": "tension"}'),
                ('d2', %(t)s, true, 'sent', '{"source": "curiosity_outreach"}'),
                ('d3', %(t)s, false, 'cooldown', '{}')""", {"t": now - timedelta(minutes=1)})

        energy = await physiology_store.read_energy(pool, now, tz)
        assert energy.read_ok and len(energy.hours) == 1 and abs(energy.hours[0].kwh - 0.8) < 1e-9
        assert [p.marginal_usd_per_kwh for p in energy.prices] == [0.10]
        lanes = await physiology_store.read_lanes(pool, now, 3600.0, {"roles": [
            {"role": "fast", "status": "confirmed", "slots": 4}]})
        assert lanes.events_read_ok and lanes.capacity == {"fast": 4}
        assert lanes.newest_event_at is not None and abs((lanes.newest_event_at - (now - timedelta(minutes=5))).total_seconds()) < 1
        # Closes in the window (with the child's hold), and open leases from the pool's projection.
        assert sorted((e.lease_id, e.event, e.hold_lease_id) for e in lanes.events) == [
            ("C1", "released", "L2"), ("L1", "released", None), ("L2", "granted", None), ("L3", "released", None)]
        jun = await physiology_store.read_juniper(pool, now, tz)
        # Local-date bucketing: the 1-minute-old rows are "today" unless local midnight just passed.
        if (now - timedelta(minutes=1)).astimezone(tz).date() == now.astimezone(tz).date():
            assert jun.read_ok and jun.sent_today == 2 and jun.cap_counted_today == 1
        assert jun.newest_decision_at is not None

    asyncio.run(_with_schema(scenario))


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_store_missing_tables_read_failed_not_zero():
    tz = ZoneInfo("America/Denver")
    now = datetime.now(timezone.utc)

    async def scenario(pool):
        assert not (await physiology_store.read_energy(pool, now, tz)).read_ok
        assert not (await physiology_store.read_lanes(pool, now, 3600.0, None)).events_read_ok
        assert not (await physiology_store.read_juniper(pool, now, tz)).read_ok

    asyncio.run(_with_schema(scenario))
