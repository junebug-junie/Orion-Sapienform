"""GET /physiology/report (PhysiologyReportV1): every section shaped from rows, every empty
section explained in ``missing``, one failing read never failing the report, the route's 503 /
422 / cache, and the SQL on a real Postgres when ORION_ADMISSION_TEST_DSN is set (CI)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app import physiology_report as pr  # noqa: E402
from orion.schemas.physiology_report import REPORT_SECTIONS, PhysiologyReportV1  # noqa: E402

NOW = datetime(2026, 10, 29, 20, 0, tzinfo=timezone.utc)
HEAT_CFG = str(REPO_ROOT / "config/physiology/heat_sensitivity.yaml")
POOL_CFG = str(REPO_ROOT / "config/gpu_pool.yaml")
DSN = os.environ.get("ORION_ADMISSION_TEST_DSN")

ENERGY_WHY = ("the envelope is spent by what stays switched on (not by what I do); month to date $182.10; "
              "forecast $196.92 by month end at $0.256/h; the always-on floor alone ($0.203/h) reaches "
              "$193.85 vs envelope $100")
HEAT_WHY = "room over the line without Orion's load (not my load); cabinet 34.8 C; Orion's share 2.43 C"
LATEST = {"generated_at": NOW, "constraints": [
    {"constraint": "heat", "unit": "c", "level": 34.77, "budget": 31.0,
     "price": {"value": 0.51, "bind_state": "exogenous_bind", "why": HEAT_WHY, "use": 2.43}},
    {"constraint": "energy", "unit": "usd", "level": 182.1, "budget": 100.0,
     "detail": {"latest_rate_usd_per_kwh": 0.1254564},
     "price": {"value": 1.0, "bind_state": "exogenous_bind", "why": ENERGY_WHY, "use": 196.92}},
    {"constraint": "lane:agent", "unit": "slot_s", "level": 3599.0, "budget": 3600.0,
     "price": {"value": 0.3, "bind_state": "slack", "why": "3600 lease-s asked of 1 slot(s) in 60 min"}},
    {"constraint": "claude", "unit": "limit_events", "level": None, "budget": 0.0, "no_reading": "no_observation"},
]}
H0 = NOW - timedelta(hours=2)


@dataclass
class Weight:
    @dataclass
    class Ev:
        n: int
        mean_rate: float | None
        engagement_rate: float | None = None
        engagement_n: int = 0
        use_rate: float | None = None
        use_n: int = 0

    evidence: "Weight.Ev"
    basis: str = "evidence"


def full_rows() -> dict:
    return {
        pr.LIMITS_LATEST_SQL: [LATEST],
        pr.LIMITS_HOURLY_SQL: [
            {"h": H0, "constraint_name": "heat", "level": 34.5, "price": 0.5, "bind_state": "exogenous_bind"},
            {"h": H0, "constraint_name": "energy", "level": 182.0, "price": 1.0, "bind_state": "exogenous_bind"},
            {"h": H0, "constraint_name": "lane:agent", "level": 3500.0, "price": 0.2, "bind_state": "slack"}],
        pr.LIMITS_FIRST_SQL: [{"first_at": datetime(2026, 10, 23, 6, 22, tzinfo=timezone.utc)}],
        pr.WANTS_HOURLY_SQL: [
            {"h": H0, "family": "hub.outreach", "outcome": "deferred", "n": 30, "tier": 2},
            {"h": H0, "family": "hub.outreach", "outcome": "funded", "n": 1, "tier": 2},
            {"h": H0, "family": "actions.journal", "outcome": "funded", "n": 8, "tier": 1}],
        pr.WANTS_FIRST_SQL: [{"first_at": datetime(2026, 10, 27, 16, 46, tzinfo=timezone.utc)}],
        pr.RHYTHM_SQL: [{"day": "2026-10-29", "hour": 4, "heavy_s": 3000.0, "total_s": 3600.0},
                        {"day": "2026-10-29", "hour": 13, "heavy_s": 0.0, "total_s": 900.0}],
        pr.ENERGY_DAILY_SQL: [{"d": "2026-10-28", "kwh": 46.4}, {"d": "2026-10-29", "kwh": 25.7}],
        pr.RUN_COST_SQL: [{"workload_kind": "reverie_diffusion", "n": 20, "kwh": 0.000363, "usd": 4.57e-05}],
        pr.ACT_COST_SQL: [{"actor": "hub.outreach", "wh_by_host": {"athena": 1.5, "circe": 0.5}, "usd": 0.0003,
                           "heat_c_by_host": {"athena": 0.001}},
                          {"actor": "hub.outreach", "wh_by_host": None, "usd": None, "heat_c_by_host": None}],
        pr.WASTE_BY_LANE_SQL: [{"lane": "agent", "held_s": 8858.2}, {"lane": "chat", "held_s": 120.0}],
        pr.WASTE_TOP_SQL: [{"holder": "durable-runs:d7d0c7d60c96", "role": "agent", "held_s": 8858.2,
                            "reason": "attempt_failed", "at": NOW - timedelta(hours=1)}],
        pr.QUEUE_WAIT_SQL: [{"lane": "agent", "family": "durable.curiosity_investigate", "p50_s": 14.4,
                             "p90_s": 5275.0, "n": 6}],
        pr.DECISIONS_SQL: [{"decided_at": NOW - timedelta(minutes=3), "family": "hub.outreach", "verb": "say",
                            "tier": 2, "outcome": "deferred", "reason": "capacity_reserved", "net": 0.35,
                            "binding_constraint": "lane:agent", "why": "lane:agent is full",
                            "prices_seen": json.dumps({"heat": 0.5, "lane:agent": 0.3})}],
        pr.CLAUDE_TOKENS_SQL: [{"input": 198, "output": 38899, "cache_read": 7253496, "cache_write": 721749,
                                "usd": 9.11, "priced": 5, "unpriced": 0, "ticks": 96, "newest": NOW}],
        pr.VRAM_SQL: [{"node": "circe", "ts": (NOW - timedelta(minutes=1)).isoformat(),
                       "measurements": {"gpu_count": 4.0, "gpu1_mem_used_mb": 30000.0, "gpu1_mem_total_mb": 32768.0}}],
        pr.SERVICES_SQL: [{"created_at": (NOW - timedelta(hours=2)).replace(tzinfo=None),
                           "context": {"event": "stopped_container", "service_id": "orion-kev/kev",
                                       "containers": ["orion-athena-kev"], "states": ["exited"],
                                       "stopped_hours": 332.85}}],
    }


class FakePool:
    def __init__(self, rows: dict, fail: set[str] = frozenset()):
        self.rows = rows
        self.fail = fail
        self.calls: list[str] = []

    def connection(self):
        pool = self

        class _Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

        class _Conn:
            async def execute(self, sql, params=None):
                pool.calls.append(sql)
                for key in pool.fail:
                    if key in sql:
                        raise RuntimeError(f"relation for {key} does not exist")
                return _Cur(list(pool.rows.get(sql, [])))

        class _CM:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *a):
                return False

        return _CM()


def _weights():
    async def fn():
        return {"hub.outreach": Weight(Weight.Ev(n=43, mean_rate=0.63, engagement_rate=0.63, engagement_n=43)),
                "topic_foundry.run": Weight(Weight.Ev(n=0, mean_rate=None), basis="prior")}
    return fn


def _deps(pool, **kw) -> pr.ReportDeps:
    return pr.ReportDeps(pool=pool, heat_sensitivity_path=HEAT_CFG, gpu_pool_path=POOL_CFG,
                         tiers={"hub.outreach": 2, "actions.journal": 1}, family_weights=_weights(), **kw)


def _build(rows, *, q=None, fail=frozenset(), **kw) -> PhysiologyReportV1:
    return asyncio.run(pr.build_report(_deps(FakePool(rows, fail), **kw), q or pr.parse_query("24h", None, None, None, None), now=NOW))


def _reasons(rep, section):
    return [m.reason for m in rep.missing if m.section == section]


def test_full_report_fills_every_section_from_rows():
    rep = _build(full_rows())
    assert rep.schema_version == "physiology.report.v1"
    assert rep.window.end == NOW and rep.window.start == NOW - timedelta(hours=24)
    heat = next(l for l in rep.limits if l.constraint == "heat")
    assert heat.line == 31.0 and heat.now.bind_state == "exogenous_bind" and "share 2.43" in heat.now.why
    assert heat.hourly[0].price == 0.5 and heat.label.startswith("Cabinet heat")
    claude_lim = next(l for l in rep.limits if l.constraint == "claude")
    assert claude_lim.now.bind_state == "no_reading" and claude_lim.now.level is None
    assert [f.family for f in rep.wants.families] == ["hub.outreach", "actions.journal"]
    assert rep.wants.families[0].deferred == 30 and rep.wants.families[1].tier == 1
    assert rep.rhythm[0].heavy == 3000.0 and rep.rhythm[0].hour == 4
    m = rep.money
    assert (m.mtd_usd, m.forecast_usd, m.envelope_usd, m.floor_usd) == (182.1, 196.92, 100.0, 193.85)
    assert m.rate_usd_per_kwh == 0.1255 and m.daily_kwh[-1].kwh == 25.7
    units = {u.unit: u for u in rep.unit_costs}
    assert units["1 kWh"].status == "derived" and units["1 kWh"].usd == 0.1255
    assert units["100 W for 2 h on athena"].heat_c == pytest.approx(0.6425, abs=1e-3)
    assert units["one reverie_diffusion run"].status == "measured"
    assert units["one hub.outreach act"].wh == 2.0 and units["one hub.outreach act"].usd == 0.0003
    assert units["per 1k tokens"].status == "not_measured" and "token" in units["per 1k tokens"].source
    assert rep.waste.failed_lease_s_by_lane["agent"] == 8858.2 and rep.waste.top[0].reason == "attempt_failed"
    assert rep.queue_wait[0].p90_s == 5275.0
    assert rep.decisions[0].prices_seen == {"heat": 0.5, "lane:agent": 0.3}
    assert rep.claude.window_tokens.output == 38899 and rep.claude.usd_est == 9.11
    assert "no_observation" in rep.claude.status and "harness sessions are not in this count" in rep.claude.status
    gpu1 = next(c for c in rep.vram if c.card == "gpu1")
    assert gpu1.headroom_mb == 2768.0 and "agent" in gpu1.roles
    vpc = rep.value_per_cost
    assert [v.family for v in vpc] == ["hub.outreach"] and vpc[0].value == 0.63
    assert vpc[0].usd_per_act == 0.0003 and vpc[0].status == "measured"
    assert any("topic_foundry.run" in r for r in _reasons(rep, "value_per_cost"))
    assert rep.services.down[0].name == "orion-kev/kev" and rep.services.down[0].host == "athena"
    assert rep.services.down[0].since == NOW - timedelta(hours=2) - timedelta(hours=332.85)
    assert "total" in rep.timing_ms
    # Only vram has partial data (circe gpu1 reported; other cards not): no section is missing outright.
    assert {m.section for m in rep.missing} <= {"value_per_cost"}


def test_every_empty_section_says_why():
    rep = _build({})
    for section in REPORT_SECTIONS:
        assert _reasons(rep, section), f"{section} is empty with no reason"
    assert any("prices" in r or "no tick" in r for r in _reasons(rep, "limits"))
    assert any("bids began 2026-10-27" in r for r in _reasons(rep, "wants"))
    assert any("gpuN_mem_used_mb" in r for r in _reasons(rep, "vram"))
    assert any("nothing reported stopped" in r for r in _reasons(rep, "services"))
    # unit_costs never empties: derived/not_measured rows always say what is known.
    assert {u.status for u in rep.unit_costs} == {"not_measured", "derived"}


def test_a_failed_read_empties_only_its_own_section():
    rep = _build(full_rows(), fail={"gpu_pool_events"})
    assert rep.waste.failed_lease_s_by_lane == {} and rep.queue_wait == [] and rep.rhythm == []
    for s in ("waste", "queue_wait", "rhythm"):
        assert any(r.startswith("read failed: RuntimeError") for r in _reasons(rep, s)), s
    assert rep.money.mtd_usd == 182.1 and rep.wants.families


def test_market_off_says_so():
    rows = full_rows()
    rep = asyncio.run(pr.build_report(pr.ReportDeps(pool=FakePool(rows), heat_sensitivity_path=HEAT_CFG,
                                                    gpu_pool_path=POOL_CFG, family_weights=None),
                                      pr.parse_query("24h", "value_per_cost", None, None, None), now=NOW))
    assert rep.value_per_cost == [] and "ORION_PHYSIOLOGY_BID_ENABLED" in _reasons(rep, "value_per_cost")[0]


def test_sections_filter_reads_only_what_is_asked():
    pool = FakePool(full_rows())
    q = pr.parse_query("7d", "money,waste", None, None, None)
    rep = asyncio.run(pr.build_report(_deps(pool), q, now=NOW))
    assert rep.window.start == NOW - timedelta(days=7)
    assert rep.money.mtd_usd == 182.1 and rep.waste.top
    assert pr.QUEUE_WAIT_SQL not in pool.calls and pr.DECISIONS_SQL not in pool.calls
    assert any(m.section == "*" and "not asked for" in m.reason for m in rep.missing)


def test_seven_and_thirty_day_windows_note_that_prices_began_on_10_27():
    rep = _build(full_rows(), q=pr.parse_query("30d", "limits", None, None, None))
    assert any("prices began 2026-10-27" in r for r in _reasons(rep, "limits"))


def test_parse_query_rejects_bad_parameters():
    with pytest.raises(ValueError, match="window"):
        pr.parse_query("1h", None, None, None, None)
    with pytest.raises(ValueError, match="unknown section"):
        pr.parse_query("24h", "money,feelings", None, None, None)
    with pytest.raises(ValueError, match="outcome"):
        pr.parse_query("24h", None, None, "maybe", None)
    with pytest.raises(ValueError, match="limit"):
        pr.parse_query("24h", None, None, None, 0)
    q = pr.parse_query(None, " decisions , wants ", "hub.outreach", "deferred", 5)
    assert q.sections == ("decisions", "wants") and q.family == "hub.outreach" and q.limit == 5


def test_floor_parse_matches_the_budget_organs_own_wording():
    """floor_usd is read from the Budget organ's why; if budget.py rewords it, this must fail."""
    src = (REPO_ROOT / "orion/physiology/budget.py").read_text()
    assert "the always-on floor alone (${floor_rate:.3f}/h) reaches ${floor:.2f}" in src
    assert pr._FLOOR_RE.search(ENERGY_WHY).group(1) == "193.85"


def test_route_answers_503_422_and_caches(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi import HTTPException

    from app import main as app_main

    old = (app_main._checkpointer_cm, app_main.market, app_main._report_cache, app_main._settings.physiology_report_enabled)
    pool = FakePool(full_rows())
    try:
        app_main._settings.physiology_report_enabled = False
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_main.physiology_report())
        assert e.value.status_code == 503
        app_main._settings.physiology_report_enabled = True
        app_main._checkpointer_cm, app_main.market, app_main._report_cache = pool, None, None
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_main.physiology_report(window="2d"))
        assert e.value.status_code == 422
        out = asyncio.run(app_main.physiology_report(window="24h", sections="money,limits"))
        assert PhysiologyReportV1.model_validate(out).money.mtd_usd == 182.1
        n = len(pool.calls)
        again = asyncio.run(app_main.physiology_report(window="24h", sections="money,limits"))
        assert again == out and len(pool.calls) == n   # served from the cache, no new reads
    finally:
        (app_main._checkpointer_cm, app_main.market, app_main._report_cache,
         app_main._settings.physiology_report_enabled) = old


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_report_sql_on_postgres():
    """Every section's SQL runs on Postgres against the live tables' column shapes."""
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = f"t_{uuid4().hex[:10]}"
    now = datetime.now(timezone.utc)
    ddl = [
        "CREATE TABLE physiology_state (tick_id text PRIMARY KEY, generated_at timestamptz NOT NULL, constraints jsonb NOT NULL, warnings jsonb, created_at timestamptz DEFAULT now())",
        """CREATE TABLE want_ledger (decision_id text PRIMARY KEY, want_id text NOT NULL, family text NOT NULL,
           producer text NOT NULL, verb text NOT NULL, tier integer NOT NULL, decided_at timestamptz NOT NULL,
           outcome text NOT NULL, reason text NOT NULL, lane text, net double precision, binding_constraint text,
           shadow boolean NOT NULL, correlation_id text, decision jsonb NOT NULL, created_at timestamptz DEFAULT now())""",
        """CREATE TABLE gpu_pool_events (event_id varchar PRIMARY KEY, created_at timestamptz DEFAULT now(),
           generated_at timestamptz NOT NULL, event varchar NOT NULL, lease_id varchar, holder varchar, work_class varchar,
           priority varchar, role varchar, cards json, turn_correlation_id varchar, attempt integer,
           waited_ms double precision, held_ms double precision, reason varchar, detail json)""",
        "CREATE TABLE orion_energy_hourly (hour_start timestamptz, source varchar, kwh double precision, PRIMARY KEY (hour_start, source))",
        """CREATE TABLE energy_run_cost (id bigserial PRIMARY KEY, intent_id varchar, workload_kind varchar,
           window_start timestamptz, energy_kwh double precision, estimated_run_cost_usd double precision)""",
        "CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar, opened_at timestamptz, cost_json jsonb)",
        """CREATE TABLE dev_economics_ledger_log (event_id varchar PRIMARY KEY, observed_at timestamptz,
           total_input_tokens integer, total_output_tokens integer, total_cache_creation_input_tokens integer,
           total_cache_read_input_tokens integer, total_estimated_cost_usd double precision,
           priced_session_count integer, unpriced_session_count integer)""",
        "CREATE TABLE orion_biometrics_summary (id serial PRIMARY KEY, \"timestamp\" varchar, node varchar, measurements jsonb)",
        "CREATE TABLE notify_requests (notification_id varchar PRIMARY KEY, source_service varchar, created_at timestamp, context jsonb)",
    ]

    async def run():
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=4, open=False,
                                           kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0,
                                                   "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    for stmt in ddl:
                        await conn.execute(stmt)
                    await conn.execute("INSERT INTO physiology_state VALUES ('t1', %s, %s)",
                                       (now - timedelta(minutes=5), json.dumps(LATEST["constraints"])))
                    await conn.execute("INSERT INTO want_ledger (decision_id, want_id, family, producer, verb, tier, decided_at, "
                                       "outcome, reason, shadow, decision) VALUES ('d1', 'run1', 'durable.self_inquiry', "
                                       "'orion-durable-runs.admission', 'ask', 2, %s, 'deferred', 'capacity_reserved', true, %s)",
                                       (now - timedelta(minutes=9), json.dumps({"why": "full", "prices_seen": {"heat": 0.5}})))
                    await conn.execute("INSERT INTO want_ledger (decision_id, want_id, family, producer, verb, tier, decided_at, "
                                       "outcome, reason, shadow, decision) VALUES ('p1', 'probe-1', 'x', 'probe', 'ask', 2, %s, "
                                       "'funded', 'net_positive', true, '{}')", (now - timedelta(minutes=8),))
                    for i, (ev, reason, held, waited, holder) in enumerate([
                            ("granted", None, None, 60000.0, "durable-runs:run1"),
                            ("released", "attempt_failed", 8500000.0, None, "durable-runs:run1"),
                            ("released", "ok", 1000.0, None, "cortex-exec:x")]):
                        await conn.execute("INSERT INTO gpu_pool_events (event_id, generated_at, event, holder, priority, role, "
                                           "waited_ms, held_ms, reason) VALUES (%s, %s, %s, %s, 'background', 'agent', %s, %s, %s)",
                                           (f"e{i}", now - timedelta(minutes=30 - i), ev, holder, waited, held, reason))
                    await conn.execute("INSERT INTO orion_energy_hourly VALUES (%s, 'athena', 1.5)", (now - timedelta(hours=3),))
                    await conn.execute("INSERT INTO energy_run_cost (workload_kind, window_start, energy_kwh, estimated_run_cost_usd) "
                                       "VALUES ('reverie_diffusion', %s, 0.0004, 0.00005)", (now - timedelta(hours=1),))
                    await conn.execute("INSERT INTO grammar_acts VALUES ('a1', 'hub.outreach', %s, %s)",
                                       (now - timedelta(hours=1), json.dumps({"wh_by_host": {"athena": 2.0}, "usd": 0.001})))
                    await conn.execute("INSERT INTO dev_economics_ledger_log VALUES ('x', %s, 1, 2, 3, 4, 0.5, 1, 0)",
                                       (now - timedelta(hours=1),))
                    await conn.execute("INSERT INTO orion_biometrics_summary (\"timestamp\", node, measurements) VALUES (%s, 'circe', %s)",
                                       (now.isoformat(), json.dumps({"gpu_count": 4, "gpu0_mem_used_mb": 100, "gpu0_mem_total_mb": 32768})))
                    await conn.execute("INSERT INTO notify_requests VALUES ('n1', 'orion-mesh-guardian', %s, %s)",
                                       ((now - timedelta(hours=1)).replace(tzinfo=None),
                                        json.dumps({"event": "stopped_container", "service_id": "orion-kev/kev",
                                                    "containers": ["orion-athena-kev"], "states": ["exited"], "stopped_hours": 3})))
                deps = pr.ReportDeps(pool=pool, heat_sensitivity_path=HEAT_CFG, gpu_pool_path=POOL_CFG)
                rep = await pr.build_report(deps, pr.parse_query("30d", None, None, None, None), now=now)
                failed = [m for m in rep.missing if m.reason.startswith("read failed")]
                assert failed == [], failed
                assert rep.limits and rep.money.floor_usd == 193.85 and rep.money.daily_kwh
                assert [f.family for f in rep.wants.families] == ["durable.self_inquiry"]   # probe left out
                assert rep.waste.failed_lease_s_by_lane == {"agent": 8500.0}
                assert rep.queue_wait[0].family == "durable.self_inquiry" and rep.queue_wait[0].p50_s == 60.0
                assert rep.decisions[0].why == "full" and rep.decisions[0].prices_seen == {"heat": 0.5}
                assert rep.rhythm and rep.rhythm[0].heavy > 0
                assert rep.claude.window_tokens.cache_write == 3 and rep.claude.usd_est == 0.5
                assert next(c for c in rep.vram if c.card == "gpu0").headroom_mb == 32668.0
                assert rep.services.down[0].name == "orion-kev/kev"
                assert any(u.unit == "one hub.outreach act" and u.wh == 2.0 for u in rep.unit_costs)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    asyncio.run(run())
