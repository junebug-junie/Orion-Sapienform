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
                       "measurements": {"gpu_count": 4.0, "gpu1_mem_used_mb": 30000.0, "gpu1_mem_total_mb": 32768.0,
                                        "gpu1_watts": 210.0, "cpu_busy_cores": 9.0, "cpu_idle_watts": 100.0,
                                        "chassis_watts": 1150.0, "gpu_watts_total": 450.0, "cpu_watts_total": 120.0,
                                        "fan_pct_max": 50.0, "gpu1_temp_c": 71.0,
                                        "container_cpu_watts.orion-circe-atlas-agent": 18.5,
                                        "container_cpu_watts.orion-circe-vision-host": 4.25,
                                        "container_cpu_watts_small": 1.0, "cpu_watts_host_other": 2.0}}],
        pr.ATTRIBUTION_PRESENT_SQL: [{"present": True}],
        # circe, 2026-10-30 00-03Z replayed live (evals/act_energy_closure.py): 0.8% coverage.
        pr.ATTRIBUTION_DAILY_SQL: [
            {"host": "circe", "day": "2026-10-30", "hours": 3, "rest_wh": 1623.0, "above_rest_wh": 1748.35,
             "sponsored_wh": 0.0, "metered_wh": 13.91, "leased_unmetered_wh": 1495.07, "unattributed_wh": 1734.44,
             "min_hour_unattributed_wh": 500.0, "covered_h": 3.0, "split_unknown_h": 0.0},
            {"host": "athena", "day": "2026-10-30", "hours": 3, "rest_wh": 1020.0, "above_rest_wh": 1228.95,
             "sponsored_wh": 549.37, "metered_wh": 0.0, "leased_unmetered_wh": 0.0, "unattributed_wh": 679.57,
             "min_hour_unattributed_wh": 200.0, "covered_h": 3.0, "split_unknown_h": 0.0}],
        pr.ATTRIBUTION_CHARGED_SQL: [{"host": "circe", "day": "2026-10-30", "charged_wh": 13.97, "acts": 58}],
        pr.HOSTS_REST_SQL: [{"node": "circe", "chassis_rest": 541.0, "gpu_rest": 190.0, "cpu_rest": 45.0,
                              "other_rest": 280.0, "gpu1_rest": None}],
        pr.HOSTS_FLOOR_SQL: [{"node": "circe", "chassis_floor": 578.0, "gpu1_floor": 42.0}],
        pr.HOSTS_DAILY_SQL: [{"node": "circe", "d": "2026-10-28", "w": 1027.4}, {"node": "circe", "d": "2026-10-29", "w": 962.0}],
        pr.HOSTS_HOLDERS_SQL: [{"card": "gpu1", "holder": "durable-runs:curiosity.investigate", "lease_s": 38791.0},
                               {"card": "gpu1", "holder": "http:anthropic", "lease_s": 13572.3}],
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
    assert units["one hub.outreach act (estimate: family mean)"].wh == 2.0 and units["one hub.outreach act (estimate: family mean)"].usd == 0.0003
    assert units["per 1k tokens"].status == "not_measured" and "token" in units["per 1k tokens"].source
    circe = units["work not yet metered on circe, 2026-10-30 UTC"]
    assert circe.status == "measured" and circe.wh == 1734.4 and circe.usd == pytest.approx(1734.44 / 1000 * 0.1255, abs=1e-4)
    assert "coverage 0.8% of activity" in circe.source and "GPU leases whose turn has no act 1495" in circe.source
    assert "acts actually charged 14 (58 acts)" in circe.source and "OVER" not in circe.source
    assert "sponsored 549" in units["work not yet metered on athena, 2026-10-30 UTC"].source
    assert rep.waste.failed_lease_s_by_lane["agent"] == 8858.2 and rep.waste.top[0].reason == "attempt_failed"
    assert rep.queue_wait[0].p90_s == 5275.0
    assert rep.decisions[0].prices_seen == {"heat": 0.5, "lane:agent": 0.3}
    assert rep.claude.window_tokens.output == 38899 and rep.claude.usd_est == 9.11
    assert "no_observation" in rep.claude.status and "harness sessions are not in this count" in rep.claude.status
    gpu1 = next(c for c in rep.vram if c.card == "gpu1")
    assert gpu1.headroom_mb == 2768.0 and "agent" in gpu1.roles and gpu1.watts == 210.0
    assert [(p.name, p.watts) for p in rep.processes] == [
        ("orion-circe-atlas-agent", 18.5), ("orion-circe-vision-host", 4.2), ("(host, not in a container)", 2.0),
        ("(small containers)", 1.0)]
    assert all(p.host == "circe" and p.cores is None for p in rep.processes)
    assert "family total 2 W·h over 1 acts" in units["one hub.outreach act (estimate: family mean)"].source
    vpc = rep.value_per_cost
    assert [v.family for v in vpc] == ["hub.outreach"] and vpc[0].value == 0.63
    assert vpc[0].usd_per_act == 0.0003 and vpc[0].status == "measured"
    assert any("topic_foundry.run" in r for r in _reasons(rep, "value_per_cost"))
    assert rep.services.down[0].name == "orion-kev/kev" and rep.services.down[0].host == "athena"
    assert rep.services.down[0].since == NOW - timedelta(hours=2) - timedelta(hours=332.85)
    assert "total" in rep.timing_ms
    assert any(r.startswith("partial:") for r in _reasons(rep, "vram"))      # only circe gpu1 reports
    assert any("hecate" in r and "athena" in r for r in _reasons(rep, "vram"))  # no fresh rows for them
    circe = rep.hosts[0]
    assert (circe.host, circe.chassis_w_now, circe.chassis_w_rest, circe.above_rest_w) == ("circe", 1150.0, 541.0, 609.0)
    assert circe.other_w == 580.0 and circe.fan_pct == 50.0
    assert {m.section for m in rep.missing} <= {"value_per_cost", "vram", "hosts"}
    assert rep.query == {"window": "24h", "sections": list(REPORT_SECTIONS), "family": None, "outcome": None,
                         "limit": 50}


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
        # services/orion-sql-writer/app/models/orion_energy_attribution_hourly.py
        """CREATE TABLE orion_energy_attribution_hourly (hour_start timestamptz, host varchar, rest_w double precision,
           rest_wh double precision, above_rest_wh double precision, sponsored_wh double precision,
           metered_wh double precision, leased_unmetered_wh double precision, unattributed_wh double precision,
           covered_sec double precision, split_unknown_sec double precision, rest_source varchar,
           updated_at timestamptz, PRIMARY KEY (hour_start, host))""",
        """CREATE TABLE energy_run_cost (id bigserial PRIMARY KEY, intent_id varchar, workload_kind varchar,
           window_start timestamptz, energy_kwh double precision, estimated_run_cost_usd double precision)""",
        "CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar, opened_at timestamptz, cost_json jsonb, correlation_id varchar)",
        """CREATE TABLE dev_economics_ledger_log (event_id varchar PRIMARY KEY, observed_at timestamptz,
           total_input_tokens integer, total_output_tokens integer, total_cache_creation_input_tokens integer,
           total_cache_read_input_tokens integer, total_estimated_cost_usd double precision,
           priced_session_count integer, unpriced_session_count integer)""",
        "CREATE TABLE orion_biometrics_summary (id serial PRIMARY KEY, \"timestamp\" varchar, node varchar, measurements jsonb)",
        "CREATE TABLE notify_requests (notification_id varchar PRIMARY KEY, source_service varchar, created_at timestamp, context jsonb)",
        "CREATE TABLE durable_run_workflow (run_id text PRIMARY KEY, workflow text NOT NULL, created_at timestamptz, terminal text)",
        # services/orion-sql-db/manual_migration_llm_call_usage_v1.sql (PR #2974), the columns this report reads
        """CREATE TABLE llm_call_usage (call_id text PRIMARY KEY, recorded_at timestamptz NOT NULL, recorder text NOT NULL,
           correlation_id text, prompt_tokens integer, cached_prompt_tokens integer, completion_tokens integer)""",
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
                            ("released", "ok", 1000.0, None, "cortex-exec:x"),
                            # an expiry writes two close rows for one lease: counted once
                            ("expired", "heartbeat_lost", 5000.0, None, "durable-runs:run2"),
                            ("released", "expire:heartbeat_lost", 5000.0, None, "durable-runs:run2"),
                            # an operator bench is not waste
                            ("released", "operator:gpu-power-bench", 7000.0, None, "operator:bench")]):
                        await conn.execute("INSERT INTO gpu_pool_events (event_id, generated_at, event, holder, priority, role, "
                                           "waited_ms, held_ms, reason) VALUES (%s, %s, %s, %s, 'background', 'agent', %s, %s, %s)",
                                           (f"e{i}", now - timedelta(minutes=30 - i), ev, holder, waited, held, reason))
                    await conn.execute("INSERT INTO orion_energy_hourly VALUES (%s, 'athena', 1.5)", (now - timedelta(hours=3),))
                    for h in (2, 3):
                        await conn.execute("INSERT INTO orion_energy_attribution_hourly VALUES (%s, 'circe', 541, 541, "
                                           "600, 0, 10, 500, 590, 3600, 0, 'fixture', now())",
                                           (now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=h),))
                    await conn.execute("INSERT INTO energy_run_cost (workload_kind, window_start, energy_kwh, estimated_run_cost_usd) "
                                       "VALUES ('reverie_diffusion', %s, 0.0004, 0.00005)", (now - timedelta(hours=1),))
                    await conn.execute("INSERT INTO grammar_acts VALUES ('a2', 'hub.chat_reply', %s, %s, 'corr-2')",
                                       (now - timedelta(hours=2), json.dumps({"wh_by_host": {"circe": 7.0},
                                                                              "wh_gpu_by_host": {"circe": 3.0}})))
                    await conn.execute("INSERT INTO grammar_acts VALUES ('a1', 'hub.outreach', %s, %s, 'corr-1')",
                                       (now - timedelta(hours=1), json.dumps({"wh_by_host": {"athena": 1.5, "circe": 0.5}, "usd": 0.001})))
                    for cid, rec, corr, p_, c_ in (("c1", "gateway", "corr-1", 900, 100), ("c2", "harness_relay", "corr-1", 400, 100),
                                                   ("c3", "gateway", None, 50, 50)):
                        await conn.execute("INSERT INTO llm_call_usage VALUES (%s, %s, %s, %s, %s, 0, %s)",
                                           (cid, now - timedelta(minutes=50), rec, corr, p_, c_))
                    await conn.execute("INSERT INTO dev_economics_ledger_log VALUES ('x', %s, 1, 2, 3, 4, 0.5, 1, 0)",
                                       (now - timedelta(hours=1),))
                    # the writer's varchar shape ('2026-10-30 03:09:37.461097+00'), which the hosts reads compare as text
                    await conn.execute("INSERT INTO orion_biometrics_summary (\"timestamp\", node, measurements) VALUES (%s, 'circe', %s)",
                                       ((now - timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S.%f+00"), json.dumps({"gpu_count": 4, "gpu0_mem_used_mb": 100, "gpu0_mem_total_mb": 32768,
                                                                     "chassis_watts": 1150, "gpu_watts_total": 450,
                                                                     "cpu_watts_total": 120, "gpu1_watts": 230})))
                    # inside the pinned rest window (2026-10-01..13), in the varchar shape the writer uses
                    await conn.execute("INSERT INTO orion_biometrics_summary (\"timestamp\", node, measurements) VALUES "
                                       "('2026-10-05 12:00:00.000001+00', 'circe', %s)",
                                       (json.dumps({"chassis_watts": 541, "gpu_watts_total": 190, "cpu_watts_total": 45}),))
                    await conn.execute("INSERT INTO durable_run_workflow VALUES ('run9', 'curiosity.investigate', %s, 'completed')",
                                       (now - timedelta(hours=2),))
                    await conn.execute("INSERT INTO gpu_pool_events (event_id, generated_at, event, holder, priority, role, "
                                       "cards, held_ms, reason) VALUES ('h1', %s, 'released', 'durable-runs:run9', 'interactive', "
                                       "'agent', '[\"gpu1\"]', 600000.0, 'ok')", (now - timedelta(days=5),))
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
                assert rep.waste.failed_lease_s_by_lane == {"agent": 8505.0}   # 8500 + one 5 s expiry
                assert [t.reason for t in rep.waste.top] == ["attempt_failed", "heartbeat_lost"]
                assert rep.queue_wait[0].family == "durable.self_inquiry" and rep.queue_wait[0].p50_s == 60.0
                assert rep.decisions[0].why == "full" and rep.decisions[0].prices_seen == {"heat": 0.5}
                # the 8500 s hold is spread over the hours it was held, not piled into its close hour
                assert len(rep.rhythm) >= 3 and max(c.total for c in rep.rhythm) <= 3600.0 + 1e-6
                assert sum(c.heavy for c in rep.rhythm) == pytest.approx(8500.0 + 1.0 + 5.0 + 7.0, abs=0.5)
                assert rep.claude.window_tokens.cache_write == 3 and rep.claude.usd_est == 0.5
                assert next(c for c in rep.vram if c.card == "gpu0").headroom_mb == 32668.0
                assert rep.services.down[0].name == "orion-kev/kev"
                circe = next(h for h in rep.hosts if h.host == "circe")
                assert (circe.chassis_w_now, circe.chassis_w_rest, circe.other_w) == (1150.0, 541.0, 580.0)
                assert (circe.gpu_w_rest, circe.cpu_w_rest, circe.other_w_rest) == (190.0, 45.0, 306.0)
                assert circe.recent_floor_w == 1150.0 and circe.daily
                gpu1 = next(c for c in circe.cards if c.card == "gpu1")
                assert gpu1.watts == 230.0 and gpu1.idle_w == 230.0   # only one recent sample: its own floor
                assert [(x.holder, x.label, x.lease_s) for x in gpu1.top_holders] == [
                    ("durable-runs:curiosity.investigate", "curiosity.investigate runs", 600.0)]
                assert any(u.unit == "one hub.outreach act (estimate: family mean)" and u.wh == 2.0 for u in rep.unit_costs)
                tok = next(u for u in rep.unit_costs if u.unit == "per 1k tokens")
                # act corr-1: 2.0 Wh over 1,000 tokens -- gateway rows only. The leftover harness_relay row
                # (retired 2026-10-29) is the same call as c1 now that FCC forwards the turn id; counting it
                # would double the act's tokens. Totals: gateway only (2 calls, 1,100).
                assert tok.status == "measured" and tok.wh == pytest.approx(2.0 * 1000 / 1000, abs=1e-3)
                assert "2 gateway calls, 1,100 tokens" in tok.source
                met = [u for u in rep.unit_costs if u.unit.startswith("work not yet metered on circe")]
                assert met and sum(u.wh for u in met) == pytest.approx(1180.0) and "coverage 1.7%" in met[0].source
                assert any("acts actually charged 7 (1 acts)" in u.source for u in met)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    asyncio.run(run())


def test_stale_newest_tick_is_said_not_served_as_now():
    rows = full_rows()
    rows[pr.LIMITS_LATEST_SQL] = [{**LATEST, "generated_at": NOW - timedelta(hours=3)}]
    rep = _build(rows)
    for s in ("limits", "money", "claude", "unit_costs"):
        assert any(r.startswith("stale: the newest physiology tick is 180 min old") for r in _reasons(rep, s)), s


def test_family_filter_says_which_sections_it_narrows():
    rows = full_rows()
    rows[pr.QUEUE_WAIT_SQL] = []
    rows[pr.WANTS_HOURLY_SQL] = []
    rep = _build(rows, q=pr.parse_query("24h", None, "hub.outreach", None, None))
    assert any("only durable-run admissions carry a want family" in r for r in _reasons(rep, "queue_wait"))
    assert any("for family hub.outreach" in r for r in _reasons(rep, "wants"))
    star = " ".join(_reasons(rep, "*"))
    assert "family=hub.outreach narrows" in star and "waste" in star and "limits" in star
    assert [v.family for v in rep.value_per_cost] == ["hub.outreach"]


def test_holder_named_queue_rows_are_marked():
    rows = full_rows()
    rows[pr.QUEUE_WAIT_SQL] = [{"lane": "chat", "family": "cortex-exec", "family_source": "holder", "p50_s": 0.1,
                                "p90_s": 0.4, "n": 600}]
    rep = _build(rows)
    assert rep.queue_wait[0].family_source == "holder"
    assert any(r.startswith("partial: only durable-run admissions") for r in _reasons(rep, "queue_wait"))


def test_no_token_rows_reads_unknown_not_zero():
    rows = full_rows()
    rows[pr.CLAUDE_TOKENS_SQL] = [{"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "usd": None,
                                   "priced": 0, "unpriced": 0, "ticks": 0, "newest": None}]
    rep = _build(rows)
    t = rep.claude.window_tokens
    assert (t.input, t.output, t.cache_read, t.cache_write) == (None, None, None, None) and rep.claude.usd_est is None
    assert any("token counts unknown" in r for r in _reasons(rep, "claude"))


def test_a_helper_read_failure_names_the_dependent_sections():
    rep = _build(full_rows(), fail={"ORDER BY generated_at DESC LIMIT 1"})
    for s in ("limits", "money", "claude", "unit_costs"):
        assert any(r.startswith("read failed (latest)") for r in _reasons(rep, s)), s


def test_cache_skips_failed_reports_and_shares_one_build():
    cache = pr.ReportCache(60.0)
    calls = []

    async def build_ok():
        calls.append(1)
        await asyncio.sleep(0.01)
        return _build_sync_ok

    _build_sync_ok = _build(full_rows())
    failed = _build(full_rows(), fail={"gpu_pool_events"})

    async def run():
        a, b = await asyncio.gather(cache.get_or_build(("k",), build_ok), cache.get_or_build(("k",), build_ok))
        assert a is b and len(calls) == 1
        assert await cache.get_or_build(("k",), build_ok) is a and len(calls) == 1

        async def build_failed():
            calls.append(2)
            return failed
        await cache.get_or_build(("f",), build_failed)
        await cache.get_or_build(("f",), build_failed)
        assert calls.count(2) == 2   # a report with a failed read is never cached

    asyncio.run(run())


def test_per_1k_tokens_joins_act_energy_to_the_token_ledger():
    absent = pr.token_cost_row({"present": False})
    assert absent["status"] == "not_measured" and "llm_call_usage does not exist yet" in absent["source"]
    no_energy = pr.token_cost_row({"present": True, "calls": 10, "prompt": 900, "completion": 100,
                                   "join": {"acts_with_energy": 0, "joined": 0}})
    assert no_energy["status"] == "not_measured" and "nothing to divide" in no_energy["source"]
    no_join = pr.token_cost_row({"present": True, "calls": 10, "prompt": 900, "completion": 100,
                                 "join": {"acts_with_energy": 3, "joined": 0, "tokens": None}})
    assert "none of the 3 acts" in no_join["source"]
    ok = pr.token_cost_row({"present": True, "calls": 10, "prompt": 900, "completion": 100,
                            "join": {"acts_with_energy": 3, "joined": 2, "wh": 4.0, "usd": 0.002, "tokens": 2000}})
    assert ok["status"] == "measured" and ok["wh"] == 2.0 and ok["usd"] == 0.001
    assert "J/token 7.2" in ok["source"] and "2 of 3 acts" in ok["source"]
    rows = full_rows()
    rows[pr.TOKENS_PRESENT_SQL] = [{"present": True}]
    rows[pr.TOKENS_TOTAL_SQL] = [{"calls": 10, "prompt": 900, "completion": 100, "first_at": NOW}]
    rows[pr.TOKENS_ACT_JOIN_SQL] = [{"acts_with_energy": 3, "joined": 2, "wh": 4.0, "usd": 0.002, "tokens": 2000}]
    rep = _build(rows)
    assert next(u for u in rep.unit_costs if u.unit == "per 1k tokens").status == "measured"


def test_processes_below_idle_floor_is_a_reason_not_zero():
    rows = full_rows()
    rows[pr.VRAM_SQL] = [{"node": "athena", "ts": NOW.isoformat(), "measurements": {"cpu_busy_cores": 3.0}}]
    rep = _build(rows, q=pr.parse_query("24h", "processes", None, None, None))
    assert rep.processes == []
    assert any("below the fitted idle floor" in r for r in _reasons(rep, "processes"))


def test_attribution_says_absent_table_and_flags_over_attribution():
    rows = full_rows()
    rows[pr.ATTRIBUTION_PRESENT_SQL] = [{"present": False}]
    units = {u.unit: u for u in _build(rows).unit_costs}
    assert "does not exist yet" in units["work not yet metered (per host per day)"].source
    rows[pr.ATTRIBUTION_PRESENT_SQL] = [{"present": True}]
    rows[pr.ATTRIBUTION_DAILY_SQL] = [dict(rows[pr.ATTRIBUTION_DAILY_SQL][0], min_hour_unattributed_wh=-3.5)]
    units = {u.unit: u for u in _build(rows).unit_costs}
    assert "OVER-ATTRIBUTED: an hour reads -3.50" in units["work not yet metered on circe, 2026-10-30 UTC"].source
    rows[pr.ATTRIBUTION_DAILY_SQL] = [dict(rows[pr.ATTRIBUTION_DAILY_SQL][0], split_unknown_h=2.0)]
    rows[pr.ATTRIBUTION_CHARGED_SQL] = [{"host": "circe", "day": "2026-10-30", "charged_wh": 5000.0, "acts": 3}]
    src = {u.unit: u for u in _build(rows).unit_costs}["work not yet metered on circe, 2026-10-30 UTC"].source
    assert "OVER-CHARGED: acts were charged 5000" in src and "PARTIAL: for 2.0 h" in src
    rows[pr.ATTRIBUTION_DAILY_SQL] = []
    units = {u.unit: u for u in _build(rows).unit_costs}
    assert units["work not yet metered (per host per day)"].status == "not_measured"


# --- hosts: what is spinning each machine ------------------------------------------------------


def _hosts_rows() -> dict:
    rows = full_rows()
    rows[pr.VRAM_SQL] = [
        {"node": "circe", "ts": (NOW - timedelta(minutes=1)).isoformat(),
         "measurements": {"gpu_count": 4.0, "chassis_watts": 1150.0, "gpu_watts_total": 450.0, "cpu_watts_total": 120.0,
                          "fan_pct_max": 50.0, "gpu1_watts": 230.0, "gpu1_temp_c": 71.0, "gpu1_mem_used_mb": 24888.0,
                          "gpu3_watts": 220.0, "gpu0_watts": 0.0,
                          "container_cpu_watts.d590a9d48af2": 8.4, "container_cpu_watts.e5287fa21cba": 6.1,
                          "container_cpu_watts.0123456789ab": 1.0, "cpu_watts_host_other": 2.0}},
        {"node": "athena", "ts": (NOW - timedelta(minutes=2)).isoformat(),
         "measurements": {"gpu_count": 2.0, "chassis_watts": 720.0, "gpu_watts_total": 90.0, "cpu_watts_total": 350.0,
                          "gpu0_watts": 26.0, "gpu1_watts": 64.0, "container_cpu_watts.orion-athena-sql-db": 13.8,
                          **{f"container_cpu_watts.zelda-dojo-live-{i}": 5.5 for i in range(32)}}},
        {"node": "hecate", "ts": (NOW - timedelta(hours=5)).isoformat(), "measurements": {"chassis_watts": 96.0}},
    ]
    rows[pr.HOSTS_REST_SQL] = [{"node": "circe", "chassis_rest": 541.0, "gpu_rest": 190.0, "cpu_rest": 45.0,
                                "other_rest": 280.0, "gpu1_rest": None, "gpu3_rest": None},
                               {"node": "athena", "chassis_rest": 340.0, "gpu_rest": 76.0, "cpu_rest": 118.0,
                                "other_rest": 92.0}]
    rows[pr.HOSTS_FLOOR_SQL] = [{"node": "circe", "chassis_floor": 578.0, "gpu1_floor": 41.8, "gpu3_floor": 72.3},
                                {"node": "athena", "chassis_floor": 690.0}]
    rows[pr.HOSTS_HOLDERS_SQL] = [
        {"card": "gpu1", "holder": "durable-runs:curiosity.investigate", "lease_s": 38791.0},
        {"card": "gpu1", "holder": "http:anthropic", "lease_s": 13572.3},
        {"card": "gpu3", "holder": "vision-council", "lease_s": 32944.1},
        {"card": "gpu3", "holder": "http:openai", "lease_s": 11640.2}]
    return rows


@pytest.fixture(autouse=True)
def _fresh_pinned_parts_cache():
    pr._PINNED_PARTS_CACHE.clear()   # FakePool ids can be reused across tests
    yield
    pr._PINNED_PARTS_CACHE.clear()


@pytest.mark.parametrize("body", [
    "- just\n- a list\n",                                                   # top level is not a mapping
    "hosts:\n  circe: 541\n  athena: 340\n",                               # bare numbers
    "hosts:\n  circe: {rest_w: 541, measured: '2026-10-01..2026-10-32'}\n",  # not a real date
    "hosts: [1, 2]\nsponsored: 7\n",
])
def test_a_malformed_rest_body_is_never_a_500(tmp_path, body):
    (tmp_path / "heat_sensitivity.yaml").write_text(Path(HEAT_CFG).read_text())
    (tmp_path / "rest_body.yaml").write_text(body)
    deps = pr.ReportDeps(pool=FakePool(_hosts_rows()), heat_sensitivity_path=str(tmp_path / "heat_sensitivity.yaml"),
                         gpu_pool_path=POOL_CFG)
    rep = asyncio.run(pr.build_report(deps, pr.parse_query("24h", "hosts", None, None, None), now=NOW))
    circe = next(h for h in rep.hosts if h.host == "circe")
    assert circe.chassis_w_rest == 541.0


def _names():
    async def fn():
        return {"d590a9d48af2": "orion-circe-atlas-llamacpp-fast", "e5287fa21cba": "orion-circe-atlas-llamacpp-metacog"}
    return fn


def test_hosts_say_what_spins_each_machine():
    rep = _build(_hosts_rows(), q=pr.parse_query("24h", "hosts,processes", None, None, None), container_names=_names())
    assert [h.host for h in rep.hosts] == ["circe", "athena"]          # largest draw first; hecate asleep
    c = rep.hosts[0]
    # pinned rest (no rest_body.yaml in the repo yet): the dated default, never the rolling floor
    assert c.chassis_w_rest == 541.0 and c.recent_floor_w == 578.0 and "2026-10-01..2026-10-13" in c.rest_basis
    # rest_body.yaml now exists (#2992), so the basis names the file; without it, "pinned default".
    assert "pinned in config/physiology/rest_body.yaml" in c.rest_basis or "pinned default" in c.rest_basis
    assert c.above_rest_w == 609.0 and c.usd_per_month_at_this_rate == round(609 * 730 / 1000 * 0.1255, 2)
    assert c.rest_usd_per_month == round(541 * 730 / 1000 * 0.1255, 2)
    assert (c.gpu_w, c.cpu_w, c.other_w) == (450.0, 120.0, 580.0)       # other = chassis - gpu - cpu
    assert (c.gpu_w_rest, c.cpu_w_rest, c.other_w_rest) == (190.0, 45.0, 280.0)
    gpu1 = next(x for x in c.cards if x.card == "gpu1")
    assert gpu1.watts == 230.0 and gpu1.idle_w == 41.8 and "agent" in gpu1.roles and gpu1.temp_c == 71.0
    assert [(h.holder, h.label) for h in gpu1.top_holders] == [
        ("durable-runs:curiosity.investigate", "curiosity.investigate runs"), ("http:anthropic", "harness turns")]
    assert next(x for x in c.cards if x.card == "gpu3").top_holders[1].label == "AI Town"
    assert next(x for x in c.cards if x.card == "gpu0").watts == 0.0   # 0 W is a reading, kept
    assert [x.card for x in c.cards] == ["gpu0", "gpu1", "gpu3"]        # gpu2 reports nothing: not a 0 W card
    # container ids -> names from the census; an id it does not know stays an id, and is said
    assert [x.name for x in c.top_containers][:2] == ["orion-circe-atlas-llamacpp-fast", "orion-circe-atlas-llamacpp-metacog"]
    assert "0123456789ab" in [x.name for x in c.top_containers]
    assert any(r.startswith("circe: some containers show as 12-char ids") for r in _reasons(rep, "hosts"))
    assert any(p.name == "orion-circe-atlas-llamacpp-fast" for p in rep.processes)
    assert [d.chassis_w_avg for d in c.daily] == [1027.4, 962.0]
    # athena: the Zelda dojo is Juniper's sponsored work, apart from Orion's own top containers
    a = rep.hosts[1]
    assert a.sponsored[0].name == "zelda_dojo" and a.sponsored[0].containers == 32 and a.sponsored[0].cpu_w == 176.0
    assert all(not x.name.startswith("zelda-dojo") for x in a.top_containers)
    assert a.top_containers[0].name == "orion-athena-sql-db"
    assert c.sponsored == []
    reasons = " | ".join(_reasons(rep, "hosts"))
    assert "hecate (asleep or its reporter is down)" in reasons
    assert "fix/act-energy-whole-machine" in reasons and all(h.unattributed_above_rest_w is None for h in rep.hosts)
    assert "per-card watts began 2026-10-29" in reasons


def test_rest_body_yaml_overrides_the_defaults(tmp_path):
    (tmp_path / "heat_sensitivity.yaml").write_text(Path(HEAT_CFG).read_text())
    (tmp_path / "rest_body.yaml").write_text(
        "version: 1\nhosts:\n  circe: {rest_w: 500, measured: '2026-10-01..2026-10-13', method: 'p05 chassis_watts', "
        "approved_rest_w: 541}\n  athena: {rest_w: 400, measured: '2026-10-01..2026-10-13', approved_rest_w: 340}\n"
        "sponsored: []\n")
    pool = FakePool(_hosts_rows())
    deps = pr.ReportDeps(pool=pool, heat_sensitivity_path=str(tmp_path / "heat_sensitivity.yaml"), gpu_pool_path=POOL_CFG)
    rep = asyncio.run(pr.build_report(deps, pr.parse_query("24h", "hosts", None, None, None), now=NOW))
    c = next(h for h in rep.hosts if h.host == "circe")
    # rest_w is the value in use (lowering is always allowed); approved_rest_w is only a ceiling
    assert c.chassis_w_rest == 500.0 and c.rest_basis.startswith("pinned in config/physiology/rest_body.yaml")
    a = next(h for h in rep.hosts if h.host == "athena")
    assert a.chassis_w_rest == 340.0
    assert any("athena: rest_w 400 W is above Juniper's approved 340 W" in r for r in _reasons(rep, "hosts"))
    assert a.sponsored == [] and any(x.name.startswith("zelda-dojo") for x in a.top_containers)
    assert pr.rest_window(pr.DEFAULT_REST_BODY) == ("2026-10-01", "2026-10-14")


def test_hosts_holder_read_failure_keeps_the_machines():
    rep = _build(_hosts_rows(), q=pr.parse_query("24h", "hosts", None, None, None), fail={"durable_run_workflow"})
    assert rep.hosts and rep.hosts[0].chassis_w_now == 1150.0 and rep.hosts[0].cards[1].top_holders == []
    assert any(r.startswith("read failed (hosts_holders)") for r in _reasons(rep, "hosts"))


def test_fold_holder_twin_and_census_names():
    assert pr.fold_holder("durable-runs:reading-ad8cac98-4dee-56fd-be1b-a7c1636534a5") == "durable-runs:reading"
    assert pr.fold_holder("durable-runs:d590a9d48af2") == "durable-runs:run"
    assert pr.fold_holder("durable-runs:20261029T220907Z-45bc34") == "durable-runs:run"
    assert pr.fold_holder("durable-runs:curiosity-compaction:weekly:2026-10-12:r0") == "durable-runs:curiosity-compaction:weekly"
    assert pr.fold_holder("durable-runs:c8363ab11231", "curiosity.investigate") == "durable-runs:curiosity.investigate"
    assert pr.fold_holder("vision-council") == "vision-council"
    raw = {b"gpu-lane-controller@circe": json.dumps({"details": {"container_restarts": {"counts": {
               "orion-circe-atlas-llamacpp-fast@d590a9d48af2": 0, "bad-key": 1}}}}).encode(),
           "biometrics@circe": json.dumps({"details": {}}), "x": "not json container_restarts"}
    assert pr.container_names_from_census(raw) == {"d590a9d48af2": "orion-circe-atlas-llamacpp-fast"}
    assert pr.container_label("d590a9d48af2", {"d590a9d48af2": "n"}) == "n"
    assert pr.container_label("unknown12345", {}) == "unknown12345"
