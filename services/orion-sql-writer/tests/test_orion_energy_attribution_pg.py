"""Hourly energy attribution (app/orion_energy_attribution.py) against a REAL Postgres.

``ORION_TEST_PG_URI`` must be a throwaway database: these tests DROP the tables they read.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55511:5432 postgres:16-alpine
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55511/postgres pytest tests/test_orion_energy_attribution_pg.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

PG = os.getenv("ORION_TEST_PG_URI", "")
pytestmark = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

H0 = datetime(2026, 10, 30, 1, 0, tzinfo=timezone.utc)
NOW = H0 + timedelta(hours=1, minutes=10)


@pytest.fixture()
def pg(monkeypatch):
    from sqlalchemy import create_engine, text

    from app.models.orion_energy_attribution_hourly import OrionEnergyAttributionHourlySQL
    from orion.grammar import act_energy_fill as fill
    from orion.physiology.rest_body import parse_rest_body

    engine = create_engine(PG)
    with engine.begin() as conn:
        for table in ("orion_biometrics_summary", "gpu_pool_events", "grammar_acts",
                      "orion_energy_attribution_hourly"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text('CREATE TABLE orion_biometrics_summary (id VARCHAR PRIMARY KEY, "timestamp" VARCHAR NOT NULL, '
                          "node VARCHAR, measurements JSONB)"))
        conn.execute(text("CREATE TABLE gpu_pool_events (event_id VARCHAR PRIMARY KEY, generated_at TIMESTAMPTZ NOT NULL, "
                          "event VARCHAR NOT NULL, lease_id VARCHAR, cards JSON, turn_correlation_id VARCHAR, "
                          "held_ms DOUBLE PRECISION)"))
        conn.execute(text("CREATE TABLE grammar_acts (act_id VARCHAR PRIMARY KEY, correlation_id VARCHAR)"))
    OrionEnergyAttributionHourlySQL.__table__.create(engine)
    rest = parse_rest_body({"version": 1, "hosts": {"circe": {"rest_w": 600, "approved_rest_w": 600},
                                                    "athena": {"rest_w": 300, "approved_rest_w": 300}},
                            "sponsored": [{"name": "dojo", "containers": "zelda-dojo-live-*", "host": "athena"}]})
    fill._FLOOR_CACHE.clear()
    monkeypatch.setattr(fill, "rest_body_config", lambda: (rest, None))
    monkeypatch.setattr(fill, "_card_locations", lambda: ({"gpu1": ("circe", 1), "gpu2": ("circe", 2)}, None))
    yield engine
    engine.dispose()


def _summary(conn, node, at, m):
    from sqlalchemy import text

    conn.execute(text('INSERT INTO orion_biometrics_summary VALUES (:i, :t, :n, CAST(:m AS jsonb))'),
                 {"i": uuid.uuid4().hex, "t": at.strftime("%Y-%m-%d %H:%M:%S.%f+00"), "n": node, "m": json.dumps(m)})


def _lease(conn, lease, card, start, end, corr):
    from sqlalchemy import text

    conn.execute(text("INSERT INTO gpu_pool_events VALUES (:e, :g, 'released', :l, CAST(:c AS json), :corr, :h)"),
                 {"e": uuid.uuid4().hex, "g": end, "l": lease, "c": json.dumps([card]), "corr": corr,
                  "h": (end - start).total_seconds() * 1000.0})


def _seed(engine):
    from sqlalchemy import text

    with engine.begin() as conn:
        day0 = H0.replace(hour=0)
        for i in range(120):  # the day before: card idle floors (gpu1 30 W, gpu2 40 W)
            _summary(conn, "circe", day0 - timedelta(hours=2) + timedelta(seconds=30 * i),
                     {"gpu1_watts": 30.0, "gpu2_watts": 40.0})
        for i in range(-4, 125):
            t = H0 + timedelta(seconds=30 * i)
            # circe: gpu2 100 W above idle, CPU 50 W above its fitted idle, chassis 300 W above rest.
            _summary(conn, "circe", t, {"chassis_watts": 900.0, "cpu_watts_total": 150.0, "cpu_idle_watts": 100.0,
                                        "gpu1_watts": 30.0, "gpu2_watts": 140.0})
            # athena: no pool cards; the dojo holds 40 of 80 W above CPU idle; chassis 160 W above rest.
            _summary(conn, "athena", t, {"chassis_watts": 460.0, "cpu_watts_total": 180.0, "cpu_idle_watts": 100.0,
                                         "container_cpu_watts.zelda-dojo-live-3": 30.0,
                                         "container_cpu_watts.zelda-dojo-live-9": 10.0,
                                         "container_cpu_watts.orion-athena-sql-writer": 25.0})
        # gpu2: an act's lease for the first half hour, a turn with no act for the second.
        _lease(conn, "L1", "gpu2", H0, H0 + timedelta(minutes=30), "corr-act")
        _lease(conn, "L2", "gpu2", H0 + timedelta(minutes=30), H0 + timedelta(hours=1), "corr-noact")
        conn.execute(text("INSERT INTO grammar_acts VALUES ('a1', 'corr-act')"))


def test_hour_rows_close_on_the_measured_chassis_energy(pg):
    from sqlalchemy import text

    from app.orion_energy_attribution import run_one_attribution

    _seed(pg)
    run_one_attribution(pg, now=NOW)
    with pg.connect() as conn:
        rows = {r.host: r for r in conn.execute(text(
            "SELECT * FROM orion_energy_attribution_hourly WHERE hour_start = :h"), {"h": H0}).fetchall()}
    c = rows["circe"]
    assert c.rest_w == 600.0 and c.rest_wh == pytest.approx(600.0, rel=1e-3)
    assert c.above_rest_wh == pytest.approx(300.0, rel=1e-3)
    # load 150 W (gpu2 100 + CPU 50): gpu2 carries 200 W of the 300; half the hour metered.
    assert c.metered_wh == pytest.approx(100.0, rel=1e-3)
    assert c.leased_unmetered_wh == pytest.approx(100.0, rel=1e-3)
    assert c.unattributed_wh == pytest.approx(200.0, rel=1e-3)  # leased-without-act 100 + CPU 100
    assert c.above_rest_wh == pytest.approx(c.sponsored_wh + c.metered_wh + c.unattributed_wh)
    assert c.split_unknown_sec == 0.0 and "rest_w 600 W" in c.rest_source
    a = rows["athena"]
    # athena: chassis 160 W above rest over 80 W of CPU above idle: the dojo's 40 W carries 80 W.
    assert a.sponsored_wh == pytest.approx(80.0, rel=1e-3) and a.metered_wh == 0.0
    assert a.unattributed_wh == pytest.approx(80.0, rel=1e-3)
    # Hours with no chassis reading have no row (unknown, never 0). The hour before H0 only holds the
    # last 2 min plus the first sample (00:58:00) held back 90 s: 210 s, not a whole hour of zeros.
    with pg.connect() as conn:
        before = conn.execute(text("SELECT hour_start, covered_sec FROM orion_energy_attribution_hourly "
                                   "WHERE hour_start < :h"), {"h": H0}).fetchall()
        hours = {r[0] for r in conn.execute(text("SELECT hour_start FROM orion_energy_attribution_hourly"))}
    assert hours == {H0 - timedelta(hours=1), H0}
    assert before and all(r.covered_sec == pytest.approx(210.0, abs=1.0) for r in before)


def test_negative_unattributed_is_logged_and_stored_unclipped(pg, monkeypatch, caplog):
    from sqlalchemy import text

    from app import orion_energy_attribution as job
    from orion.energy.act_energy import HostAttribution
    from orion.grammar import act_energy_fill as fill

    over = HostAttribution(rest_wh=600.0, above_rest_wh=100.0, sponsored_wh=0.0, metered_wh=130.0,
                           leased_unmetered_wh=0.0, covered_s=3600.0, split_unknown_s=0.0)
    _seed(pg)  # hosts with no chassis reading are skipped before any hour is computed
    monkeypatch.setattr(fill, "attribute_host_hour", lambda conn, host, s, e, now=None: (over, 600.0, "fixture"))
    with caplog.at_level(logging.WARNING, logger="sql-writer.orion_energy_attribution"):
        job.run_one_attribution(pg, now=NOW)
    with pg.connect() as conn:
        vals = [r[0] for r in conn.execute(text("SELECT unattributed_wh FROM orion_energy_attribution_hourly"))]
    assert vals and all(v == pytest.approx(-30.0) for v in vals)
    assert "orion_energy_attribution_negative" in caplog.text and "over-attribution" in caplog.text


def test_a_host_with_no_chassis_reading_is_skipped_and_one_failing_host_does_not_stop_others(pg, monkeypatch, caplog):
    from sqlalchemy import text

    from app import orion_energy_attribution as job
    from orion.grammar import act_energy_fill as fill

    _seed(pg)
    real = fill.attribute_host_hour
    seen = []

    def flaky(conn, host, s, e, now=None):
        seen.append(host)
        if host == "athena":
            raise RuntimeError("boom")
        return real(conn, host, s, e, now=now)
    monkeypatch.setattr(fill, "attribute_host_hour", flaky)
    with caplog.at_level(logging.WARNING, logger="sql-writer.orion_energy_attribution"):
        job.run_one_attribution(pg, now=NOW)
    assert "orion_energy_attribution_host_failed host=athena" in caplog.text
    with pg.connect() as conn:
        assert {r[0] for r in conn.execute(text("SELECT host FROM orion_energy_attribution_hourly"))} == {"circe"}
    # hecate is not in this fixture's rest; a host with rest but no rows never reaches the hour loop
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM orion_biometrics_summary WHERE node = 'circe'"))
        conn.execute(text("DELETE FROM orion_energy_attribution_hourly"))
    seen.clear()
    job.run_one_attribution(pg, now=NOW)
    assert "circe" not in seen
