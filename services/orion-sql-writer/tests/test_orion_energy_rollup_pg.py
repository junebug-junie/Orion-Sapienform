"""Orion energy hourly rollup (app/orion_energy_rollup.py) against a REAL Postgres.

``ORION_TEST_PG_URI`` must be a throwaway database: these tests DROP the raw watt tables.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55511:5432 postgres:16-alpine
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55511/postgres pytest tests/test_orion_energy_rollup_pg.py
"""

from __future__ import annotations

import json
import os
import sys
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

H0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    from app.models.orion_energy_hourly import OrionEnergyHourlySQL

    engine = create_engine(PG)
    with engine.begin() as conn:
        for table in ("orion_biometrics_cluster", "home_cooling_sample", "orion_energy_hourly"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        # Only the columns the rollup reads; the real tables are wider.
        conn.execute(text(
            "CREATE TABLE orion_biometrics_cluster (observed_at TIMESTAMPTZ NOT NULL, measurements_by_node JSONB)"))
        conn.execute(text(
            "CREATE TABLE home_cooling_sample (ts TIMESTAMPTZ NOT NULL, zwave_node_id INT NOT NULL, "
            "cooling_watts DOUBLE PRECISION, stale BOOLEAN, device_online BOOLEAN NOT NULL)"))
    OrionEnergyHourlySQL.__table__.create(engine)
    yield engine
    engine.dispose()


def nodes(engine, start: datetime, end: datetime, watts: dict, step=15) -> None:
    from sqlalchemy import text

    rows, t = [], start
    while t < end:
        rows.append({"t": t, "m": json.dumps({k: {"chassis_watts": v} for k, v in watts.items()})})
        t += timedelta(seconds=step)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orion_biometrics_cluster VALUES (:t, CAST(:m AS JSONB))"), rows)


def cooling(engine, start: datetime, end: datetime, watts: float, stale=False, step=5) -> None:
    from sqlalchemy import text

    rows, t = [], start
    while t < end:
        rows.append({"t": t, "w": watts, "s": stale})
        t += timedelta(seconds=step)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO home_cooling_sample VALUES (:t, 7, :w, :s, true)"), rows)


def hourly(engine) -> dict:
    from sqlalchemy import text

    with engine.begin() as conn:
        rows = conn.execute(text("SELECT hour_start, source, kwh, covered_sec FROM orion_energy_hourly")).all()
    return {(r.hour_start, r.source): (r.kwh, r.covered_sec) for r in rows}


def test_full_hour_integrates_watts_per_source(pg):
    from app.orion_energy_rollup import run_one_rollup

    nodes(pg, H0, H0 + timedelta(hours=1, minutes=1), {"athena": 400.0, "circe": 900.0})
    cooling(pg, H0, H0 + timedelta(hours=1, minutes=1), 1200.0)
    run_one_rollup(pg)
    got = hourly(pg)
    assert got[(H0, "athena")] == pytest.approx((0.4, 3600.0))
    assert got[(H0, "circe")] == pytest.approx((0.9, 3600.0))
    assert got[(H0, "cooling")] == pytest.approx((1.2, 3600.0))


def test_gap_is_left_uncovered_not_filled_in(pg):
    from app.orion_energy_rollup import run_one_rollup

    # 20 minutes of silence mid-hour: only the 60 s hold after the last sample counts.
    nodes(pg, H0, H0 + timedelta(minutes=20), {"athena": 600.0})
    nodes(pg, H0 + timedelta(minutes=40), H0 + timedelta(hours=1, minutes=1), {"athena": 600.0})
    run_one_rollup(pg)
    kwh, covered = hourly(pg)[(H0, "athena")]
    assert covered == pytest.approx(40 * 60 + 60 - 15)
    assert kwh == pytest.approx(600.0 * covered / 3_600_000.0)


def test_stale_and_absent_cooling_is_unknown_not_zero(pg):
    from app.orion_energy_rollup import run_one_rollup

    cooling(pg, H0, H0 + timedelta(hours=1), 1000.0, stale=True)
    nodes(pg, H0, H0 + timedelta(hours=1, minutes=1), {"hecate": 96.0})
    run_one_rollup(pg)
    got = hourly(pg)
    assert (H0, "cooling") not in got
    assert got[(H0, "hecate")][0] == pytest.approx(0.096)


def test_empty_table_backfills_old_hours_then_recomputes_recent_only(pg):
    from app.orion_energy_rollup import run_one_rollup

    old = H0 - timedelta(days=20)
    nodes(pg, old, old + timedelta(hours=1, minutes=1), {"athena": 300.0})
    nodes(pg, H0, H0 + timedelta(hours=1, minutes=1), {"athena": 300.0})
    run_one_rollup(pg)
    assert set(hourly(pg)) == {(old, "athena"), (old + timedelta(hours=1), "athena"),
                               (H0, "athena"), (H0 + timedelta(hours=1), "athena")}
    # Second pass is idempotent.
    before = hourly(pg)
    run_one_rollup(pg)
    assert hourly(pg) == pytest.approx(before)


def test_future_stamped_sample_does_not_freeze_the_rollup(pg):
    from datetime import datetime as dt

    from app.orion_energy_rollup import run_one_rollup

    # One sample from a node with a clock a day ahead.
    future = dt.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    nodes(pg, future, future + timedelta(seconds=30), {"athena": 400.0})
    run_one_rollup(pg)
    # Real samples arrive afterwards for the last two hours and must still be rolled up.
    now_hour = dt.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    nodes(pg, now_hour - timedelta(hours=2), now_hour - timedelta(hours=1), {"circe": 900.0})
    run_one_rollup(pg)
    assert (now_hour - timedelta(hours=2), "circe") in hourly(pg)


def test_pruned_raw_never_overwrites_a_better_row(pg):
    from sqlalchemy import text

    from app.orion_energy_rollup import run_one_rollup

    nodes(pg, H0, H0 + timedelta(hours=1, minutes=1), {"athena": 500.0})
    run_one_rollup(pg)
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM orion_biometrics_cluster WHERE observed_at < :t"),
                     {"t": H0 + timedelta(minutes=30)})
    run_one_rollup(pg)
    assert hourly(pg)[(H0, "athena")] == pytest.approx((0.5, 3600.0))
