"""Rhythm grading (app/vision_rhythm.py) against a REAL Postgres: a parked
sighting inside the arrival window is ``unscorable``, never ``missed`` (#3388).

``ORION_TEST_PG_URI`` must be a throwaway database: these tests TRUNCATE the
walkway tables. Migrations: see tests/test_vision_handoff_pg.py.
"""

from __future__ import annotations

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

from app import vision_rhythm as vr  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
pytestmark = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

WS = datetime(2026, 11, 14, 1, 0, tzinfo=timezone.utc)
WE = WS + timedelta(minutes=60)
NOW = WE + timedelta(minutes=30)
TABLES = ("vision_percept_expectation", "vision_rhythm_cursor", "vision_individual_sighting", "vision_individual",
          "vision_individuals_cursor", "vision_scene_inventory", "vision_events")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vision_events (event_id VARCHAR PRIMARY KEY, event_type VARCHAR, narrative TEXT, "
            "entities JSONB, tags JSONB, confidence DOUBLE PRECISION, salience DOUBLE PRECISION, evidence_refs JSONB, "
            "correlation_id VARCHAR, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), stream_id TEXT)"))
        conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " CASCADE"))
        # The individuals reducer has processed everything through the window end.
        conn.execute(text("INSERT INTO vision_individuals_cursor (stream_id, last_created_at) VALUES ('walkway', :t)"),
                     {"t": NOW})
        # The camera watched the whole window (one census row every 20 s).
        t = WS - timedelta(seconds=20)
        i = 0
        while t <= WE + timedelta(seconds=20):
            conn.execute(text(
                "INSERT INTO vision_scene_inventory (window_id, stream_id, observed_at, window_start_ts, window_end_ts, "
                "frame_count) VALUES (:w, 'walkway', :t, :a, :b, 1)"),
                {"w": f"w{i}", "t": t, "a": t.timestamp(), "b": t.timestamp() + 5})
            t += timedelta(seconds=20)
            i += 1
    yield engine
    engine.dispose()


def _subject(conn, iid: str, sightings) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, first_seen_at, last_seen_at) "
        "VALUES (:i, 'walkway', 'vehicle', :c, :t, :t)"), {"i": iid, "c": [1.0, 0.0], "t": WS})
    for n, (start, end, stationary) in enumerate(sightings):
        conn.execute(text(
            "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
            "stationary) VALUES (:s, :i, 'walkway', :a, :b, :st)"),
            {"s": f"{iid}-{n}", "i": iid, "a": start, "b": end, "st": stationary})
    conn.execute(text(
        "INSERT INTO vision_percept_expectation (expectation_id, stream_id, subject_key, subject_label, day_kind, "
        "window_start, window_end, peak_minute, support_days, support_sightings, confidence, status, emitted_at) "
        "VALUES (:e, 'walkway', :k, 'a vehicle', 'any', :ws, :we, 0, 7, 9, 0.8, 'open', :t)"),
        {"e": f"exp-{iid}", "k": f"individual:{iid}", "ws": WS, "we": WE, "t": WS - timedelta(hours=1)})


def test_cycle_grades_parked_as_unscorable_a_true_absence_as_missed_and_an_arrival_as_met(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _subject(conn, "parked", [(WS - timedelta(hours=3), WE + timedelta(hours=2), True)])
        _subject(conn, "absent", [])
        _subject(conn, "arrived", [(WS + timedelta(minutes=10), WS + timedelta(minutes=11), False)])
        # Parked before the window, then drove in during it: an arrival.
        _subject(conn, "both", [(WS - timedelta(hours=3), WS + timedelta(minutes=5), True),
                                (WS + timedelta(minutes=20), WS + timedelta(minutes=21), False)])
    summary = vr.run_one_rhythm_cycle(postgres_uri=PG, cfg=vr.RhythmConfig(), now=NOW)
    assert (summary["met"], summary["missed"], summary["unscorable"], summary["parked_in_window"]) == (2, 1, 1, 1)

    with pg.connect() as conn:
        status = dict(conn.execute(text(
            "SELECT subject_key, status FROM vision_percept_expectation")).fetchall())
        events = dict(conn.execute(text("SELECT event_id, event_type FROM vision_events")).fetchall())
    assert status == {"individual:parked": "unscorable", "individual:absent": "missed",
                      "individual:arrived": "met", "individual:both": "met"}
    # No "it did not come" row is written about the parked car.
    assert events == {"rhythm-exp-absent": "expected_absent", "rhythm-exp-arrived": "arrived_as_expected",
                      "rhythm-exp-both": "arrived_as_expected"}
