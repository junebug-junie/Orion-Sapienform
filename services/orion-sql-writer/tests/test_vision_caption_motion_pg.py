"""Caption motion cross-check (app/vision_caption_motion.py) against a REAL Postgres.

Proves the SQL the unit tests fake: the jsonb ``?|`` verdict filter with a
``text[]`` bind, the skip list, the ``||`` tag append that keeps the row's
own tags, and the "only a row with no verdict yet" guard.

``ORION_TEST_PG_URI`` must be a throwaway database: these tests TRUNCATE the
walkway tables. Setup: see tests/test_vision_handoff_pg.py (same migrations).
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

from app import vision_caption_motion as cm  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
pytestmark = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

T0 = datetime(2026, 10, 19, 17, 12, 0, tzinfo=timezone.utc)
FRAME = T0 - timedelta(hours=1)
CAR = (1236.8, 468.1, 1339.9, 539.6)
OWN_TAGS = ["camera_caption", "motion_unverified", "detector:person"]


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vision_events (event_id VARCHAR PRIMARY KEY, event_type VARCHAR, narrative TEXT, "
            "entities JSONB, tags JSONB, confidence DOUBLE PRECISION, salience DOUBLE PRECISION, evidence_refs JSONB, "
            "correlation_id VARCHAR, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), stream_id TEXT)"))
        conn.execute(text("TRUNCATE vision_events, vision_motion_story, vision_motion_story_cursor, "
                          "vision_crop_observation CASCADE"))
        conn.execute(text("INSERT INTO vision_motion_story_cursor (stream_id, processed_until) VALUES ('walkway', :p)"),
                     {"p": T0})
    cm._final_rows.clear()
    yield engine
    cm._final_rows.clear()
    engine.dispose()


def _caption(conn, event_id, narrative, created_at):
    from sqlalchemy import text

    refs = ["artifact:a", f"frame_ts:{FRAME.isoformat()}", "box:person:980,713,1095,967",
            "crop:1237,468,1340,540"]
    conn.execute(text(
        "INSERT INTO vision_events (event_id, event_type, narrative, entities, tags, evidence_refs, created_at, stream_id) "
        "VALUES (:id, 'camera_caption', :n, '[]'::jsonb, CAST(:tags AS jsonb), CAST(:refs AS jsonb), :c, 'walkway')"),
        {"id": event_id, "n": narrative, "tags": json.dumps(OWN_TAGS), "refs": json.dumps(refs), "c": created_at})


def _car_in_story(conn, speed_class, drop_reason, offsets=(-10, -5, 0, 5, 10)):
    """The caption's car in a story whose boxes sit still around the frame
    (``offsets``: seconds from the frame of each box)."""
    from sqlalchemy import text

    evidence = [{"crop_id": "car" if dt == 0 else f"car{dt}", "t": (FRAME + timedelta(seconds=dt)).isoformat(),
                 "label": "vehicle", "box": list(CAR), "clipped": False} for dt in offsets]

    conn.execute(text(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, box_xyxy) "
        "VALUES ('car', 'obs', 'walkway', :t, 'vehicle', 0.7, :b)"), {"t": FRAME, "b": list(CAR)})
    conn.execute(text(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "dir_lateral, dir_depth, zone_path, individual_id, published, crop_ids, evidence, rules_version, "
        "speed_class, drop_reason) VALUES ('s1', 'walkway', 'vehicle', :a, :b, 3, 0, 0, :zp, NULL, false, "
        "ARRAY['car'], CAST(:ev AS jsonb), 1, :sc, :dr)"),
        {"a": FRAME - timedelta(seconds=60), "b": FRAME + timedelta(seconds=60), "zp": ["walkway"],
         "ev": json.dumps(evidence), "sc": speed_class, "dr": drop_reason})


def _tags(pg, event_id):
    from sqlalchemy import text

    with pg.connect() as conn:
        return conn.execute(text("SELECT tags FROM vision_events WHERE event_id = :id"), {"id": event_id}).scalar()


def _step(pg):
    return cm.caption_motion_step(pg, "walkway", T0, lookback_sec=21600, settle_sec=911, min_iou=0.5)


def test_contradicted_tags_are_appended_once_and_keep_the_rows_own_tags(pg) -> None:
    with pg.begin() as conn:
        _caption(conn, "e-drive", "A car is driving down a residential street.", FRAME + timedelta(seconds=4))
        _caption(conn, "e-still", "A car is parked on the street.", FRAME + timedelta(seconds=5))
        _car_in_story(conn, "parked", "prop")
    first = _step(pg)
    assert first["tagged"] == 1 and first["contradicted"] == 1 and first["no_claim"] == 1
    assert _tags(pg, "e-drive") == [*OWN_TAGS, "motion_contradicted", "motion_check:vehicle:driving:parked"]
    assert _tags(pg, "e-still") == OWN_TAGS
    # Second tick: the tagged row is filtered by the jsonb verdict check, the
    # no-claim row by the skip list. Nothing is read or written again.
    second = _step(pg)
    assert second["checked"] == 0 and second["tagged"] == 0
    assert _tags(pg, "e-drive") == [*OWN_TAGS, "motion_contradicted", "motion_check:vehicle:driving:parked"]


def test_update_guard_never_adds_a_second_verdict(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _caption(conn, "e", "A car is driving.", FRAME)
        conn.execute(text("UPDATE vision_events SET tags = tags || '[\"motion_supported\"]'::jsonb WHERE event_id='e'"))
        n = conn.execute(text(cm.ADD_TAGS_SQL), {"id": "e", "add": json.dumps(["motion_contradicted"]),
                                                 "verdicts": list(cm.VERDICT_TAGS)}).rowcount
    assert n == 0
    assert _tags(pg, "e") == [*OWN_TAGS, "motion_supported"]


def test_thin_local_evidence_leaves_the_row_untagged(pg) -> None:
    with pg.begin() as conn:
        _caption(conn, "e", "A car is driving.", FRAME)
        _car_in_story(conn, "parked", "glimpse", offsets=(0, 5))
    counts = _step(pg)
    assert counts["undecided"] == 1 and counts["tagged"] == 0
    assert _tags(pg, "e") == OWN_TAGS
