"""Hourly frame index summary (app/vision_hourly_frames.db_summarizer) against
REAL Postgres (``ORION_TEST_PG_URI``, a throwaway database: it TRUNCATEs the
walkway tables). CI applies the walkway migrations and refuses a skip."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from app.vision_hourly_frames import db_summarizer

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

T = datetime(2026, 10, 22, 18, 0, 0, 123000, tzinfo=timezone.utc)


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE vision_scene_inventory, vision_crop_observation CASCADE"))
    yield engine
    engine.dispose()


def _window(conn, wid, t, counts):
    from sqlalchemy import text
    import json

    conn.execute(text(
        "INSERT INTO vision_scene_inventory (window_id, stream_id, camera_id, observed_at, frame_count, counts) "
        "VALUES (:w, 'walkway', 'walkway', :t, 2, CAST(:c AS jsonb))"), {"w": wid, "t": t, "c": json.dumps(counts)})


def _crop(conn, cid, t, label, score):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, camera_id, observed_at, "
        "label, score, box_xyxy) VALUES (:c, :c, 'walkway', 'walkway', :t, :l, :s, ARRAY[0,0,10,10]::real[])"),
        {"c": cid, "t": t, "l": label, "s": score})


@needs_pg
def test_summary_counts_the_span_before_the_frame(pg) -> None:
    with pg.begin() as conn:
        _window(conn, "w1", T - timedelta(minutes=1), {"vehicle": 1})
        _window(conn, "w2", T - timedelta(minutes=2), {})
        _window(conn, "w3", T - timedelta(minutes=3), {"vehicle": 2, "person": 1})
        _window(conn, "old", T - timedelta(minutes=20), {"vehicle": 1})   # outside the 10 min span
        _window(conn, "after", T + timedelta(minutes=1), {"vehicle": 1})  # after the frame
        _crop(conn, "c1", T, "vehicle", 0.36)                              # this exact frame
        _crop(conn, "c2", T - timedelta(minutes=3), "vehicle", 0.40)
        _crop(conn, "c3", T - timedelta(minutes=3), "person", 0.45)
        _crop(conn, "c4", T - timedelta(minutes=30), "vehicle", 0.9)      # outside the span
    out = db_summarizer(pg, 600.0)("walkway", T)
    assert out["windows"] == 3
    assert out["windows_with_detection"] == 2
    assert out["detector_frames"] == 6
    assert out["label_windows"] == {"vehicle": 2, "person": 1}
    assert out["crop_scores"]["vehicle"] == {"n": 2, "p50": 0.38, "max": 0.4}
    assert out["crop_scores"]["person"]["n"] == 1
    assert out["frame_crops"] == [{"label": "vehicle", "score": 0.36}]


@needs_pg
def test_summary_of_an_empty_span_is_zero_not_error(pg) -> None:
    out = db_summarizer(pg, 600.0)("walkway", T)
    assert out["windows"] == 0 and out["windows_with_detection"] == 0
    assert out["label_windows"] == {} and out["crop_scores"] == {} and out["frame_crops"] == []
