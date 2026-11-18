"""DETECTION_RATE_SQL against REAL Postgres (``ORION_TEST_PG_URI``, a throwaway
database: it TRUNCATEs vision_scene_inventory). CI runs it in the sql-writer
Postgres job, which applies the vision_scene_inventory migration and refuses
a skip."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from orion import vision_detection_rate_watch as dr

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

# Three days after the 2026-11-01 DST change in America/Denver: d=3 crosses it.
END = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    import psycopg2

    c = psycopg2.connect(PG)
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute("TRUNCATE vision_scene_inventory")
    yield c
    c.close()


def _win(cur, wid, stream, t, counts):
    cur.execute(
        "INSERT INTO vision_scene_inventory (window_id, stream_id, observed_at, frame_count, counts) "
        "VALUES (%s, %s, %s, 1, %s::jsonb)", [wid, stream, t, json.dumps(counts)])


@needs_pg
def test_sql_counts_each_day_span_with_half_open_bounds(conn) -> None:
    span = dr.DEFAULT_CONFIG.span
    with conn.cursor() as cur:
        # Session time zone with DST: offsets must still be exact 24 h.
        cur.execute("SET TIME ZONE 'America/Denver'")
        _win(cur, "now-in", "walkway", END - timedelta(minutes=5), {"vehicle": 1})
        _win(cur, "now-empty", "walkway", END - timedelta(hours=1), {})
        _win(cur, "now-edge-end", "walkway", END, {"person": 1})              # <= end: in
        _win(cur, "now-edge-start", "walkway", END - span, {"person": 1})      # > start: out
        _win(cur, "after", "walkway", END + timedelta(seconds=1), {"x": 1})    # future: out
        _win(cur, "d1", "walkway", END - timedelta(days=1, minutes=10), {"vehicle": 2})
        _win(cur, "d3a", "walkway", END - timedelta(days=3, minutes=10), {})
        _win(cur, "d3b", "walkway", END - timedelta(days=3, minutes=20), {"dog": 1})
        _win(cur, "other", "cam0", END - timedelta(minutes=5), {"desk": 1})    # other stream: out
        cur.execute(dr.DETECTION_RATE_SQL, dr.sql_params("walkway", END))
        rows = cur.fetchall()
    assert [r[0] for r in rows] == list(range(dr.DEFAULT_CONFIG.baseline_days + 1))
    current, baseline = dr.split_rows(rows)
    assert current == dr.SpanCount(3, 2)
    assert baseline[0] == dr.SpanCount(1, 1)
    assert baseline[1] == dr.SpanCount(0, 0)
    assert baseline[2] == dr.SpanCount(2, 1)
