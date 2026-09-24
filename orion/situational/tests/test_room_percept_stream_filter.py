"""The room percept never comes from the walkway camera.

`vision_events` is shared: the council narrates the walkway (street and
patio) and the walkway reducers write `arrived_as_expected` /
`expected_absent` / `attention_worthy` rows into it. Real SQL on SQLite (the
reader's statement is written to run on both), not a mocked cursor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text

from orion.situational import perception_reader


def _engine(rows):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vision_events (event_id TEXT PRIMARY KEY, narrative TEXT, "
            "stream_id TEXT, created_at TIMESTAMP)"))
        for i, (narrative, stream_id, created_at) in enumerate(rows):
            conn.execute(text("INSERT INTO vision_events VALUES (:i, :n, :s, :c)"),
                         {"i": str(i), "n": narrative, "s": stream_id, "c": created_at})
    return engine


def test_newer_walkway_row_is_not_the_room_percept(monkeypatch):
    now = datetime.now(timezone.utc)
    engine = _engine([
        ("A mug on the desk.", "cam0", now - timedelta(minutes=3)),
        ("Two people sit on the patio.", "walkway", now - timedelta(seconds=10)),
        ("The usual van did not come.", "walkway", now - timedelta(seconds=5)),
    ])
    monkeypatch.setattr(perception_reader, "_get_engine", lambda: engine)
    percept = perception_reader.fetch_latest_percept(stream_ids=["carbon", "cam0"])
    assert percept is not None
    assert percept["scene_summary"] == "A mug on the desk."
    assert percept["observed_at"].tzinfo is not None


def test_legacy_rows_without_stream_are_still_read(monkeypatch):
    now = datetime.now(timezone.utc)
    engine = _engine([
        ("Legacy room narrative.", None, now - timedelta(seconds=30)),
        ("Patio.", "walkway", now - timedelta(seconds=1)),
    ])
    monkeypatch.setattr(perception_reader, "_get_engine", lambda: engine)
    got = perception_reader.fetch_latest_percept(stream_ids=["cam0"], legacy_cutoff=now)
    assert got["scene_summary"] == "Legacy room narrative."


def test_null_stream_row_after_the_legacy_cutoff_is_refused(monkeypatch):
    # A not-rebuilt scribe drops stream_id, so a walkway narrative lands NULL.
    now = datetime.now(timezone.utc)
    engine = _engine([
        ("A mug on the desk.", "cam0", now - timedelta(minutes=3)),
        ("Two people sit on the patio.", None, now - timedelta(seconds=5)),
    ])
    monkeypatch.setattr(perception_reader, "_get_engine", lambda: engine)
    monkeypatch.delenv("ORION_VISION_EVENTS_LEGACY_CUTOFF", raising=False)
    assert perception_reader.fetch_latest_percept(stream_ids=["cam0"])["scene_summary"] == "A mug on the desk."


def test_legacy_cutoff_default_and_override(monkeypatch):
    monkeypatch.delenv("ORION_VISION_EVENTS_LEGACY_CUTOFF", raising=False)
    assert perception_reader.vision_events_legacy_cutoff() == datetime(2026, 9, 24, tzinfo=timezone.utc)
    monkeypatch.setenv("ORION_VISION_EVENTS_LEGACY_CUTOFF", "2026-10-01T00:00:00Z")
    assert perception_reader.vision_events_legacy_cutoff() == datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_only_walkway_rows_means_no_percept(monkeypatch):
    now = datetime.now(timezone.utc)
    engine = _engine([("Patio.", "walkway", now)])
    monkeypatch.setattr(perception_reader, "_get_engine", lambda: engine)
    assert perception_reader.fetch_latest_percept(stream_ids=["cam0"]) is None
    # An empty room-stream list is legacy-only, never "every camera".
    assert perception_reader.fetch_latest_percept(stream_ids=[]) is None
