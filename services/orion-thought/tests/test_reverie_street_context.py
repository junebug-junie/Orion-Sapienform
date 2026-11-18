"""The street (walkway camera) as its own reverie input, 2026-10-15.

Properties: walkway captions and graded walkway predictions reach the reverie
prompt under their own "OUTSIDE, ON THE WALKWAY" heading; they never enter
`recent_percepts` (the room, which the room-only expectation judge reads);
empty means no section; the readers are room/street disjoint and fail open.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1

GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)
_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
HEADING = "OUTSIDE, ON THE WALKWAY"


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
    )


STREET = [{"narrative": "Two vehicles are parked by the curb.", "age_sec": 30.0}]
GRADED = ["On the walkway I expected the black dog around 07:40 (Thu 15 Oct); it did not come."]


def test_no_street_section_when_empty():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast(), street_percepts=[], street_predictions=[])
    assert "street_percepts" not in ctx and "street_predictions" not in ctx
    assert HEADING not in _render(**ctx)


def test_street_section_in_coalition_mode_and_kept_out_of_room_percepts():
    from app import reverie

    room = [{"narrative": "the desk lamp is on", "age_sec": 10.0}]
    ctx = reverie.build_reverie_context(
        _broadcast(), recent_percepts=room, street_percepts=STREET, street_predictions=GRADED
    )
    assert ctx["recent_percepts"] == room
    out = _render(**ctx)
    assert HEADING in out
    assert "just now: Two vehicles are parked by the curb." in out
    assert "it did not come" in out
    assert "about the room, never the street" in out
    # The room block lists only the room.
    room_block = out.split("what you can currently see")[1].split(HEADING)[0]
    assert "vehicles" not in room_block


def test_street_section_in_concern_card_mode_graded_only():
    from app import reverie

    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="x", thread_label="t")
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], street_predictions=GRADED)
    out = _render(**ctx)
    assert HEADING in out and "black dog" in out
    assert "just now:" not in out


@pytest.mark.asyncio
async def test_tick_threads_street_into_its_own_keys(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")
    seen = {}
    monkeypatch.setattr(vision_reader, "read_recent_vision_events",
                        lambda **kw: [{"narrative": "a mug on the desk", "age_sec": 4.0}])

    def _street(**kw):
        seen["captions"] = kw
        return STREET

    def _graded(**kw):
        seen["graded"] = kw
        return GRADED

    monkeypatch.setattr(vision_reader, "read_recent_street_events", _street)
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", _graded)
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert result is not None
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    assert ctx["recent_percepts"] == [{"narrative": "a mug on the desk", "age_sec": 4.0}]
    assert ctx["street_percepts"] == STREET
    assert ctx["street_predictions"] == GRADED
    assert seen["captions"]["stream_ids"] == ["walkway"]
    assert seen["graded"]["stream_ids"] == ["walkway"]


@pytest.mark.asyncio
async def test_tick_skips_street_reads_when_disabled_or_no_streams(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    def boom(**kw):
        raise AssertionError("street reader must not be called when off")

    monkeypatch.setattr(vision_reader, "read_recent_street_events", boom)
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", boom)
    monkeypatch.setattr(settings, "reverie_perception_enabled", False)
    for enabled, ids in ((False, "walkway"), (True, "")):
        monkeypatch.setattr(settings, "reverie_street_enabled", enabled)
        monkeypatch.setattr(settings, "reverie_street_stream_ids", ids)
        cortex = AsyncMock()
        cortex.execute_plan = AsyncMock(return_value={
            "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
        })
        assert await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
        assert "street_percepts" not in cortex.execute_plan.call_args.kwargs["req"].context


@pytest.mark.asyncio
async def test_a_failing_street_read_costs_only_the_street(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")

    def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(vision_reader, "read_recent_street_events", boom)
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", lambda **kw: GRADED)
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    assert await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert "street_percepts" not in cortex.execute_plan.call_args.kwargs["req"].context


# --- the readers themselves, against SQLite ------------------------------------


@pytest.fixture
def sqlite_engine(monkeypatch):
    from sqlalchemy import create_engine, text

    from app import vision_reader

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vision_events (event_id TEXT, stream_id TEXT, event_type TEXT, "
            "narrative TEXT, created_at TIMESTAMP, tags TEXT DEFAULT '[]')"))
        conn.execute(text(
            "CREATE TABLE vision_percept_expectation (expectation_id TEXT, stream_id TEXT, "
            "subject_key TEXT, subject_label TEXT, status TEXT, peak_minute INTEGER, "
            "window_start TIMESTAMP, scored_at TIMESTAMP)"))
    monkeypatch.setattr(vision_reader, "_get_engine", lambda: engine)
    return engine


def _ins(engine, sql, **params):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(sql), params)


def test_street_reader_reads_only_walkway_and_room_reader_never_does(sqlite_engine):
    from app import vision_reader

    now = datetime.now(timezone.utc)
    for sid, et, text_, age in (
        ("walkway", "visual_observation", "A van is parked.", 30),
        ("walkway", "visual_observation", "a van is parked", 40),
        ("walkway", "arrived_as_expected", "The black dog came.", 10),
        ("walkway", "visual_observation", "Old caption.", 4000),
        ("cam0", "visual_observation", "The desk lamp is on.", 20),
    ):
        _ins(sqlite_engine, "INSERT INTO vision_events (event_id, stream_id, event_type, narrative, created_at) VALUES ('e', :s, :t, :n, :c)",
             s=sid, t=et, n=text_, c=now - timedelta(seconds=age))
    street = vision_reader.read_recent_street_events(max_age_sec=900, limit=3, stream_ids=["walkway"])
    assert [s["narrative"] for s in street] == ["A van is parked."]
    room = vision_reader.read_recent_vision_events(max_age_sec=900, limit=5, stream_ids=["cam0"])
    assert [r["narrative"] for r in room] == ["The desk lamp is on."]
    assert vision_reader.read_recent_street_events(max_age_sec=900, limit=3, stream_ids=[]) == []


def test_graded_reader_lines_with_hit_rate(sqlite_engine):
    from app import vision_reader

    now = datetime.now(timezone.utc)
    rows = [("e1", "met", 2), ("e2", "missed", 26), ("e3", "met", 50), ("e4", "unscorable", 1)]
    for eid, status, hours in rows:
        _ins(sqlite_engine,
             "INSERT INTO vision_percept_expectation VALUES (:e, 'walkway', 'individual:dog', "
             "'the black dog', :st, 460, :ws, :sa)",
             e=eid, st=status, ws=now - timedelta(hours=hours, minutes=30), sa=now - timedelta(hours=hours))
    lines = vision_reader.read_recent_street_graded(
        stream_ids=["walkway"], lookback_hours=24, limit=4, tz_name="America/Denver")
    assert len(lines) == 1
    assert "the black dog around 07:40" in lines[0] and "it came." in lines[0]
    assert "Over the last 14 days it came 2 of 3 times." in lines[0]


def test_graded_reader_empty_and_failing_are_nothing(sqlite_engine, monkeypatch):
    from app import vision_reader

    assert vision_reader.read_recent_street_graded(
        stream_ids=["walkway"], lookback_hours=24, limit=4, tz_name="America/Denver") == []

    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(vision_reader, "_get_engine", _boom)
    assert vision_reader.read_recent_street_graded(
        stream_ids=["walkway"], lookback_hours=24, limit=4, tz_name="America/Denver") == []
    assert vision_reader.read_recent_street_events(max_age_sec=900, limit=2, stream_ids=["walkway"]) == []


def test_reducer_rows_are_excluded_before_the_row_limit(sqlite_engine):
    """A burst of reducer rows (attention_worthy, ...) newer than every caption
    must not fill the fetch window and hide the real caption."""
    from app import vision_reader

    now = datetime.now(timezone.utc)
    for i in range(vision_reader._STREET_FETCH_ROWS + 5):
        _ins(sqlite_engine, "INSERT INTO vision_events (event_id, stream_id, event_type, narrative, created_at) VALUES ('r', 'walkway', 'attention_worthy', :n, :c)",
             n=f"worth a look {i}", c=now - timedelta(seconds=1 + i * 0.01))
    _ins(sqlite_engine, "INSERT INTO vision_events (event_id, stream_id, event_type, narrative, created_at) VALUES ('e', 'walkway', 'visual_observation', "
         "'A cyclist rides past.', :c)", c=now - timedelta(seconds=60))
    got = vision_reader.read_recent_street_events(max_age_sec=900, limit=2, stream_ids=["walkway"])
    assert [g["narrative"] for g in got] == ["A cyclist rides past."]


def test_reverie_reads_a_camera_caption_labelled_not_as_fact(sqlite_engine):
    """orion-vision-window's camera_caption rows (2026-10-16) reach reverie
    labelled as a caption of one frame, with the detector boxes behind it."""
    from app import vision_reader

    now = datetime.now(timezone.utc)
    _ins(sqlite_engine,
         "INSERT INTO vision_events (event_id, stream_id, event_type, narrative, created_at, tags) "
         "VALUES ('c', 'walkway', 'camera_caption', 'A person is pushing a stroller.', :c, "
         "'[\"camera_caption\", \"motion_unverified\", \"detector:person\", \"detector:stroller\"]')",
         c=now - timedelta(seconds=30))
    (got,) = vision_reader.read_recent_street_events(max_age_sec=900, limit=2, stream_ids=["walkway"])
    assert got["narrative"] == (
        'camera caption (one frame; detector saw person, stroller; motion not verified): '
        '"A person is pushing a stroller."'
    )


@pytest.mark.asyncio
async def test_expectation_is_dropped_when_only_the_street_was_offered(monkeypatch):
    """The expectation is judged against a ROOM percept. Street-only context
    (room stale) must not produce a checkable claim."""
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", True)
    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")
    monkeypatch.setattr(vision_reader, "read_recent_vision_events", lambda **kw: [])
    monkeypatch.setattr(vision_reader, "read_recent_street_events", lambda **kw: STREET)
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", lambda **kw: [])
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps({
        "interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"],
        "expectation": "The vehicles will still be parked by the curb."})})
    thought = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert thought is not None
    assert thought.expectation is None and thought.expectation_checkable_by is None
    out = _render(**cortex.execute_plan.call_args.kwargs["req"].context)
    assert "never the street" not in out  # no room block -> no invitation to predict


@pytest.mark.asyncio
async def test_expectation_kept_when_the_room_was_offered(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", True)
    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_enabled", False)
    monkeypatch.setattr(vision_reader, "read_recent_vision_events",
                        lambda **kw: [{"narrative": "the desk lamp is on", "age_sec": 3.0}])
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps({
        "interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"],
        "expectation": "The desk lamp will still be on."})})
    thought = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert thought.expectation == "The desk lamp will still be on."
    assert thought.expectation_checkable_by is not None


def test_street_section_names_orions_own_outdoor_camera(monkeypatch):
    """2026-10-15: the heading says which eye this is -- Orion's own outdoor
    camera and what it looks over -- from config, in both prompt branches."""
    from app import reverie

    monkeypatch.delenv("ORION_CAMERA_DESCRIPTIONS", raising=False)
    for kwargs in ({}, {"concern_cards": None}):
        ctx = reverie.build_reverie_context(_broadcast(), street_percepts=STREET, **kwargs)
        assert ctx["street_camera"] == (
            "your outdoor camera (walkway), looking over the community walkway, the street and our patio"
        )
        text = _render(**ctx)
        assert (
            "OUTSIDE, ON THE WALKWAY (your outdoor camera (walkway), looking over the community walkway,"
            " the street and our patio -- your own eye outside the house, public, NOT your room):"
        ) in text
    quiet = reverie.build_reverie_context(_broadcast(), street_percepts=[], street_predictions=[])
    assert "street_camera" not in quiet


def test_street_camera_label_reads_config(monkeypatch):
    from app import reverie

    monkeypatch.setenv(
        "ORION_CAMERA_DESCRIPTIONS", json.dumps({"porch": {"name": "the porch eye", "view": "the steps"}})
    )
    assert reverie.street_camera_label(["porch"]) == "the porch eye (porch), looking over the steps"
    assert reverie.street_camera_label(["nobody"]) == "camera nobody (nobody)"
