"""The walkway's day so far as one reverie line, 2026-10-23 (spec PR #2763, item 1).

Properties: the line comes from vision_rollup through walkway_rollup.display_line,
so people are a range and an old rules_version 1 row is re-rendered, never shown
with a bare count; only today's day row (or a complete hour from the last 3 h)
counts; a missing/failing read costs only this line; the trace log says rollup=1.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1
from orion.vision import walkway_rollup as wr

_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
HEADING = "OUTSIDE, ON THE WALKWAY"
GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)
# A live rules_version 2 day line (vision_rollup, 2026-10-23 16:41 UTC).
V2_DAY_LINE = (
    "Walkway Fri 23 Oct 2026 (so far, through 10:00): people: at least 1, at most 2 "
    "(the camera splits one person into several); 2 strollers (2 distinct), 12 vehicles "
    "(8 distinct) passing; named: unsure (09:51); 4 unnamed regulars (seen on another day "
    "this week too); busiest 09:00 (at most 10 passings, usual at most 8.8; people at least 1, "
    "at most 2, most at once 1 stroller), quietest 00:00 (0 passings, usual at most 0.1); "
    "usual day at most 145.6 passings."
)
RANGE = "people: at least 1, at most 2"


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
    )


def _cortex():
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    return cortex


# --- the reader, against SQLite ------------------------------------------------


@pytest.fixture
def rollup_db(monkeypatch):
    from sqlalchemy import create_engine, text

    from app import vision_reader

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vision_rollup (stream_id TEXT, grain TEXT, period_start TIMESTAMP, "
            "label TEXT, complete BOOLEAN, line TEXT, rollup TEXT, computed_at TIMESTAMP)"))
    monkeypatch.setattr(vision_reader, "_get_engine", lambda: engine)
    return engine


def _row(engine, *, grain, period_start, line, rollup, complete=True, stream="walkway"):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO vision_rollup (stream_id, grain, period_start, label, complete, line, rollup, computed_at) "
            "VALUES (:s, :g, :p, 'x', :c, :l, :r, :p)"),
            {"s": stream, "g": grain, "p": period_start, "c": complete, "l": line,
             "r": rollup if isinstance(rollup, str) or rollup is None else json.dumps(rollup)})


# 10:41 MDT on Fri 23 Oct 2026; local midnight is 06:00 UTC.
NOW = datetime(2026, 10, 23, 16, 41, tzinfo=timezone.utc)
TODAY = datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc)


def _read(**kw):
    from app import vision_reader

    args = {"stream_ids": ["walkway", "doorbell"], "tz_name": "America/Denver", "max_chars": 1000, "now": NOW}
    args.update(kw)
    return vision_reader.read_street_day_line(**args)


def test_today_day_row_is_read_with_people_as_a_range(rollup_db):
    _row(rollup_db, grain="day", period_start=TODAY, line=V2_DAY_LINE, rollup={"rules_version": 2})
    got = _read()
    assert got == V2_DAY_LINE
    assert RANGE in got and wr.bare_people_counts(got) == []


def test_old_rules_row_goes_through_display_line_never_its_bare_stored_line(rollup_db, monkeypatch):
    """A rules_version 1 row stored "6 passings (6 people)". The reader must hand the row to
    display_line (which re-renders it as a range) and show what that returns -- never the
    stored line."""
    stored = "Walkway Fri 23 Oct 2026: 6 passings (6 people) by 4 distinct."
    rerendered = "Walkway Fri 23 Oct 2026: people: at least 1, at most 6; 6 passings."
    calls = []

    def _display(line, rollup):
        calls.append((line, json.loads(rollup) if isinstance(rollup, str) else rollup))
        return rerendered

    monkeypatch.setattr(wr, "display_line", _display)
    _row(rollup_db, grain="day", period_start=TODAY, line=stored, rollup={"rules_version": 1})
    got = _read()
    assert calls == [(stored, {"rules_version": 1})]
    assert got == rerendered and wr.bare_people_counts(got) == []
    assert wr.bare_people_counts(stored) == ["6 people"]  # the guard would have caught it


def test_unreadable_rollup_costs_the_line(rollup_db):
    _row(rollup_db, grain="day", period_start=TODAY, line="OLD 79 people", rollup="{not json")
    assert _read() == ""


def test_yesterdays_day_row_and_old_hours_are_not_the_day_so_far(rollup_db):
    _row(rollup_db, grain="day", period_start=TODAY - timedelta(days=1),
         line="Walkway Thu 22 Oct 2026: people: at least 2, at most 21.", rollup={"rules_version": 2})
    _row(rollup_db, grain="hour", period_start=NOW - timedelta(hours=5),
         line="Walkway 05:00-06:00: 1 vehicle passing.", rollup={"rules_version": 2})
    assert _read() == ""


def test_just_after_midnight_falls_back_to_the_newest_complete_hour(rollup_db):
    now = datetime(2026, 10, 24, 7, 10, tzinfo=timezone.utc)  # 01:10 MDT, no day row yet
    _row(rollup_db, grain="hour", period_start=datetime(2026, 10, 24, 6, 0, tzinfo=timezone.utc),
         line="Walkway 00:00-01:00 (Sat 24 Oct): nobody passed.", rollup={"rules_version": 2})
    _row(rollup_db, grain="hour", period_start=datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc),
         line="Walkway 01:00-02:00 (Sat 24 Oct): running.", rollup={"rules_version": 2}, complete=False)
    assert _read(now=now) == "Walkway 00:00-01:00 (Sat 24 Oct): nobody passed."


def test_day_row_wins_over_hours(rollup_db):
    _row(rollup_db, grain="hour", period_start=NOW - timedelta(hours=1),
         line="Walkway 09:00-10:00: 2 strollers.", rollup={"rules_version": 2})
    _row(rollup_db, grain="day", period_start=TODAY, line=V2_DAY_LINE, rollup={"rules_version": 2})
    assert _read() == V2_DAY_LINE


def test_other_streams_and_off_switches(rollup_db):
    _row(rollup_db, grain="day", period_start=TODAY, line=V2_DAY_LINE, rollup={"rules_version": 2}, stream="cam0")
    assert _read() == ""
    assert _read(stream_ids=[]) == ""
    assert _read(max_chars=0) == ""


def test_failing_db_is_no_line(monkeypatch):
    from app import vision_reader

    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(vision_reader, "_get_engine", _boom)
    assert _read() == ""


def test_budget_cuts_at_a_clause_and_keeps_the_range(rollup_db):
    _row(rollup_db, grain="day", period_start=TODAY, line=V2_DAY_LINE, rollup={"rules_version": 2})
    got = _read(max_chars=200)
    assert len(got) <= 200
    assert got.endswith("…")
    assert RANGE in got
    # Never cut inside the range clause: too small a budget is no line, not "at least 1".
    assert _read(max_chars=60) == ""


def test_trim_rollup_line_pure():
    from app.vision_reader import trim_rollup_line

    assert trim_rollup_line("a; b; c", 100) == "a; b; c"
    assert trim_rollup_line("aaaa; bbbb; cccc", 11) == "aaaa; bbbb…"
    assert trim_rollup_line("", 100) == ""


# --- the prompt and the tick ---------------------------------------------------


def test_day_line_alone_opens_the_outside_section_in_both_branches():
    from app import reverie

    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="x", thread_label="t")
    for kwargs in ({}, {"concern_cards": [card]}):
        ctx = reverie.build_reverie_context(_broadcast(), street_day=V2_DAY_LINE, **kwargs)
        assert ctx["street_day"] == V2_DAY_LINE
        assert ctx["street_camera"].startswith("your outdoor camera")
        out = _render(**ctx)
        assert HEADING in out
        assert f"the day so far, counted from this camera: {V2_DAY_LINE}" in out
        assert "just now:" not in out


def test_no_day_line_no_key():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast(), street_day="")
    assert "street_day" not in ctx and "street_camera" not in ctx
    assert HEADING not in _render(**ctx)


@pytest.mark.asyncio
async def test_tick_threads_day_line_and_logs_rollup_1(monkeypatch, caplog):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_day_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")
    monkeypatch.setattr(vision_reader, "read_recent_street_events", lambda **kw: [])
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", lambda **kw: [])
    seen = {}

    def _day(**kw):
        seen.update(kw)
        return V2_DAY_LINE

    monkeypatch.setattr(vision_reader, "read_street_day_line", _day)
    cortex = _cortex()
    with caplog.at_level(logging.INFO):
        assert await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    assert ctx["street_day"] == V2_DAY_LINE
    assert seen["stream_ids"] == ["walkway"]
    assert seen["max_chars"] == settings.reverie_street_day_max_chars
    lines = [r.getMessage() for r in caplog.records if "reverie_street_context" in r.getMessage()]
    assert lines and lines[-1].endswith("captions=0 graded=0 rollup=1")


@pytest.mark.asyncio
async def test_failing_day_read_costs_only_the_day_line(monkeypatch, caplog):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")
    monkeypatch.setattr(vision_reader, "read_recent_street_events",
                        lambda **kw: [{"narrative": "A van is parked.", "age_sec": 30.0}])
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", lambda **kw: [])

    def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(vision_reader, "read_street_day_line", boom)
    cortex = _cortex()
    with caplog.at_level(logging.INFO):
        assert await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    assert "street_day" not in ctx
    assert ctx["street_percepts"][0]["narrative"] == "A van is parked."
    lines = [r.getMessage() for r in caplog.records if "reverie_street_context" in r.getMessage()]
    assert lines and lines[-1].endswith("captions=1 graded=0 rollup=0")


@pytest.mark.asyncio
async def test_day_switch_off_skips_the_read(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    def boom(**kw):
        raise AssertionError("day reader must not be called when off")

    monkeypatch.setattr(settings, "reverie_street_enabled", True)
    monkeypatch.setattr(settings, "reverie_street_day_enabled", False)
    monkeypatch.setattr(settings, "reverie_street_stream_ids", "walkway")
    monkeypatch.setattr(vision_reader, "read_recent_street_events", lambda **kw: [])
    monkeypatch.setattr(vision_reader, "read_recent_street_graded", lambda **kw: [])
    monkeypatch.setattr(vision_reader, "read_street_day_line", boom)
    cortex = _cortex()
    assert await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert "street_day" not in cortex.execute_plan.call_args.kwargs["req"].context
