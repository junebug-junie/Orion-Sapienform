"""The street as labelled material (walkway captions + graded predictions).

Properties: captions are fresh, deduped, never reducer rows, and always
labelled as the street; graded predictions render met/missed only, with the
14-day hit rate when there is one; nothing renders as filler when empty; the
room percept read stays room-only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from orion.situational.perception_reader import (
    _room_percept_stmt,
    fetch_street_summary,
    summarize_street,
)
from orion.situational.street_material import (
    build_graded_expectations,
    dedupe_street_captions,
    graded_expectation_lines,
    parse_stream_ids,
    street_caption_line,
)

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)  # 08:00 MDT


def _cap(text, minutes_ago, event_type="visual_observation"):
    return {"narrative": text, "created_at": NOW - timedelta(minutes=minutes_ago), "event_type": event_type,
            "stream_id": "walkway"}


def test_captions_newest_first_deduped_fresh_and_never_reducer_rows() -> None:
    rows = [
        _cap("Two vehicles are observed in the walkway area.", 1),
        _cap("two vehicles are observed in the walkway area", 2),  # repeat, different case
        _cap("The black dog arrived as expected.", 0, event_type="arrived_as_expected"),
        _cap("   ", 0),
        _cap("A person walks a bicycle past the gate.", 5),
        _cap("A van is parked across the street.", 40),  # stale
        {"narrative": "no timestamp", "created_at": None},
    ]
    got = dedupe_street_captions(rows, now=NOW, max_age_sec=900, limit=3)
    assert [c["narrative"] for c in got] == [
        "Two vehicles are observed in the walkway area.",
        "A person walks a bicycle past the gate.",
    ]
    assert got[0]["age_sec"] == 60.0


def test_caption_line_is_labelled_as_the_street_and_absent_when_empty() -> None:
    assert street_caption_line([], camera="walkway") is None
    assert street_caption_line([{"narrative": ""}]) is None
    line = street_caption_line([{"narrative": "A car passes."}, {"narrative": "A dog sniffs the gate"}])
    assert line == "On the walkway just now: A car passes; A dog sniffs the gate."


def test_parse_stream_ids_empty_string_disables() -> None:
    assert parse_stream_ids(None) == ["walkway"]
    assert parse_stream_ids("") == []
    assert parse_stream_ids(" walkway , porch ") == ["walkway", "porch"]


def test_summarize_street_puts_the_caption_line_first() -> None:
    lines = summarize_street(
        sightings=[], expectations=[], unresolved=[{"observed_at": NOW - timedelta(minutes=3),
                                                     "description": "a shape"}],
        patio=None, now=NOW, tz=TZ, camera="walkway",
        captions=[_cap("A cyclist rides past.", 2), _cap("A cyclist rides past.", 3)],
    )
    assert lines[0] == "On the walkway just now: A cyclist rides past."
    assert len(lines) == 2


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        class _R:
            def __init__(self, m):
                self._mapping = m
        return [_R(r) for r in self.rows]


class _Conn:
    def __init__(self, tables):
        self.tables = tables
        self.params = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        sql = str(stmt)
        for needle, rows in self.tables.items():
            if needle in sql:
                self.params[needle] = params
                if isinstance(rows, Exception):
                    raise rows
                return _Result(rows)
        return _Result([])

    def rollback(self):
        pass


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def test_fetch_street_summary_reads_walkway_captions_by_stream() -> None:
    now = datetime.now(timezone.utc)
    conn = _Conn({"FROM vision_events": [
        {"event_type": "visual_observation", "narrative": "Two vehicles are parked.", "created_at": now},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=now)
    assert got.read_ok is True
    assert got.lines == ["On the walkway just now: Two vehicles are parked."]
    assert conn.params["FROM vision_events"]["stream_id"] == "walkway"


def test_caption_read_failure_costs_only_its_line() -> None:
    now = datetime.now(timezone.utc)
    conn = _Conn({
        "FROM vision_events": RuntimeError("boom"),
        "vision_unresolved": [{"observed_at": now - timedelta(minutes=3), "description": "a shape"}],
    })
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=now)
    assert got.read_ok is True
    assert len(got.lines) == 1 and "could not name" in got.lines[0]


def test_room_percept_statement_still_room_only() -> None:
    sql = str(_room_percept_stmt())
    assert "stream_id IN" in sql
    assert "walkway" not in sql


# --- graded predictions ---------------------------------------------------------


def _exp(subject, status, minutes_ago, *, key=None, peak=460):
    return {"expectation_id": f"e-{subject}-{minutes_ago}", "stream_id": "walkway",
            "subject_key": key or f"individual:{subject}", "subject_label": subject, "status": status,
            "peak_minute": peak, "window_start": NOW - timedelta(minutes=minutes_ago + 30),
            "scored_at": NOW - timedelta(minutes=minutes_ago)}


def test_graded_lines_met_missed_only_with_hit_rate() -> None:
    rows = [
        _exp("the black dog", "met", 10),
        _exp("the mail van", "missed", 5, peak=None),
        _exp("a cyclist", "unscorable", 1),
        _exp("a runner", "open", 1),
    ]
    cal = [{"stream_id": "walkway", "subject_key": "individual:the black dog", "met": 4, "missed": 1},
           {"stream_id": "walkway", "subject_key": "individual:the mail van", "met": 0, "missed": 0}]
    lines = graded_expectation_lines(rows, cal, tz=TZ, limit=5)
    assert lines == [
        "On the walkway I expected the mail van (Thu 15 Oct); it did not come.",
        "On the walkway I expected the black dog around 07:40 (Thu 15 Oct); it came. "
        "Over the last 14 days it came 4 of 5 times.",
    ]


def test_graded_lines_empty_means_nothing() -> None:
    assert graded_expectation_lines([], [], tz=TZ, limit=5) == []
    assert graded_expectation_lines([_exp("x", "open", 1)], None, tz=TZ, limit=5) == []
    assert build_graded_expectations([{"status": "met", "subject_label": "", "subject_key": "k"}]) == []


def test_reducer_types_are_excluded_in_every_caption_sql() -> None:
    from orion.curiosity.study_material import STREET_CAPTIONS_RECENT_SQL
    from orion.situational.perception_reader import _STREET_CAPTIONS_SQL
    from orion.situational.street_material import STREET_CAPTIONS_SQL, WALKWAY_REDUCER_EVENT_TYPES

    for sql in (STREET_CAPTIONS_SQL, str(_STREET_CAPTIONS_SQL), STREET_CAPTIONS_RECENT_SQL):
        assert "NOT IN" in sql
        for t in WALKWAY_REDUCER_EVENT_TYPES:
            assert f"'{t}'" in sql


def test_brief_captions_are_clipped_so_the_rhythm_line_survives_the_stance_clip() -> None:
    long = "A " + "very " * 60 + "long caption."
    lines = summarize_street(
        sightings=[], expectations=[{"subject_label": "the black dog", "status": "missed", "peak_minute": 460}],
        unresolved=[], patio=None, now=NOW, tz=TZ, camera="walkway",
        captions=[_cap(long, 1), _cap(long.replace("long", "other"), 2)],
    )
    assert len(" ".join(lines)) <= 420
    assert "did not come" in " ".join(lines)
