"""Which camera speaks for "where is Juniper right now", 2026-08-29.

A single hardcoded `perception_stream_id` was not merely inflexible, it was
measurably wrong live: the cortex-exec chat replica read `cam0` (the interior
room camera, `absent` for 70 minutes) while `carbon` -- the laptop webcam
Juniper was sitting at -- reported a person present with `last_seen_sec=0.0`.
The prompt narrated an empty room at someone at their desk.

Row ages here are hand-computed against an explicit `max_age_seconds=120`, per
this repo's standing rule that a green test proves the rule rather than
agreement with the code that wrote it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.situational.context import _split_stream_ids
from orion.situational.perception_reader import (
    fetch_presence_resolved,
    presence_row_age_seconds,
)

MAX_AGE = 120.0


def _ago(seconds: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.params = None

    def execute(self, _stmt, params):
        self.params = params
        return self

    def all(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    """Stands in for the SQLAlchemy engine. Rows are (presence_id,
    presence_json, updated_at), matching the real SELECT's column order."""

    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def connect(self):
        return self.conn


class _RaisingEngine:
    def connect(self):
        raise RuntimeError("db unreachable")


def _resolve(rows, stream_ids=("carbon", "cam0")):
    return fetch_presence_resolved(
        list(stream_ids), max_age_seconds=MAX_AGE, engine=_FakeEngine(rows)
    )


# -- the live bug ------------------------------------------------------------


def test_the_camera_with_a_person_wins_over_the_first_configured_one() -> None:
    """The exact live shape from 2026-08-29, with cam0 listed first to prove
    ordering is not what decides this."""
    stream_id, presence = _resolve(
        [
            ("cam0", {"state": "absent", "since_sec": 4170.2}, _ago(1.0)),
            ("carbon", {"state": "present", "since_sec": 41.1}, _ago(1.0)),
        ],
        stream_ids=("cam0", "carbon"),
    )
    assert stream_id == "carbon"
    assert presence["state"] == "present"


def test_a_stale_present_row_loses_to_a_fresh_recent_one() -> None:
    """A camera that went dark still SAYS 'present' -- nothing overwrites the
    row, it just stops being rewritten. 3600s > 120s, so cam0 is not a live
    sighting; carbon at 5s is."""
    stream_id, _ = _resolve(
        [
            ("cam0", {"state": "present", "since_sec": 10.0}, _ago(3600.0)),
            ("carbon", {"state": "recent", "since_sec": 30.0}, _ago(5.0)),
        ]
    )
    assert stream_id == "carbon"


def test_present_outranks_recent_even_when_recent_is_fresher() -> None:
    """Tier before recency: someone in frame now beats someone who just left,
    regardless of which row was written a fraction of a second sooner."""
    stream_id, _ = _resolve(
        [
            ("cam0", {"state": "recent", "since_sec": 10.0}, _ago(1.0)),
            ("carbon", {"state": "present", "since_sec": 10.0}, _ago(30.0)),
        ]
    )
    assert stream_id == "carbon"


def test_two_live_cameras_break_the_tie_on_the_fresher_row() -> None:
    """Deterministic rather than dict-ordering-dependent."""
    stream_id, _ = _resolve(
        [
            ("cam0", {"state": "present", "since_sec": 10.0}, _ago(60.0)),
            ("carbon", {"state": "present", "since_sec": 10.0}, _ago(2.0)),
        ]
    )
    assert stream_id == "carbon"


def test_age_boundary_is_inclusive() -> None:
    """Exactly at the threshold still counts as fresh: 120.0 <= 120.0.
    A hair over does not."""
    inside, _ = _resolve([("carbon", {"state": "present"}, _ago(119.0))])
    assert inside == "carbon"
    at_edge, presence = _resolve([("carbon", {"state": "present"}, _ago(120.5))])
    # Past the edge it falls through to the "first configured stream that
    # returned a row" fallback -- still carbon here, but as a fallback rather
    # than as a live sighting, which the caller distinguishes by row age.
    assert at_edge == "carbon"
    assert presence_row_age_seconds(presence) > MAX_AGE


def test_nothing_fresh_falls_back_to_the_first_configured_stream() -> None:
    """Order is the documented tiebreak, so this must follow the configured
    list rather than whatever the database returned first."""
    stream_id, _ = _resolve(
        [
            ("cam0", {"state": "absent"}, _ago(9000.0)),
            ("carbon", {"state": "absent"}, _ago(9000.0)),
        ],
        stream_ids=("carbon", "cam0"),
    )
    assert stream_id == "carbon"


def test_no_rows_at_all_is_a_clean_miss() -> None:
    assert _resolve([]) == (None, None)


def test_empty_stream_list_never_queries() -> None:
    assert fetch_presence_resolved([], max_age_seconds=MAX_AGE, engine=_RaisingEngine()) == (
        None,
        None,
    )


def test_read_failure_fails_open() -> None:
    """Same contract as every other provider in this module: a database
    problem costs the enrichment, never the turn."""
    assert fetch_presence_resolved(
        ["carbon"], max_age_seconds=MAX_AGE, engine=_RaisingEngine()
    ) == (None, None)


def test_row_without_presence_json_is_skipped() -> None:
    stream_id, _ = _resolve(
        [
            ("cam0", None, _ago(1.0)),
            ("carbon", {"state": "present"}, _ago(1.0)),
        ]
    )
    assert stream_id == "carbon"


def test_naive_timestamps_are_treated_as_utc() -> None:
    """Postgres can hand back a naive datetime depending on column type; a
    naive value compared against an aware `now` raises, and this module's
    fail-open contract would swallow that into 'no presence' -- silently
    turning every read into a miss."""
    naive = datetime.utcnow() - timedelta(seconds=5)
    stream_id, presence = _resolve([("carbon", {"state": "present"}, naive)])
    assert stream_id == "carbon"
    age = presence_row_age_seconds(presence)
    assert age is not None and age < MAX_AGE


def test_age_is_none_when_the_row_carries_no_timestamp() -> None:
    assert presence_row_age_seconds({"state": "present"}) is None
    assert presence_row_age_seconds(None) is None


def test_only_the_requested_streams_are_queried() -> None:
    engine = _FakeEngine([("carbon", {"state": "present"}, _ago(1.0))])
    fetch_presence_resolved(["carbon", "cam0"], max_age_seconds=MAX_AGE, engine=engine)
    assert engine.conn.params == {"stream_ids": ["carbon", "cam0"]}


# -- config parsing ----------------------------------------------------------


def test_stream_ids_parse_preserves_order_and_dedupes() -> None:
    assert _split_stream_ids("carbon, cam0 ,carbon") == ["carbon", "cam0"]


def test_stream_ids_parse_handles_absent_and_empty() -> None:
    assert _split_stream_ids(None) == []
    assert _split_stream_ids("") == []
    assert _split_stream_ids(" , ") == []


def test_stream_ids_parse_accepts_a_real_list() -> None:
    assert _split_stream_ids(["carbon", "cam0"]) == ["carbon", "cam0"]
