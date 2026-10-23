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


# --- sibling PR #2683: patio fingerprinting, stationary fixtures, head_count_zone ---


def test_stationary_sightings_are_excluded_with_fallback_before_the_migration() -> None:
    now = datetime.now(timezone.utc)
    sighting = {"individual_id": "i1", "kind": "person", "label": None, "distinct_days": 1,
                "first_at": now - timedelta(minutes=3), "last_at": now - timedelta(minutes=1)}

    class _NoColumnYet(_Conn):
        def execute(self, stmt, params):
            if "s.stationary" in str(stmt):
                raise RuntimeError('column s.stationary does not exist')
            return super().execute(stmt, params)

    conn = _NoColumnYet({"vision_individual_sighting": [sighting]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=now)
    assert got.lines == ["On the walkway in the last 15 minutes: an unfamiliar person."]

    seen = []

    class _Migrated(_Conn):
        def execute(self, stmt, params):
            if "vision_individual_sighting" in str(stmt):
                seen.append(str(stmt))
            return super().execute(stmt, params)

    fetch_street_summary("walkway", engine=_Engine(_Migrated({"vision_individual_sighting": []})), now=now)
    assert len(seen) == 1 and "AND NOT s.stationary" in seen[0]


def test_head_count_row_found_by_head_count_zone_flag_and_kept_a_count() -> None:
    now = datetime.now(timezone.utc)
    conn = _Conn({"substrate_embodied_presence": [
        {"presence_id": "walkway:patio",
         "presence_json": '{"state": "present", "zone": "patio", "subject": {"count": 2}, '
                          '"identity_confirmed": true, "head_count_zone": true}',
         "updated_at": now},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=now)
    assert got.lines == ["2 people are on the patio."]
    from orion.situational.perception_reader import _PATIO_PRESENCE_SQL

    sql = str(_PATIO_PRESENCE_SQL)
    assert "head_count_zone" in sql and "no_embed_zone" in sql


# --- camera captions (orion-vision-window caption events, 2026-10-16) ----------
# Live 2026-10-16 the gate accepted "A person is pushing a stroller on a
# sidewalk ..." at 22:53 UTC and Orion's street line still read only the
# council's "A vehicle is observed in the walkway area." These rows now exist;
# they must read as a caption of one frame, never as a checked fact.

STROLLER = ("A person is pushing a stroller on a sidewalk in a suburban neighborhood, "
            "with a car parked on the street and a mountain in the background.")


def _camcap(text, minutes_ago, labels=("person", "stroller")):
    row = _cap(text, minutes_ago, event_type="camera_caption")
    row["tags"] = ["camera_caption", "motion_unverified", *(f"detector:{l}" for l in labels)]
    return row


def test_camera_caption_is_labelled_with_detector_boxes_for_every_consumer() -> None:
    (got,) = dedupe_street_captions([_camcap(STROLLER, 1)], now=NOW, max_age_sec=900, limit=2)
    assert got["event_type"] == "camera_caption"
    assert got["detector_labels"] == ["person", "stroller"]
    assert got["narrative"].startswith("camera caption (one frame; detector saw person, stroller; motion not verified): ")
    assert '"A person is pushing a stroller' in got["narrative"]
    assert got["caption_text"].startswith("A person is pushing")


def test_camera_caption_tags_as_json_text_still_parse() -> None:
    row = _camcap("A dog walks by.", 1)
    row["tags"] = '["camera_caption", "detector:dog"]'
    (got,) = dedupe_street_captions([row], now=NOW, max_age_sec=900, limit=2)
    assert got["detector_labels"] == ["dog"]


DRIVING = "A car is driving down a residential street."


def _checked(text, tags, minutes_ago=1):
    row = _camcap(text, minutes_ago, labels=("person",))
    row["tags"] = [*row["tags"], *tags]
    return row


def test_contradicted_caption_says_the_thing_did_not_move_for_every_consumer() -> None:
    """dedupe_street_captions is the shared step of the chat brief, reverie
    (orion-thought vision_reader) and curiosity (study_material) readers."""
    row = _checked(DRIVING, ["motion_contradicted", "motion_check:vehicle:driving:parked"])
    (got,) = dedupe_street_captions([row], now=NOW, max_age_sec=900, limit=2)
    assert got["narrative"] == (
        'camera caption (one frame; detector saw person; caption said driving; the vehicle did not move): '
        f'"{DRIVING}"')
    assert "motion not verified" not in got["narrative"]
    assert got["motion_check"].contradicted is True
    line = street_caption_line([got], camera="walkway", tz=TZ)
    assert "caption said driving; the vehicle did not move" in line


def test_supported_caption_says_the_thing_did_move() -> None:
    row = _checked("A person is walking a dog.", ["motion_supported", "motion_check:person:walking:walking"])
    (got,) = dedupe_street_captions([row], now=NOW, max_age_sec=900, limit=2)
    assert "caption said walking; the person did move" in got["narrative"]


def test_unchecked_or_malformed_motion_tags_stay_not_verified() -> None:
    for tags in ([], ["motion_contradicted"], ["motion_check:vehicle:driving:parked"],
                 ["motion_contradicted", "motion_check:vehicle:driving"]):
        (got,) = dedupe_street_captions([_checked(DRIVING, tags)], now=NOW, max_age_sec=900, limit=2)
        assert "motion not verified" in got["narrative"], tags
        assert got["motion_check"] is None


def test_one_frame_people_count_is_not_shown_as_a_head_count() -> None:
    """Live council row 2026-10-16 17:52 and caption 2026-10-18 22:16: one
    frame's count of people must not read as how many people there were."""
    council = _cap("Two persons are observed walking on the walkway.", 1)
    caption = _camcap("In the image, three people are riding bicycles on a residential street.", 2,
                      labels=("bicycle",))
    got = dedupe_street_captions([council, caption], now=NOW, max_age_sec=900, limit=3)
    assert got[0]["narrative"] == (
        "Two persons are observed walking on the walkway (people count from one frame, not a head count)")
    assert got[1]["narrative"].startswith(
        "camera caption (one frame; detector saw bicycle; motion not verified; "
        "people count from one frame, not a head count): ")
    line = street_caption_line(got, camera="walkway", tz=TZ)
    assert line.count("people count from one frame, not a head count") == 2


def test_no_count_and_range_counts_are_left_alone() -> None:
    for text in ("A person walks a dog.", "Two cars are parked.", "At least 2, at most 5 people passed."):
        (got,) = dedupe_street_captions([_cap(text, 1)], now=NOW, max_age_sec=900, limit=2)
        assert got["narrative"] == text


def test_council_rows_stay_unlabelled() -> None:
    (got,) = dedupe_street_captions([_cap("A vehicle is observed in the walkway area.", 1)],
                                    now=NOW, max_age_sec=900, limit=2)
    assert got["narrative"] == "A vehicle is observed in the walkway area."
    assert got["caption_text"] is None


def test_brief_line_shows_the_camera_caption_with_its_local_time_alongside_the_council() -> None:
    lines = summarize_street(
        sightings=[], expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ, camera="walkway",
        captions=[_camcap(STROLLER, 1), _cap("A vehicle is observed in the walkway area.", 2)],
    )
    assert lines[0].startswith(
        'On the walkway just now: camera caption logged 07:59 (one frame; detector saw person, stroller; '
        'motion not verified): "A person is pushing a stroller'
    )
    assert lines[0].endswith("; A vehicle is observed in the walkway area.")


def test_brief_shows_at_most_one_camera_caption_and_keeps_the_rhythm_line() -> None:
    long = "A " + "very " * 60 + "long caption."
    lines = summarize_street(
        sightings=[], expectations=[{"subject_label": "the black dog", "status": "missed", "peak_minute": 460}],
        unresolved=[], patio=None, now=NOW, tz=TZ, camera="walkway",
        captions=[_camcap("First " + long, 1), _camcap("Second " + long, 2), _cap("Third " + long, 3)],
    )
    assert lines[0].count("camera caption") == 1
    assert '"First' in lines[0] and "Second" not in lines[0] and "; Third" in lines[0]
    assert len(" ".join(lines)) <= 420
    assert "did not come" in " ".join(lines)


def test_fetch_street_summary_selects_tags_and_labels_a_camera_caption() -> None:
    now = datetime.now(timezone.utc)
    row = _camcap(STROLLER, 0)
    row["created_at"] = now
    conn = _Conn({"FROM vision_events": [row]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=now)
    assert "camera caption logged " in got.lines[0] and "detector saw person, stroller" in got.lines[0]


def test_every_caption_sql_selects_tags() -> None:
    from orion.curiosity.study_material import STREET_CAPTIONS_RECENT_SQL
    from orion.situational.perception_reader import _STREET_CAPTIONS_SQL
    from orion.situational.street_material import STREET_CAPTIONS_SQL

    for sql in (STREET_CAPTIONS_SQL, str(_STREET_CAPTIONS_SQL), STREET_CAPTIONS_RECENT_SQL):
        assert " tags," in sql


def test_seven_detector_labels_still_fit_the_street_budget() -> None:
    every = ("person", "dog", "bicycle", "stroller", "vehicle", "package", "mail truck")
    long = "A " + "very " * 60 + "long caption."
    lines = summarize_street(
        sightings=[], expectations=[{"subject_label": "the black dog", "status": "missed", "peak_minute": 460}],
        unresolved=[], patio=None, now=NOW, tz=TZ, camera="walkway",
        captions=[_camcap("Cam " + long, 1, labels=every), _cap("Council " + long, 2)],
    )
    assert "detector saw person, dog, bicycle +4;" in lines[0]
    assert len(" ".join(lines)) <= 420
    assert "did not come" in " ".join(lines)


def test_many_captions_never_crowd_out_the_council_row_in_any_consumer() -> None:
    rows = [_camcap(f"Caption number {i} of a busy street.", i * 0.1) for i in range(25)]
    rows.append(_cap("A vehicle is observed in the walkway area.", 5))
    got = dedupe_street_captions(rows, now=NOW, max_age_sec=900, limit=3)
    assert [g["event_type"] for g in got] == ["camera_caption", "visual_observation"]


def test_caption_sql_fetches_captions_and_council_rows_as_separate_limits() -> None:
    """Runs the real shared SQL on SQLite: 25 newer captions, 1 older council
    row; the council row is still fetched."""
    from sqlalchemy import bindparam, create_engine, text

    from orion.situational.street_material import STREET_CAPTIONS_SQL

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE vision_events (event_id TEXT, stream_id TEXT, event_type TEXT, "
                          "narrative TEXT, tags TEXT, created_at TIMESTAMP)"))
        for i in range(25):
            conn.execute(text("INSERT INTO vision_events VALUES (:i, 'walkway', 'camera_caption', :n, '[]', :c)"),
                         {"i": f"c{i}", "n": f"cap {i}", "c": NOW - timedelta(seconds=i)})
        conn.execute(text("INSERT INTO vision_events VALUES ('v', 'walkway', 'visual_observation', "
                          "'A vehicle is observed.', '[]', :c)"), {"c": NOW - timedelta(minutes=5)})
        stmt = text(STREET_CAPTIONS_SQL).bindparams(bindparam("stream_ids", expanding=True))
        rows = conn.execute(stmt, {"stream_ids": ["walkway"], "since": NOW - timedelta(hours=1),
                                   "limit": 20}).mappings().all()
    types = [r["event_type"] for r in rows]
    assert types.count("camera_caption") == 2
    assert "visual_observation" in types
