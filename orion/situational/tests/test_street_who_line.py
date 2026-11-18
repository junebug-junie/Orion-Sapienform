"""The "who is around" street line: names first, grouped phrases, honest cap.

Live defect (2026-10-15 23:35 UTC, orion-athena-cortex-exec-chat):

    On the walkway in the last 15 minutes: a vehicle I have seen on 2 days but
    have no name for, <name>, a vehicle I have seen on 2 days but have no name
    for, 4 unfamiliar bicycles, and 2 more.

Names in these tests are invented placeholders, not real labels.

Phrases were ordered by most recent sighting, so two parked cars took two of
the four slots (the same sentence twice) and a name Juniper gave could fall
into "and N more".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from orion.situational import perception_reader
from orion.situational.perception_reader import fetch_street_summary, summarize_street

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 15, 23, 35, tzinfo=timezone.utc)
PREFIX = "On the walkway in the last 15 minutes: "


def _s(iid, kind="person", label=None, days=1, minutes_ago=2):
    return {"individual_id": iid, "kind": kind, "label": label, "distinct_days": days,
            "first_at": NOW - timedelta(minutes=minutes_ago + 1),
            "last_at": NOW - timedelta(minutes=minutes_ago)}


def _who(sightings):
    lines = summarize_street(sightings=sightings, expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ)
    assert len(lines) <= 1
    return lines[0] if lines else ""


# Most recent first, the way the sightings SQL orders them: the rows behind
# the live line above.
LIVE_ROWS = [
    _s("edcb5152", "vehicle", days=2, minutes_ago=1),
    _s("f5242d40", "person", label="Pip", minutes_ago=2),
    _s("9b1729bb", "vehicle", days=2, minutes_ago=3),
    *[_s(f"bike{i}", "bicycle", minutes_ago=4 + i) for i in range(4)],
    *[_s(f"p{i}", "person", minutes_ago=5 + i) for i in range(3)],
    *[_s(f"v{i}", "vehicle", minutes_ago=6 + i) for i in range(2)],
]


def test_live_line_puts_the_name_first_and_groups_the_parked_cars() -> None:
    # No window count read: the unnamed people can only be "up to" (PR #2763 decision 5).
    assert _who(LIVE_ROWS) == (
        PREFIX + "Pip, up to 3 unfamiliar people (the camera splits one person into several), "
        "2 vehicles I have each seen on 2 days but have no name for, 2 unfamiliar vehicles, and 4 more."
    )


def test_named_first_then_living_then_things_regardless_of_recency() -> None:
    line = _who([
        _s("car", "vehicle", minutes_ago=1),
        _s("dog", "dog", days=3, minutes_ago=2),
        _s("bike", "bicycle", minutes_ago=3),
        _s("p", "person", minutes_ago=4),
        _s("m", "person", label="Wren", minutes_ago=14),
    ])
    assert line == (
        PREFIX + "Wren, a dog I have seen on 3 days but have no name for, an unfamiliar person, "
        "an unfamiliar bicycle, and 1 more."
    )


def test_names_never_fall_into_and_more_while_they_fit_the_cap() -> None:
    rows = [_s(f"v{i}", "vehicle", days=2 + i, minutes_ago=1) for i in range(6)]
    rows += [_s(f"b{i}", "bicycle", minutes_ago=1) for i in range(5)]
    rows += [_s(n, "person", label=n, minutes_ago=12) for n in ("Pip", "Wren", "Rowan Example")]
    line = _who(rows)
    assert line.startswith(PREFIX + "Pip, Wren, Rowan Example, ")
    assert line.endswith(", and 5 more.")  # the 5 bicycles, counted as individuals


def test_more_names_than_the_cap_cut_names_last_and_count_them() -> None:
    names = ["A", "B", "C", "D", "E", "F"]
    rows = [_s(n, "person", label=n, minutes_ago=1 + i) for i, n in enumerate(names)]
    rows.append(_s("car", "vehicle", minutes_ago=1))
    assert _who(rows) == PREFIX + "A, B, C, D, and 3 more."


def test_identical_unnamed_phrases_are_grouped_with_honest_day_counts() -> None:
    same = _who([_s("a", "vehicle", days=4), _s("b", "vehicle", days=4)])
    assert same == PREFIX + "2 vehicles I have each seen on 4 days but have no name for."
    mixed = _who([_s("a", "vehicle", days=5), _s("b", "vehicle", days=2), _s("c", "vehicle", days=9)])
    assert mixed == PREFIX + "3 vehicles I have each seen on at least 2 days but have no name for."
    one = _who([_s("a", "person", days=2)])
    assert one == PREFIX + "a person I have seen on 2 days but have no name for."
    many = _who([_s("a", "person", days=2), _s("b", "person", days=5), _s("c", "person")])
    assert many == PREFIX + (
        "up to 3 people I have no name for (some seen here on other days; "
        "the camera splits one person into several)."
    )


def test_no_phrase_repeats() -> None:
    line = _who(LIVE_ROWS)[len(PREFIX):-1]
    parts = [p.strip() for p in line.split(", ")]
    assert len(parts) == len(set(parts))


def test_two_individuals_with_the_same_name_are_one_name() -> None:
    assert _who([_s("a", label="Pip"), _s("b", label="pip ")]) == PREFIX + "Pip."


def test_old_sightings_and_bad_day_counts_do_not_break_the_line() -> None:
    assert _who([_s("old", label="Pip", minutes_ago=40)]) == ""
    assert _who([{**_s("x", "vehicle"), "distinct_days": "?"}]) == PREFIX + "an unfamiliar vehicle."


# --- `now=` reaches the SQL window --------------------------------------------


class _Result:
    def all(self):
        return []


class _Conn:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        self.calls.append((str(stmt), dict(params)))
        return _Result()

    def rollback(self):
        pass


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def test_now_is_bound_into_every_windowed_street_read() -> None:
    conn = _Conn()
    fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    windowed = [(sql, p) for sql, p in conn.calls if "make_interval" in sql]
    # sightings, expectations, unresolved, captions, people windows, stationary people
    assert len(windowed) == 6
    for sql, params in windowed:
        assert "now()" not in sql
        assert ":now" in sql
        assert params["now"] == NOW
        # Nothing from after `now` (plus a small clock grace): a replay must
        # not see today's rows.
        assert ":until" in sql
        assert params["until"] == NOW + timedelta(seconds=perception_reader.STREET_CLOCK_GRACE_SECONDS)


def test_no_street_statement_uses_the_server_clock() -> None:
    for name in ("_STREET_SIGHTINGS_SQL", "_STREET_SIGHTINGS_PRE_STATIONARY_SQL",
                 "_STREET_EXPECTATIONS_SQL", "_STREET_UNRESOLVED_SQL", "_STREET_CAPTIONS_SQL"):
        assert "now()" not in str(getattr(perception_reader, name)), name


def test_a_named_thing_sorts_among_the_names() -> None:
    line = _who([_s("p", "person", minutes_ago=1), _s("van", "vehicle", label="the blue van", minutes_ago=9)])
    assert line == PREFIX + "the blue van, and an unfamiliar person."


def test_one_kind_stays_together_and_the_last_item_gets_and() -> None:
    line = _who([
        _s("a", "vehicle", days=2, minutes_ago=1),
        _s("b", "bicycle", minutes_ago=2),
        _s("c", "bicycle", minutes_ago=3),
        _s("d", "vehicle", minutes_ago=4),
        _s("e", "vehicle", minutes_ago=5),
        _s("f", "vehicle", minutes_ago=6),
    ])
    assert line == (
        PREFIX + "a vehicle I have seen on 2 days but have no name for, 3 unfamiliar vehicles, "
        "and 2 unfamiliar bicycles."
    )
