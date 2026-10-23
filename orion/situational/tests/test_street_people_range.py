"""The street line says people as a range, never one bare count (spec PR #2763, decision 5).

Live defect (read-only replay of origin/main, 2026-10-23): over 673 moments of the last 7 days,
55 who-is-around lines stated a bare people count, up to "13 unfamiliar people" (2026-10-22
00:09 UTC). That count is person clusters; the matcher splits one person into several. The
line reaches every chat turn through PerceptionContextV1.street_summary.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from orion.situational.perception_reader import (
    fetch_street_summary,
    summarize_street,
    unnamed_people_range,
)
from orion.vision.walkway_rollup import bare_people_counts

TZ = ZoneInfo("America/Denver")
NOW = datetime(2026, 10, 22, 0, 9, tzinfo=timezone.utc)
PREFIX = "On the walkway in the last 15 minutes: "
SPLIT = " (the camera splits one person into several)"


def _s(iid, kind="person", label=None, days=1, minutes_ago=2):
    return {"individual_id": iid, "kind": kind, "label": label, "distinct_days": days,
            "first_at": NOW - timedelta(minutes=minutes_ago + 1), "last_at": NOW - timedelta(minutes=minutes_ago)}


def _who(sightings, peak=None, props=None):
    windows = None if peak is None else [{"counts": {"person": peak}}]
    lines = summarize_street(sightings=sightings, expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ,
                             people_windows=windows, stationary_people=props)
    return lines[0] if lines else ""


def test_live_13_clusters_with_2_at_once_reads_as_a_range() -> None:
    line = _who([_s(f"p{i}") for i in range(13)], peak=2)
    assert line == PREFIX + "at least 2, at most 13 unfamiliar people" + SPLIT + "."
    assert bare_people_counts(line) == []


def test_named_people_come_off_the_floor_not_the_names() -> None:
    line = _who([_s("n", label="Pip"), *[_s(f"p{i}") for i in range(4)]], peak=3)
    assert line == PREFIX + "Pip, and at least 2, at most 4 unfamiliar people" + SPLIT + "."


def test_a_split_of_a_named_person_is_only_possibly_someone() -> None:
    assert _who([_s("n", label="Pip"), _s("p")], peak=1) == PREFIX + "Pip, and possibly an unfamiliar person."
    assert _who([_s("n", label="Pip"), _s("a"), _s("b")], peak=1) == (
        PREFIX + "Pip, and up to 2 unfamiliar people" + SPLIT + ".")


def test_more_at_once_than_clusters_raises_the_high() -> None:
    assert _who([_s("p")], peak=3) == PREFIX + "at least 3 unfamiliar people."


def test_one_cluster_one_at_once_is_one_person() -> None:
    assert _who([_s("p")], peak=1) == PREFIX + "an unfamiliar person."


def test_a_stationary_prop_comes_off_the_floor() -> None:
    # The patio skeleton stood in every window: 2 at once, 1 of them a prop.
    assert _who([_s("a"), _s("b")], peak=2, props=1) == (
        PREFIX + "at least 1, at most 2 unfamiliar people" + SPLIT + ".")


def test_unread_window_count_can_only_say_up_to() -> None:
    assert _who([_s("a"), _s("b"), _s("c")]) == PREFIX + "up to 3 unfamiliar people" + SPLIT + "."
    assert _who([_s("a")]) == PREFIX + "an unfamiliar person."


def test_people_seen_on_other_days_are_said_as_some_never_counted() -> None:
    line = _who([_s("a", days=3), _s("b"), _s("c")], peak=2)
    assert line == PREFIX + ("at least 2, at most 3 people I have no name for "
                             "(some seen here on other days; the camera splits one person into several).")


def test_a_cut_people_range_is_and_others_not_a_count() -> None:
    rows = [_s(n, label=n, minutes_ago=1) for n in ("A", "B", "C", "D")] + [_s("p"), _s("q")]
    assert _who(rows, peak=6) == PREFIX + "A, B, C, D, and others."


def test_range_never_inverts_and_floor_never_exceeds_what_was_seen() -> None:
    for unnamed, named, peak, props in itertools.product(range(7), range(3), [None, *range(7)], range(2)):
        low, high = unnamed_people_range(unnamed=unnamed, named=named, peak=peak, props=props)
        assert high >= unnamed
        if low is not None:
            assert 0 <= low <= high
            assert low <= max(peak or 0, 1)


def test_no_orion_facing_street_line_states_a_bare_people_count() -> None:
    """Every who-is-around and patio line over a grid of rows: zero bare people counts."""
    patio = {"state": "present", "zone": "patio", "row_updated_at": datetime.now(timezone.utc)}
    for unnamed, named, recurring, peak, props in itertools.product(
        range(6), range(3), range(3), [None, *range(5)], [None, 0, 1]
    ):
        rows = [_s(f"n{i}", label=f"Name{i}") for i in range(named)]
        rows += [_s(f"r{i}", days=2 + i) for i in range(min(recurring, unnamed))]
        rows += [_s(f"p{i}") for i in range(unnamed - min(recurring, unnamed))]
        windows = None if peak is None else [{"counts": {"person": peak, "vehicle": 9}}]
        for count in (None, 1, 2, 7):
            lines = summarize_street(
                sightings=rows, expectations=[], unresolved=[], patio=dict(patio, count=count),
                now=NOW, tz=TZ, people_windows=windows, stationary_people=props,
            )
            for line in lines:
                assert bare_people_counts(line) == [], line


# --- the read --------------------------------------------------------------------------------------


class _Conn:
    def __init__(self, tables):
        self.tables = tables

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        sql = str(stmt)
        for needle, rows in self.tables.items():
            if needle in sql:
                if isinstance(rows, Exception):
                    raise rows
                return SimpleNamespace(all=lambda rows=rows: [SimpleNamespace(_mapping=r) for r in rows])
        return SimpleNamespace(all=lambda: [])

    def rollback(self):
        pass


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def test_fetch_reads_the_window_counts_as_dicts_or_json_text() -> None:
    conn = _Conn({
        "GROUP BY s.individual_id": [_s(f"p{i}") for i in range(5)],
        "vision_scene_inventory": [{"counts": '{"person": 1, "vehicle": 3}'},
                                   {"counts": {"person": 2, "bicycle person": 1}}, {"counts": "not json"}],
        "count(DISTINCT s.individual_id)": [{"n": 0}],
    })
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.lines == [PREFIX + "at least 2, at most 5 unfamiliar people" + SPLIT + "."]


def test_fetch_with_the_window_read_failing_says_up_to() -> None:
    conn = _Conn({
        "GROUP BY s.individual_id": [_s("a"), _s("b")],
        "vision_scene_inventory": RuntimeError("relation does not exist"),
    })
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.lines == [PREFIX + "up to 2 unfamiliar people" + SPLIT + "."]
