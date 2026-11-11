"""Spot-checked motion stories reach the street line (2026-11-03).

Plan: docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md
(PR #2736): "Orion's street line gains what moved". Stories reach Orion only
behind Juniper's spot-check gate (orion/vision/motion_story_gate.py), the same
rule the walkway rollups use. Pins: gate passed -> the newest stories of
today appear, labelled as stories; gate closed, unreadable or flag off -> no
story line; a story told "together with" one already shown is skipped; no
stories never becomes an absence claim.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from orion.situational import perception_reader
from orion.situational.perception_reader import (
    STREET_MOTION_LINE_ENV,
    fetch_street_summary,
    street_story_line,
    summarize_street,
)

TZ = ZoneInfo("America/Denver")
# 2026-11-02 18:30 MST (UTC-7).
NOW = datetime(2026, 11, 3, 1, 30, tzinfo=timezone.utc)

S1 = ("A person walked left to right and away from the house on the walkway, over 15 s (7 detections), "
      "together with 1 other person, together with a dog.")
S2 = "A person with a dog walked right to left and toward the house from the walkway to the mailbox, over 10 s (4 detections)."


def _story(sid, minutes_ago, sentence, together=()):
    return {"story_id": sid, "started_at": NOW - timedelta(minutes=minutes_ago), "sentence": sentence,
            "together_story_ids": list(together)}


# --- pure rendering -----------------------------------------------------------


def test_story_line_is_labelled_timed_and_newest_first() -> None:
    line = street_story_line([_story("a", 5, S1), _story("b", 300, S2)], camera="walkway", tz=TZ)
    assert line.startswith("Walkway motion stories today (spot-checked): at 18:25, A person walked left to right")
    assert "; at 13:30, A person with a dog walked right to left" in line
    assert line.endswith(".") and ".." not in line


def test_story_line_caps_count_and_length_and_skips_the_same_passage() -> None:
    rows = [
        _story("a", 1, S1, together=("a2",)),
        _story("a2", 1, S1.replace("15 s (7", "8 s (4"), together=("a",)),  # the other walker of the same pair
        _story("b", 20, S2),
        _story("c", 40, S2),
    ]
    line = street_story_line(rows, camera="walkway", tz=TZ)
    assert line.count("at 1") == 2  # STREET_STORY_MAX, the pair told once
    assert "8 s (4" not in line
    for part in line.split(": ", 1)[1].split("; "):
        assert len(part) <= len("at 18:29, ") + perception_reader.STREET_STORY_CHARS + 1
    assert street_story_line([], camera="walkway", tz=TZ) is None
    assert street_story_line(None, camera="walkway", tz=TZ) is None


def test_no_stories_is_no_line_never_an_absence_claim() -> None:
    lines = summarize_street(sightings=[], expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ, stories=[])
    assert lines == []
    lines = summarize_street(sightings=[], expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ,
                             stories=[_story("a", 5, S1)])
    assert lines == [street_story_line([_story("a", 5, S1)], camera="walkway", tz=TZ)]


# --- the gate, through fetch_street_summary --------------------------------------


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return [SimpleNamespace(_mapping=r) for r in self._rows]


class _Conn:
    """Needle -> rows. Checked in order, so the most specific needles go first."""

    def __init__(self, tables):
        self.tables = tables
        self.seen: dict[str, dict] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        sql = str(stmt)
        for needle, rows in self.tables.items():
            if needle in sql:
                self.seen[needle] = dict(params)
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


def _tables(verdicts, *, version=1, stories=None):
    return {
        "max(rules_version)": [{"v": version}],
        "vision_motion_story_verdict": verdicts,
        "together_story_ids": stories if stories is not None else [_story("a", 5, S1), _story("b", 300, S2)],
    }


# Live tally 2026-11-03: 129 right of 131 checked, 0 wrong names.
PASSED = [{"verdict": "right", "n": 129}, {"verdict": "wrong_motion", "n": 2}, {"verdict": "cant_tell", "n": 1}]


def test_gate_passed_puts_todays_stories_on_the_street_line(monkeypatch) -> None:
    monkeypatch.delenv(STREET_MOTION_LINE_ENV, raising=False)
    conn = _Conn(_tables(PASSED))
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW, tz_name="America/Denver")
    assert got.read_ok
    (line,) = [ln for ln in got.lines if "motion stories" in ln]
    assert line.startswith("Walkway motion stories today (spot-checked): at 18:25, A person walked")
    params = conn.seen["together_story_ids"]
    assert params["v"] == 1 and params["stream_id"] == "walkway"
    # Since local midnight (2026-11-02 00:00 MST = 07:00 UTC), up to now + grace.
    assert params["since"] == datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)
    assert params["until"] > NOW
    assert conn.seen["vision_motion_story_verdict"] == {"v": 1}


@pytest.mark.parametrize(
    "verdicts",
    [
        [{"verdict": "right", "n": 49}],  # too few checked
        [{"verdict": "right", "n": 129}, {"verdict": "wrong_name", "n": 1}],  # one wrong name
        [{"verdict": "right", "n": 60}, {"verdict": "wrong_motion", "n": 40}],  # under 80 % right
        RuntimeError('relation "vision_motion_story_verdict" does not exist'),  # unreadable: closed
    ],
)
def test_gate_closed_or_unreadable_means_no_story_line(monkeypatch, verdicts) -> None:
    monkeypatch.delenv(STREET_MOTION_LINE_ENV, raising=False)
    conn = _Conn(_tables(verdicts))
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert not any("motion stories" in ln for ln in got.lines)
    assert "together_story_ids" not in conn.seen  # no story content is even read


def test_no_published_story_version_is_a_closed_gate(monkeypatch) -> None:
    monkeypatch.delenv(STREET_MOTION_LINE_ENV, raising=False)
    conn = _Conn({**_tables(PASSED), "max(rules_version)": [{"v": None}]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert not any("motion stories" in ln for ln in got.lines)


def test_flag_off_removes_the_story_line(monkeypatch) -> None:
    monkeypatch.setenv(STREET_MOTION_LINE_ENV, "false")
    conn = _Conn(_tables(PASSED))
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert not any("motion stories" in ln for ln in got.lines)
    assert "max(rules_version)" not in conn.seen


def test_flag_ships_on_in_every_brief_building_service() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    for service in ("orion-cortex-exec", "orion-hub"):
        lines = (repo / "services" / service / ".env_example").read_text().splitlines()
        assert f"{STREET_MOTION_LINE_ENV}=true" in lines, service


def test_a_broken_story_gate_costs_only_the_story_line(monkeypatch) -> None:
    """Review 2026-11-03: a bad verdict row must not wipe the caption/who/patio lines."""
    monkeypatch.delenv(STREET_MOTION_LINE_ENV, raising=False)
    unresolved = [{"observed_at": NOW - timedelta(minutes=3), "description": "a shape"}]
    conn = _Conn({**_tables([{"verdict": "right", "n": "lots"}]), "vision_unresolved": unresolved})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.read_ok
    assert any("could not name" in ln for ln in got.lines)
    assert not any("motion stories" in ln for ln in got.lines)


def test_a_bad_story_row_costs_only_the_story_line() -> None:
    unresolved = [{"observed_at": NOW - timedelta(minutes=3), "description": "a shape"}]
    lines = summarize_street(sightings=[], expectations=[], unresolved=unresolved, patio=None, now=NOW, tz=TZ,
                             stories=[object()])
    assert len(lines) == 1 and "could not name" in lines[0]


def test_story_line_is_last_so_a_length_cut_drops_it_first() -> None:
    patio = {"zone": "patio", "state": "present", "count": 3, "row_updated_at": datetime.now(timezone.utc)}
    lines = summarize_street(sightings=[], expectations=[], unresolved=[], patio=patio, now=NOW, tz=TZ,
                             stories=[_story("a", 5, S1)])
    assert lines[0] == "At least 3 people are on the patio."
    assert lines[-1].startswith("Walkway motion stories today")


def test_together_ids_shapes() -> None:
    assert perception_reader._ids(["a", "b"]) == {"a", "b"}
    assert perception_reader._ids('["a"]') == {"a"}
    assert perception_reader._ids('"abc"') == set()
    assert perception_reader._ids('{a,"b"}') == {"a", "b"}
    assert perception_reader._ids(None) == set() and perception_reader._ids(5) == set()
