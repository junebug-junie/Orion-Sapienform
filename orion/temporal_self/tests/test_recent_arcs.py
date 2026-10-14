"""Temporal Self rev 4: reverie's recent-arcs block renders from real frames, or renders nothing."""
from __future__ import annotations

import copy
import re
from datetime import timedelta

from orion.temporal_self.recent_arcs import MAX_LIMIT, RECENT_ARCS_MAX_CHARS, project_recent_arcs
from orion.temporal_self.tests.patch4_fixtures import BUSY_NOW, TALKING_NOW, frame, rest_frame


def _ids(text: str) -> list[str]:
    return re.findall(r"\[arc:([0-9a-f]+)\]", text)


def _closed_by_end(f: dict) -> list[str]:
    closed = [a for a in f["arcs_today"] if a["status"] == "closed" and a.get("ended_at")]
    closed.sort(key=lambda a: (a["ended_at"], a["arc_id"]), reverse=True)
    return [a["arc_id"] for a in closed]


def test_real_frame_lists_last_closed_arcs_most_recent_first():
    f = frame("talking")
    block = project_recent_arcs(f, TALKING_NOW)
    assert block.skip_reason is None
    expected = _closed_by_end(f)[:3]
    assert len(expected) == 3  # the fixture really has history
    assert block.arc_ids == expected
    assert _ids(block.text) == expected
    lines = block.text.splitlines()
    assert lines[0].startswith("Recently closed today")
    for line in lines[1:]:
        assert re.match(r"^- \d\d:\d\d-\d\d:\d\d \(.+\) .+ \[arc:[0-9a-f]+\]$", line), line


def test_closed_only():
    f = frame("busy")
    block = project_recent_arcs(f, BUSY_NOW, limit=5)
    status = {a["arc_id"]: a["status"] for a in f["arcs_today"]}
    assert block.arc_ids and all(status[i] == "closed" for i in block.arc_ids)
    # the open active arc is never listed
    assert f["active_arc"]["arc_id"] not in block.arc_ids


def test_local_time_and_duration_and_label_fallback():
    f = copy.deepcopy(frame("talking"))
    arc = dict(f["arcs_today"][0])
    arc.update(arc_id="aaaaaaaaaaaaaaaa", kind="attention", status="closed", subject_label="  ",
               subject_ref="node:x", began_at="2026-10-11T23:00:00Z", ended_at="2026-10-11T23:57:00Z")
    f["arcs_today"][0] = arc
    block = project_recent_arcs(f, TALKING_NOW, limit=1)
    # America/Denver is UTC-6 in October (MDT)
    assert block.text.splitlines()[1] == "- 17:00-17:57 (57 min) attention on node:x [arc:aaaaaaaaaaaaaaaa]"


def test_limit_capped_at_five_and_chars_capped():
    f = frame("talking")
    block = project_recent_arcs(f, TALKING_NOW, limit=50)
    assert len(block.arc_ids) <= MAX_LIMIT
    assert len(block.text) <= RECENT_ARCS_MAX_CHARS
    tight = project_recent_arcs(f, TALKING_NOW, limit=5, max_chars=150)
    assert len(tight.text) <= 150 and tight.arc_ids
    assert set(_ids(tight.text)) == set(tight.arc_ids)  # every listed arc is cited, nothing more


def test_single_overlong_line_keeps_its_citation():
    f = copy.deepcopy(frame("talking"))
    arc = dict(f["arcs_today"][0])
    arc.update(arc_id="bbbbbbbbbbbbbbbb", kind="attention", status="closed", subject_label="x" * 600,
               began_at="2026-10-11T23:00:00Z", ended_at="2026-10-11T23:58:00Z")
    f["arcs_today"][0] = arc
    block = project_recent_arcs(f, TALKING_NOW, limit=1)
    assert len(block.text) <= RECENT_ARCS_MAX_CHARS
    assert block.text.endswith("[arc:bbbbbbbbbbbbbbbb]")


def test_stale_frame_renders_nothing():
    block = project_recent_arcs(frame("talking"), TALKING_NOW + timedelta(seconds=901 + 300))
    assert block.text == "" and block.skip_reason == "stale"


def test_wrong_day_renders_nothing():
    block = project_recent_arcs(frame("talking"), TALKING_NOW + timedelta(days=1))
    assert block.text == "" and block.skip_reason == "wrong_day"


def test_future_frame_renders_nothing():
    block = project_recent_arcs(frame("talking"), TALKING_NOW - timedelta(minutes=30))
    assert block.text == "" and block.skip_reason == "clock_skew"


def test_garbage_renders_nothing():
    assert project_recent_arcs(None, TALKING_NOW).skip_reason == "fetch_failed"
    assert project_recent_arcs({"nope": 1}, TALKING_NOW).skip_reason == "invalid_frame"
    assert project_recent_arcs({"nope": 1}, TALKING_NOW).text == ""


def test_no_closed_arcs_renders_nothing():
    f = rest_frame()
    from datetime import datetime, timezone
    block = project_recent_arcs(f, datetime(2026, 10, 11, 6, 5, tzinfo=timezone.utc))
    assert block.text == "" and block.skip_reason == "no_closed_arcs"


def test_tiny_cap_renders_nothing_rather_than_overrunning():
    for cap in (10, 30, 60):
        block = project_recent_arcs(frame("talking"), TALKING_NOW, limit=1, max_chars=cap)
        assert len(block.text) <= cap
        assert block.text == "" and block.skip_reason == "cap_too_small"
