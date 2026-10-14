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


def _with_daydreams(n: int, *, start_hour: int = 22) -> dict:
    """The talking fixture plus ``n`` closed reverie arcs, 10 min apart, the last ending latest."""
    f = copy.deepcopy(frame("talking"))
    template = dict(f["arcs_today"][0])
    f["arcs_today"] = f["arcs_today"][: len(f["arcs_today"]) - n]  # stay under ARCS_IN_FRAME_CAP
    for i in range(n):
        arc = dict(template)
        m = i * 10
        arc.update(arc_id=f"{i:016x}"[-16:].replace("0", "d"), kind="reverie", status="closed",
                   subject_label="", subject_ref=f"open-loop-{i}",
                   began_at=f"2026-10-11T{start_hour + m // 60:02d}:{m % 60:02d}:00Z",
                   ended_at=f"2026-10-11T{start_hour + m // 60:02d}:{m % 60:02d}:08Z")
        f["arcs_today"].append(arc)
    f["arcs_today_total"] = len(f["arcs_today"])
    return f


def test_daydreams_grouped_into_one_line_plus_two_other_arcs():
    f = _with_daydreams(5)
    block = project_recent_arcs(f, TALKING_NOW)
    lines = block.text.splitlines()[1:]
    dream_lines = [ln for ln in lines if "daydreaming" in ln]
    assert len(dream_lines) == 1, block.text
    assert "(5 daydreams)" in dream_lines[0] and "22:00" not in dream_lines[0]  # local time, not UTC
    assert dream_lines[0].startswith("- 16:00-16:40 ")  # first began 22:00Z, last ended 22:40Z (MDT)
    # the group cites only its latest daydream; the other two lines are the newest non-reverie arcs
    reverie_ids = {a["arc_id"] for a in f["arcs_today"] if a["kind"] == "reverie"}
    latest = max((a for a in f["arcs_today"] if a["kind"] == "reverie"), key=lambda a: a["ended_at"])
    assert [i for i in block.arc_ids if i in reverie_ids] == [latest["arc_id"]]
    others = [i for i in block.arc_ids if i not in reverie_ids]
    non_reverie = [i for i in _closed_by_end(f) if i not in reverie_ids]
    assert others == non_reverie[:2]
    assert len(lines) == 3
    for line in lines:  # same shape as every other line
        assert re.match(r"^- \d\d:\d\d-\d\d:\d\d \(.+\) .+ \[arc:[0-9a-f]+\]$", line), line
    assert set(_ids(block.text)) == set(block.arc_ids)


def test_one_daydream_is_singular_and_truncated_frame_says_at_least():
    f = _with_daydreams(1)
    assert "(1 daydream)" in project_recent_arcs(f, TALKING_NOW).text
    f["arcs_today_total"] = len(f["arcs_today"]) + 40  # the frame dropped arcs (reverie first)
    line = [ln for ln in project_recent_arcs(f, TALKING_NOW).text.splitlines() if "daydream" in ln][0]
    assert "(at least 1 daydream)" in line
    assert line.startswith("- up to 16:00 ")  # oldest daydreams were dropped: no claimed start time


def test_no_daydreams_lists_limit_other_arcs():
    f = frame("talking")
    assert not any(a["kind"] == "reverie" for a in f["arcs_today"])
    assert len(project_recent_arcs(f, TALKING_NOW).arc_ids) == 3


def test_only_daydreams_renders_the_group_line_alone():
    f = _with_daydreams(4)
    f["arcs_today"] = [a for a in f["arcs_today"] if a["kind"] == "reverie"]
    f["arcs_today_total"] = len(f["arcs_today"])
    for k in ("active_arc",):
        f[k] = None
    lines = project_recent_arcs(f, TALKING_NOW).text.splitlines()
    assert len(lines) == 2 and "(4 daydreams)" in lines[1]
