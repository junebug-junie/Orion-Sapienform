"""Patch 4: the stance cue renders byte-exact from real frames, and renders nothing when it must."""
from __future__ import annotations

import re
from datetime import timedelta

from orion.schemas.temporal_self import STANCE_CUE_MAX_CHARS
from orion.temporal_self.stance_cue import project_stance_cue, span_words
from orion.temporal_self.tests.patch4_fixtures import BUSY_NOW, TALKING_NOW, frame, rest_frame

TALKING = (
    "Afternoon, Sun 17:58 local (day record as of 17:53). On now: talking with Juniper, last turn 42 min ago "
    "[arc:571bce17764b0a9e]. Before that: attention on Juniper is talking (1 message in the last 15 min), set "
    "aside 28 min ago [arc:4f06f3b115f9a3ac]. 2 open threads from conversation [arc:fd63c7ab0ad3ecbe]. Last "
    "slept 14 h 52 min ago [arc:460c1b680514c3e8]."
)
BUSY = (
    "Morning, Sun 11:20 local (day record as of 11:16). On now: attention on Bus synaptic prediction error for "
    "28 min, back to it 1 time today [arc:ae0a3cb57ae73ee4]. Before that: attention on Execution prediction "
    "error, ended 44 min ago [arc:0296c2551753e179]. Last slept 8 h 14 min ago [arc:460c1b680514c3e8]."
)


def _ids(text: str) -> list[str]:
    return re.findall(r"\[arc:([0-9a-f]{16})\]", text)


def test_talking_frame_renders_exactly():
    cue = project_stance_cue(frame("talking"), TALKING_NOW)
    assert cue.text == TALKING
    assert cue.skip_reason is None
    assert cue.arc_ids == _ids(TALKING)
    # Only labels Orion was shown are match labels (the sleep arc's label "pressure" is not shown).
    assert cue.labels == {"Juniper is talking (1 message in the last 15 min)": ["4f06f3b115f9a3ac"]}


def test_busy_frame_renders_exactly():
    cue = project_stance_cue(frame("busy"), BUSY_NOW)
    assert cue.text == BUSY
    assert set(cue.labels) == {"Bus synaptic prediction error", "Execution prediction error"}


def test_every_arc_in_a_render_is_in_the_input():
    for f, now in ((frame("talking"), TALKING_NOW), (frame("busy"), BUSY_NOW)):
        shown = {a["arc_id"] for a in f["arcs_today"]} | {f["active_arc"]["arc_id"], f["previous_arc"]["arc_id"]}
        assert set(_ids(project_stance_cue(f, now).text)) <= shown


def test_rest_state_renders_rest_not_nothing():
    cue = project_stance_cue(rest_frame(), TALKING_NOW.replace(hour=6, minute=6))
    assert cue.text == "Pre-dawn, Sun 00:06 local (day record as of 00:04). No attention or conversation arc is open right now."
    assert cue.skip_reason is None and cue.arc_ids == []


def test_stale_frame_renders_nothing():
    cue = project_stance_cue(frame("talking"), TALKING_NOW + timedelta(minutes=10))  # as_of is 23:53:23
    assert cue.text == "" and cue.skip_reason == "stale"
    assert project_stance_cue(frame("talking"), TALKING_NOW, max_age_sec=60).skip_reason == "stale"


def test_wrong_day_renders_nothing():
    cue = project_stance_cue(frame("talking"), TALKING_NOW + timedelta(hours=7))  # local 10-12 00:58
    assert cue.text == "" and cue.skip_reason == "wrong_day"


def test_missing_or_invalid_frame_renders_nothing():
    assert project_stance_cue(None, TALKING_NOW).model_dump(include={"text", "skip_reason"}) == {
        "text": "", "skip_reason": "fetch_failed"}
    bad = frame("talking") | {"unexpected_field": 1}
    assert project_stance_cue(bad, TALKING_NOW).skip_reason == "invalid_frame"


def test_silent_broadcast_drops_attention_clauses_keeps_the_rest():
    f = frame("busy")
    f["warnings"] = ["broadcast log silent for 1200 s"]
    cue = project_stance_cue(f, BUSY_NOW)
    assert "attention on" not in cue.text
    assert cue.text.endswith("Last slept 8 h 14 min ago [arc:460c1b680514c3e8].")
    assert cue.clauses_dropped == 2


def test_cap_drops_clauses_from_the_bottom():
    cue = project_stance_cue(frame("talking"), TALKING_NOW, max_chars=200)
    assert len(cue.text) <= 200 and cue.text.startswith("Afternoon, Sun 17:58 local")
    assert "Last slept" not in cue.text and cue.clauses_dropped >= 1
    assert len(project_stance_cue(frame("talking"), TALKING_NOW).text) <= STANCE_CUE_MAX_CHARS


def test_old_frame_without_last_seen_at_still_renders_honestly():
    f = frame("talking")
    for s in [f["active_arc"], f["previous_arc"], *f["arcs_today"]]:
        s.pop("last_seen_at", None)
    text = project_stance_cue(f, TALKING_NOW).text
    assert "On now: a conversation with Juniper, open for 1 h 55 min [arc:571bce17764b0a9e]." in text
    assert "Before that: attention on Juniper is talking (1 message in the last 15 min) [arc:4f06f3b115f9a3ac]." in text


def test_span_words():
    assert [span_words(s) for s in (5, 60, 25 * 60, 7200, 5 * 3600 + 13 * 60)] == [
        "under a minute", "1 min", "25 min", "2 h", "5 h 13 min"]
