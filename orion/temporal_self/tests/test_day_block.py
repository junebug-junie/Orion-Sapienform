"""Patch 4: the closed-day block for Orion's Day letter, byte-exact on the real 2026-10-09 record."""
from __future__ import annotations

import re

from orion.schemas.temporal_self import DAY_BLOCK_MAX_CHARS
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.tests.patch4_fixtures import day_1009

BLOCK_1009 = """Closed day 2026-10-09 (America/Denver, midnight to midnight), as recorded at close, 2026-10-10 00:00 local: rows that arrived later are not in it. 291 arcs: 49 attention, 2 conversation, 8 curiosity, 2 sleep, 2 concern, 184 reverie, 44 interoception [day:2026-10-09].
Talked with Juniper: 15:00 [arc:2545de72655edb66]; 18:08 [arc:b320b6c7e1308151].
Attention, most time first:
- Harness closure prediction error: 2 h 15 min over 11 arcs, 11 returns [arc:2ce0b486a356bbd2].
- Execution prediction error: 2 h 14 min over 16 arcs, 13 returns [arc:7bf5c22449afc56d].
- Chat prediction error: 1 h 17 min over 8 arcs, 3 returns [arc:e5f240a28cd2b208].
- Codebase prediction error: 53 min over 7 arcs, 0 returns [arc:fda9750a570168c3].
- Route prediction error: 23 min over 4 arcs, 1 return [arc:f6ddd30f4a7edd2b].
Curiosity runs (8):
- 10-08 13:56, 20 h 35 min (process longer than 4 h: its start column may predate a stall or retry) [arc:a6e49188fd4f7fa1].
- 08:05, 51 min [arc:9d0862f139025ffb].
- 08:58, 55 min [arc:d3a03f7df915be41].
- 10:31, 34 min [arc:6f9949a7b73969e3].
- 11:06, 27 min [arc:3909798274bc18f6].
- 12:29, 1 h [arc:8fb365f64dbea05c].
- 13:29, 1 h 7 min [arc:45ad862f4fce6b57].
- 15:05, 59 min [arc:4ed630d33f1480f1].
Sleep cycles: 00:33 [arc:3a53de69df46dfec]; 19:01 [arc:5e58443ac55520ac].
Expectations resolved: 331 (of the 256 listed: 251 unscored, 4 confirmed, 1 claim_upheld=false). 8 still pending [day:2026-10-09].
Refused or made to wait: 52 gpu_wait, 14 visual_deferral [day:2026-10-09].
Conversation threads: open-loop-5cb6cd032923, 4 returns, day_boundary [arc:1e63167741582226]; open-loop-639498bb10c0, 0 returns, day_boundary [arc:5afab5a718d5b81c].
Instrument warnings: broadcast log gaps today: 2 (recorder outage or restart) [day:2026-10-09] | curiosity arc a6e49188fd4f7fa1 (ca97ade74c1d): process longer than 4 h: its start column may predate a stall or retry [arc:a6e49188fd4f7fa1]."""


def test_closed_day_renders_exactly_and_under_cap():
    block = render_closed_day_block(day_1009())
    assert block.text == BLOCK_1009
    assert len(block.text) <= DAY_BLOCK_MAX_CHARS and block.lines_dropped == 0


def test_first_line_says_frozen_at_close():
    first = render_closed_day_block(day_1009()).text.splitlines()[0]
    assert "as recorded at close" in first and "rows that arrived later are not in it" in first


def test_every_ref_resolves_to_a_record_from_the_day():
    day = day_1009()
    block = render_closed_day_block(day)
    refs = re.findall(r"\[((?:arc|day):[^\]]+)\]", block.text)
    assert refs and set(refs) == set(block.records)
    stored = {a.arc_id for a in day.arcs}
    for ref in refs:
        if ref.startswith("arc:"):
            assert ref[4:] in stored
            assert block.records[ref]["arc_id"] == ref[4:]
    assert block.records["day:2026-10-09"]["arcs_total"] == 291


def test_every_line_cites_something():
    for line in render_closed_day_block(day_1009()).text.splitlines():
        if line.endswith(":"):  # a list heading; its items cite
            continue
        assert re.search(r"\[(arc|day):", line), line


def test_labels_are_rendered_text_only():
    block = render_closed_day_block(day_1009())
    assert block.labels["Harness closure prediction error"] == ["2ce0b486a356bbd2"]
    assert block.labels["15:00"] == ["2545de72655edb66"]
    assert all(label in block.text for label in block.labels)


def test_missing_day_renders_nothing():
    assert render_closed_day_block(None) is None


def test_cap_drops_lowest_priority_lines_and_says_so():
    block = render_closed_day_block(day_1009(), max_chars=900)
    assert len(block.text) <= 900 and block.lines_dropped > 0
    assert block.text.startswith("Closed day 2026-10-09")
    assert "left out to fit) [day:2026-10-09]" in block.text.splitlines()[-1]
    assert set(re.findall(r"\[((?:arc|day):[^\]]+)\]", block.text)) == set(block.records)


def test_empty_closed_day_is_a_real_record_not_a_placeholder():
    day = day_1009().model_copy(update={"arcs": []})
    day = day.model_copy(update={"frame": day.frame.model_copy(update={
        "expectations_resolved_today": [], "expectations_resolved_total": 0, "expectations_pending": [],
        "constraint_event_ids": [], "warnings": []})})
    text = render_closed_day_block(day).text
    assert text.splitlines()[1] == "No arcs were recorded on this day [day:2026-10-09]."
