"""Patch 4: the closed-day block for Orion's Day letter, byte-exact on the real 2026-10-09 record."""
from __future__ import annotations

import re

from orion.schemas.temporal_self import DAY_BLOCK_MAX_CHARS
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.tests.patch4_fixtures import day_1009

BLOCK_1009 = """Closed day 2026-10-09 (America/Denver, midnight to midnight), as recorded at close, 2026-10-10 00:00 local: rows that arrived later are not in it. 291 arcs: 49 attention, 2 conversation, 8 curiosity, 2 sleep, 2 concern, 184 reverie, 44 interoception [day:2026-10-09].
Talked with Juniper: 15:00 [arc:2545de72655edb66]; 18:08 [arc:b320b6c7e1308151].
Attention, most time first:
- Harness closure prediction error: 2 h 15 min over 11 arcs [day:2026-10-09]; longest 39 min, 3 returns [arc:2ce0b486a356bbd2].
- Execution prediction error: 2 h 14 min over 16 arcs [day:2026-10-09]; longest 31 min, 3 returns [arc:7bf5c22449afc56d].
- Chat prediction error: 1 h 17 min over 8 arcs [day:2026-10-09]; longest 19 min, 3 returns [arc:e5f240a28cd2b208].
- Codebase prediction error: 53 min over 7 arcs [day:2026-10-09]; longest 14 min, 0 returns [arc:fda9750a570168c3].
- Route prediction error: 23 min over 4 arcs [day:2026-10-09]; longest 16 min, 1 return [arc:f6ddd30f4a7edd2b].
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


def test_attention_line_numbers_match_what_they_cite():
    """#2623 truth eval: a group total cites the day (whose record holds it), the cited arc's own
    dwell and returns cite the arc. A one-arc group states only that arc's numbers."""
    day = day_1009()
    block = render_closed_day_block(day)
    by_id = {a.arc_id: a for a in day.arcs}
    line = next(l for l in block.text.splitlines() if l.startswith("- Harness closure prediction error:"))
    assert by_id["2ce0b486a356bbd2"].attention_returns == 3 and "3 returns [arc:2ce0b486a356bbd2]" in line
    group = block.records["day:2026-10-09"]["attention_groups"][by_id["2ce0b486a356bbd2"].subject_ref]
    assert group["arcs"] == 11 and "over 11 arcs [day:2026-10-09]" in line
    assert len(block.records["day:2026-10-09"]["attention_groups"]) == 5

    one = day.model_copy(update={"arcs": [by_id["2ce0b486a356bbd2"]]})
    solo = render_closed_day_block(one).text.splitlines()
    assert "- Harness closure prediction error: 39 min, 3 returns [arc:2ce0b486a356bbd2]." in solo


def test_attention_groups_follow_the_lines_that_survive_the_cap():
    day = day_1009()
    block = render_closed_day_block(day, max_chars=1000)
    groups = block.records["day:2026-10-09"]["attention_groups"]
    assert groups, block.text
    for subject_ref in groups:
        label = next(a.subject_label for a in day.arcs if a.subject_ref == subject_ref and a.kind == "attention")
        assert any(l.startswith(f"- {label}:") for l in block.text.splitlines())
    assert len(groups) < 5


def test_same_label_groups_keep_only_the_one_whose_line_survives():
    day = day_1009()
    by_id = {a.arc_id: a for a in day.arcs}
    harness_ref = by_id["2ce0b486a356bbd2"].subject_ref
    route_ref = by_id["f6ddd30f4a7edd2b"].subject_ref
    # Give the Route group the Harness label: same text, different subject.
    arcs = [a.model_copy(update={"subject_label": by_id["2ce0b486a356bbd2"].subject_label})
            if a.subject_ref == route_ref else a for a in day.arcs]
    full = render_closed_day_block(day.model_copy(update={"arcs": arcs}))
    assert {harness_ref, route_ref} <= set(full.records["day:2026-10-09"]["attention_groups"])
    capped = render_closed_day_block(day.model_copy(update={"arcs": arcs}), max_chars=1000)
    kept = capped.records["day:2026-10-09"]["attention_groups"]
    assert harness_ref in kept and route_ref not in kept


# --- R4 immune findings ------------------------------------------------------------------------

def _finding(node="node:substrate.route", channel="prediction_error", kind="decayed_to_zero",
             start="2026-10-09T09:10:00+00:00", end="2026-10-09T15:40:00+00:00"):
    return {"metric_id": f"{node}.{channel}", "node": node, "channel": channel, "kind": kind,
            "window_start": start, "window_end": end, "evidence": {"ratio": 0.92}}


def test_no_immune_findings_renders_nothing_new():
    assert render_closed_day_block(day_1009(), immune_findings=[]).text == BLOCK_1009
    assert render_closed_day_block(day_1009(), immune_findings=None).text == BLOCK_1009
    assert "gauge" not in BLOCK_1009


def test_immune_finding_renders_one_plain_cited_sentence_in_local_time():
    block = render_closed_day_block(day_1009(), immune_findings=[_finding()])
    lines = [ln for ln in block.text.splitlines() if "gauge" in ln]
    # 09:10Z..15:40Z is 03:10..09:40 in America/Denver (MDT).
    assert lines == ["My route prediction error gauge looked stuck (decayed to zero) from 03:10 to 09:40 "
                     "[day:2026-10-09]."]
    rec = block.records["day:2026-10-09"]["immune_findings"]
    assert rec == [{"metric_id": "node:substrate.route.prediction_error", "kind": "decayed_to_zero",
                    "window_start": "2026-10-09T09:10:00+00:00", "window_end": "2026-10-09T15:40:00+00:00"}]


def test_immune_findings_cap_at_five_then_and_n_more():
    many = [_finding(node=f"node:substrate.n{i}", kind="constant",
                     start=f"2026-10-09T0{i}:00:00+00:00") for i in range(8)]
    block = render_closed_day_block(day_1009(), immune_findings=many)
    text = block.text
    assert len(text) <= DAY_BLOCK_MAX_CHARS and block.lines_dropped > 0   # lower-priority lines yield first
    assert sum("gauge looked stuck (constant)" in ln for ln in text.splitlines()) == 5
    assert "And 3 more stuck gauges [day:2026-10-09]." in text


def test_malformed_immune_rows_are_dropped_not_invented():
    junk = [None, "x", {"kind": "constant"}, _finding(kind="made_up"), _finding(start="not a date"),
            _finding(start="2026-10-09T09:10:00")]  # naive timestamp
    assert render_closed_day_block(day_1009(), immune_findings=junk).text == BLOCK_1009
