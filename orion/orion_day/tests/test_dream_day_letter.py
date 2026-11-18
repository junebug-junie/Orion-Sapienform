"""Last night's day dream in the day letter (day-dream spec Patch 3, 2026-11-18 hardening): a bad night is visible.

The letter's input gets one section right before the organ roundup. A night that produced shows "Last night I
slept on the day", the day paragraph and the written memory ids, citable as [dream_day:<day>]. Any other night
shows one line, "no day dream for <day>: <reason>" (no row, a row with no memory and its note, the run failed, or
the row could not be read). The letter prompt asks for the ref only when the section is there."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.orion_day import day_input as di
from orion.orion_day.brief import brief_from_material
from orion.orion_day.budget import material_ref_records
from orion.orion_day.gather import DREAM_DAY_FAILED_SQL, gather_orion_day
from orion.orion_day.letter_parts import record_text
from orion.orion_day.letter_turn import CITE_DREAM_DAY, build_letter_prompt
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import OrganRoundupV1

NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
DAY = fx.LETTER_DATE.isoformat()
PARAGRAPH = "A quiet day of reading; my one sleep linked nothing. [experience:dream_cycle:dc-1]"
PRODUCED = {"run_id": f"dream-day-{DAY}", "produced": True, "note": "2 memories written", "day_text": PARAGRAPH,
            "day_refs": ["experience:dream_cycle:dc-1"], "memory_ids": ["m-1", "m-2"]}
NO_TRACE = {**PRODUCED, "produced": False, "note": "no lasting trace", "memory_ids": []}
FAILED = {"run_id": f"dream-day-{DAY}", "generated_at": NOW, "error": "dream_day_no_answer: no answer block"}


def _material(**kw):
    m = asyncio.run(gather_orion_day(fx.FakeConn(**kw), fx.LETTER_DATE, now=NOW))
    return m.model_copy(update={"organ_roundup": OrganRoundupV1(day=fx.LETTER_DATE, lines=["organs: 1 organ"])})


def test_gather_reads_the_night_in_every_state():
    conn = fx.FakeConn(dream_day=PRODUCED)
    m = asyncio.run(gather_orion_day(conn, fx.LETTER_DATE, now=NOW))
    assert m.dream_day.status == "produced" and m.dream_day.memory_ids == ["m-1", "m-2"]
    assert m.sources["dream_day"].status == "ok"
    assert not [a for sql, a in conn.calls if sql == DREAM_DAY_FAILED_SQL]   # a row: never looks for a failure
    assert _material(dream_day=NO_TRACE).dream_day.status == "not_produced"
    failed = fx.FakeConn(dream_day_failed=FAILED)
    m = asyncio.run(gather_orion_day(failed, fx.LETTER_DATE, now=NOW))
    assert m.dream_day.status == "failed" and "no answer block" in m.dream_day.error
    assert [a for sql, a in failed.calls if sql == DREAM_DAY_FAILED_SQL] == [(f"dream-day-{DAY}%",)]
    assert _material().dream_day.status == "no_row"
    broken = _material(fail={"dream_day"})
    assert broken.dream_day is None and broken.sources["dream_day"].status == "error"


def test_a_produced_night_sits_before_the_roundup_with_its_paragraph_ids_and_ref():
    m = _material(dream_day=PRODUCED)
    view = di.render_day_input(m)
    d = view.digest_md
    assert d.index(di.DREAM_DAY_HEADING) < d.index(di.ROUNDUP_HEADING)
    assert d.index("## Conversations with Juniper") < d.index(di.DREAM_DAY_HEADING)
    assert f"{di.DREAM_DAY_HEADING} [dream_day:{DAY}]" in d
    assert "A quiet day of reading; my one sleep linked nothing. (experience:dream_cycle:dc-1)" in d
    assert "Memories I wrote: m-1, m-2" in d and "no day dream" not in d
    assert f"dream_day:{DAY}" in view.included_refs and view.source_chars["dream_day"] > 0
    # The ref resolves to the stored night, and reads as its paragraph.
    rec = material_ref_records(m)[f"dream_day:{DAY}"]
    assert rec["run_id"] == f"dream-day-{DAY}" and PARAGRAPH in record_text(rec)
    assert CITE_DREAM_DAY in build_letter_prompt(brief_from_material(m))


def test_a_bad_night_is_one_line_never_silent_and_never_citable():
    cases = {
        "no lasting trace": _material(dream_day=NO_TRACE),
        "the night's run failed (dream_day_no_answer: no answer block)": _material(dream_day_failed=FAILED),
        "no dream_day row (the night did not run)": _material(),
        "could not read it (RuntimeError: relation dream_day does not exist)": _material(fail={"dream_day"}),
    }
    for reason, m in cases.items():
        view = di.render_day_input(m)
        d = view.digest_md
        line = di.dream_day_line(m)
        assert line.startswith(f"no day dream for {DAY}: ") and reason in line, line
        assert f"{di.DREAM_DAY_HEADING}\n{line}" in d
        assert d.index(di.DREAM_DAY_HEADING) < d.index(di.ROUNDUP_HEADING)
        assert f"dream_day:{DAY}" not in view.included_refs and f"dream_day:{DAY}" not in material_ref_records(m)
        assert CITE_DREAM_DAY in build_letter_prompt(brief_from_material(m))


def test_an_older_brief_without_the_field_renders_nothing():
    m = _material().model_copy(update={"dream_day": None, "sources": {
        k: v for k, v in _material().sources.items() if k != "dream_day"}})
    assert di.dream_day_line(m) is None
    d = di.render_day_input(m).digest_md
    assert di.DREAM_DAY_HEADING not in d and CITE_DREAM_DAY not in build_letter_prompt(brief_from_material(m))
