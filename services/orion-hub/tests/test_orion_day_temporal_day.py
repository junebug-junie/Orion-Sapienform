"""Temporal Self patch 4 in the Hub's letter path: the closed day is gathered (or not) and the
reread tool shows the day-citation check, with UNVERIFIED for an uncited sentence about the day."""
from __future__ import annotations

import asyncio
import gzip
import json
from datetime import datetime, timezone

import scripts.orion_day_introspect as oi
from orion.orion_day import gather
from orion.orion_day.letter_parts import split_note
from orion.orion_day.tests import fixtures as fx
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.tests.patch4_fixtures import DATA, day_1009

HARNESS = "2ce0b486a356bbd2"
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)


def _day_json(day_id: str = fx.LETTER_DATE.isoformat()) -> dict:
    """The real 10-09 record, relabelled as the fixture letter's day (gather checks day_id)."""
    day = json.loads(gzip.decompress((DATA / "day_2026-10-09.json.gz").read_bytes()))
    day["day_id"] = day_id
    return day


def test_gather_reads_the_closed_day_by_local_date():
    conn = fx.FakeConn(temporal_day=_day_json())
    material = asyncio.run(gather.gather_orion_day(conn, fx.LETTER_DATE, now=NOW))
    assert [args for sql, args in conn.calls if sql == gather.TEMPORAL_DAY_SQL] == [(fx.LETTER_DATE.isoformat(),)]
    assert material.temporal_day.text.startswith("Closed day 2026-09-29")
    assert material.sources["temporal_day"].status == "ok"


def test_day_not_closed_renders_nothing_and_kill_switch_skips_the_read():
    material = asyncio.run(gather.gather_orion_day(fx.FakeConn(), fx.LETTER_DATE, now=NOW))
    assert material.temporal_day is None and material.sources["temporal_day"].status == "empty"
    conn = fx.FakeConn(temporal_day=_day_json())
    off = asyncio.run(gather.gather_orion_day(conn, fx.LETTER_DATE, now=NOW, temporal_day=False))
    assert off.temporal_day is None and "temporal_day" not in off.sources
    assert not [c for c in conn.calls if c[0] == gather.TEMPORAL_DAY_SQL]


def test_a_row_for_another_day_renders_nothing():
    conn = fx.FakeConn(temporal_day=_day_json("2026-10-09"))
    assert asyncio.run(gather.gather_orion_day(conn, fx.LETTER_DATE, now=NOW)).temporal_day is None


def test_a_failed_read_names_its_gap_and_never_cancels_the_letter():
    material = asyncio.run(gather.gather_orion_day(fx.FakeConn(fail={"temporal_day"}), fx.LETTER_DATE, now=NOW))
    assert material.temporal_day is None and material.sources["temporal_day"].status == "error"
    assert material.curiosity_runs  # the rest of the day is still there


def _letter(note: str):
    base = fx.reread_letter()
    material = base.material.model_copy(update={"temporal_day": render_closed_day_block(day_1009())})
    return base.model_copy(update={"material": material, "note_md": note})


def test_reread_part_shows_unverified_for_an_uncited_day_sentence():
    letter = _letter("Harness closure prediction error held me most of the day.\n\n"
                     f"Harness closure prediction error held me [arc:{HARNESS}].")
    first, second = [p for p in split_note(letter.note_md) if p.kind == "paragraph"]
    unverified = oi.part_item(letter, first, full=True).extra["day_citations"]
    assert unverified["label"] == "UNVERIFIED" and unverified["uncited_day_sentences"] == 1
    assert unverified["uncited_examples"] == ["Harness closure prediction error held me most of the day."]
    verified = oi.part_item(letter, second, full=True).extra["day_citations"]
    assert verified["label"] == "verified" and verified["cited_valid"] == [f"arc:{HARNESS}"]


def test_reread_without_a_block_is_unchanged():
    letter = fx.reread_letter()
    part = [p for p in split_note(letter.note_md) if p.kind == "paragraph"][0]
    assert "day_citations" not in oi.part_item(letter, part, full=True).extra
