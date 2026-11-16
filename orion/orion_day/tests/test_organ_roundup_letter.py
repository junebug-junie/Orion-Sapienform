"""The daily organ roundup in the day letter (integration patch 4, spec Part A.7 "Where it goes"):
gathered only when asked, placed after Juniper's conversations, replacing the tail's readings, dreams,
street and visual reveries, its refs citable, and the letter prompt asking for them."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.dig import organs
from orion.orion_day import day_input as di
from orion.orion_day.brief import brief_from_material
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.letter_turn import CITE_ROUNDUP, build_letter_prompt
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import OrganRoundupV1, OrionDayLetterV1, OrionDaySourceStatusV1

NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
ROUNDUP_LINES = [
    "organs 2026-09-29 (Denver): 16 organs, 7 experiences",
    "dream: 2 experiences (dream_cycle 2)",
    '  "dream cycle: pressure" [experience:dream_cycle:dc-1] salience none',
    "  break: 0 dream_hypothesis today; usual 8 a day (median of the 14 days before)",
    "experience use rate: 1 / 7 = 14.29% (cited by another organ within 24 h); closed organs: 1 of 16 (dream)",
]


CARRIED = {"reading": 4, "dream": 2, "painting": 13, "camera": 571}


class _Conn(fx.FakeConn):
    """The fixture day plus the roundup's queries (served empty, so every organ is quiet)."""

    def __init__(self, **kw):
        super().__init__(juniper_chat=fx.JUNIPER_CHAT, **kw)
        self.roundup_sql = set(organs.queries(fx.LETTER_DATE).values())
        self.roundup_calls = 0

    async def fetch(self, sql, *args):
        if sql in self.roundup_sql:
            self.roundup_calls += 1
            return []
        return await super().fetch(sql, *args)


def _material(roundup: bool = True):
    m = asyncio.run(gather_orion_day(_Conn(), fx.LETTER_DATE, now=NOW))
    if not roundup:
        return m
    return m.model_copy(update={"organ_roundup": OrganRoundupV1(
        day=fx.LETTER_DATE, lines=ROUNDUP_LINES, refs=["experience:dream_cycle:dc-1"],
        organ_experiences=CARRIED)})


def test_gather_reads_the_roundup_only_when_asked():
    off = _Conn()
    m = asyncio.run(gather_orion_day(off, fx.LETTER_DATE, now=NOW))
    assert off.roundup_calls == 0 and m.organ_roundup is None and "organ_roundup" not in m.sources
    on = _Conn()
    m = asyncio.run(gather_orion_day(on, fx.LETTER_DATE, now=NOW, organ_roundup=True))
    assert on.roundup_calls == len(organs.queries(fx.LETTER_DATE))
    assert m.organ_roundup is not None and m.organ_roundup.day == fx.LETTER_DATE and not m.organ_roundup.unread
    assert m.sources["organ_roundup"] == OrionDaySourceStatusV1(status="ok", count=1)
    organs.check_lines(m.organ_roundup.lines)


def test_roundup_sits_after_juniper_and_before_curiosity_whole():
    view = di.render_day_input(_material())
    d = view.digest_md
    assert d.index("## Conversations with Juniper") < d.index(di.ROUNDUP_HEADING) < d.index("## Curiosity")
    for line in ROUNDUP_LINES:
        assert line in d  # never clipped
    assert view.source_chars["roundup"] > 0
    assert "experience:dream_cycle:dc-1" in view.included_refs
    assert not [r for r in view.included_refs if "<" in r]  # the heading plants no placeholder ref


def test_roundup_replaces_the_tails_readings_dreams_street_and_visual_reveries():
    with_r = di.render_day_input(_material()).digest_md
    without = di.render_day_input(_material(roundup=False)).digest_md
    for heading in ("## Readings (", "## Dreams (", "## Visual reveries ("):
        assert heading in without and heading not in with_r, heading
    assert "## World news" in with_r  # the rest of the tail stays
    assert di.ROUNDUP_HEADING not in without


def test_an_empty_roundup_replaces_nothing():
    m = _material(roundup=False)
    m = m.model_copy(update={"organ_roundup": OrganRoundupV1(day=fx.LETTER_DATE)})
    d = di.render_day_input(m).digest_md
    assert di.ROUNDUP_HEADING not in d and "## Readings (" in d


def test_letter_prompt_asks_to_cite_roundup_items_only_when_the_roundup_is_there():
    with_r = build_letter_prompt(brief_from_material(_material()))
    without = build_letter_prompt(brief_from_material(_material(roundup=False)))
    assert CITE_ROUNDUP in with_r and "your organ roundup" in with_r
    assert CITE_ROUNDUP not in without and "the street outside" in without


def test_letter_row_round_trips_the_roundup():
    m = _material()
    letter = fx.reread_letter(material=m)
    again = OrionDayLetterV1.model_validate(letter.model_dump(mode="json"))
    assert again.material.organ_roundup.lines == ROUNDUP_LINES


def test_a_roundup_whose_experience_reads_failed_does_not_displace_the_tail():
    m = _material().model_copy(update={"organ_roundup": OrganRoundupV1(
        day=fx.LETTER_DATE, lines=ROUNDUP_LINES, unread={"top": "RuntimeError: permission denied"})})
    d = di.render_day_input(m).digest_md
    assert di.ROUNDUP_HEADING in d and "## Readings (" in d and "## Dreams (" in d


def test_a_tail_section_stays_when_its_organ_carried_nothing_that_day():
    # Chronicle lagging at 08:30 or ingest broken: every organ quiet -> the letter keeps readings and dreams.
    quiet = _material().model_copy(update={"organ_roundup": OrganRoundupV1(
        day=fx.LETTER_DATE, lines=ROUNDUP_LINES, organ_experiences={})})
    d = di.render_day_input(quiet).digest_md
    assert "## Readings (" in d and "## Dreams (" in d and "## Visual reveries (" in d
    only_dreams = _material().model_copy(update={"organ_roundup": OrganRoundupV1(
        day=fx.LETTER_DATE, lines=ROUNDUP_LINES, organ_experiences={"dream": 2})})
    d = di.render_day_input(only_dreams).digest_md
    assert "## Dreams (" not in d and "## Readings (" in d


def test_roundup_refs_resolve_for_grounding_and_reread():
    from orion.orion_day.budget import material_ref_records
    from orion.orion_day.evals.run_orion_day_eval import evaluate

    m = _material()
    rec = material_ref_records(m)["experience:dream_cycle:dc-1"]
    assert rec["line"].startswith('"dream cycle: pressure"') and rec["roundup_day"] == fx.LETTER_DATE.isoformat()
    view = di.render_day_input(m)
    assert all(r in material_ref_records(m) for r in view.included_refs)
    results = {name: (ok, detail) for name, ok, detail in evaluate(m)}
    assert results["grounding"][0], results["grounding"][1]
    assert results["blind"][0], results["blind"][1]
