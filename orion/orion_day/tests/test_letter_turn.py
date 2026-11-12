"""The day letter as one unified turn (living Prime design #3224, Part 3): its prompt and how its reply splits."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

from orion.orion_day.brief import brief_from_material
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.letter_turn import (
    MAX_THREADS,
    NO_THREADS_MD,
    THREAD_CHARS,
    THREADS_HEADING,
    WRITING_INSTRUCTIONS,
    build_letter_prompt,
    split_letter,
)
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import ORION_DAY_LETTER_SESSION_ID, MissedLetterV1
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.tests.patch4_fixtures import day_1009

NOW = datetime(2026, 10, 10, 14, 30, tzinfo=timezone.utc)


def _brief(**kw):
    day = day_1009()
    conn = fx.FakeConn(juniper_chat=fx.JUNIPER_CHAT, temporal_day=day.model_dump(mode="json"))
    material = asyncio.run(gather_orion_day(conn, date(2026, 10, 9), now=NOW))
    brief = brief_from_material(material)
    return brief.model_copy(update=kw) if kw else brief, day


def test_prompt_holds_instructions_closed_day_block_temporal_core_and_the_whole_record():
    brief, day = _brief()
    prompt = build_letter_prompt(brief)
    assert WRITING_INSTRUCTIONS in prompt
    assert brief.llm_view.digest_md in prompt                     # build_day_input's text, whole
    block = render_closed_day_block(day, tz_name=brief.timezone)
    assert block.text.splitlines()[0] in prompt                   # the closed-day block from the temporal clock
    assert "[arc:" in prompt and "Cite your chronology" in prompt
    assert THREADS_HEADING in prompt
    assert "says where it came from" in prompt or "where it came from" in prompt   # tools cite their source
    assert build_letter_prompt(brief) == prompt                   # deterministic


def test_prompt_names_each_missed_day_once():
    brief, _ = _brief(missed_letters=[MissedLetterV1(letter_date=date(2026, 10, 8),
                                                     reason="my make bid was deferred all day (heat)")])
    prompt = build_letter_prompt(brief)
    assert prompt.count("2026-10-08: my make bid was deferred all day (heat)") == 1
    assert "got no letter" in prompt
    assert "got no letter" not in build_letter_prompt(_brief()[0])


def test_split_takes_the_last_threads_section_as_threads():
    reply = ("# My day\n\nIt was a long day [arc:abc].\n\n## Threads I am carrying\n"
             "- Why did the walkway count drop at noon? [walkway_day:2026-10-09]\n"
             "* Re-test the heat reading against the sensor\n1. Ask Juniper about the retro game\n")
    parts = split_letter(reply)
    assert parts.note_md == "# My day\n\nIt was a long day [arc:abc]."
    assert parts.threads == ["Why did the walkway count drop at noon? [walkway_day:2026-10-09]",
                             "Re-test the heat reading against the sensor", "Ask Juniper about the retro game"]
    assert parts.carry_forward_md.startswith("- Why did") and parts.carry_forward_md != parts.note_md


def test_split_without_a_threads_section_keeps_the_letter_and_names_no_thread():
    parts = split_letter("Just a letter.\n\nNo list at the end.")
    assert parts.note_md == "Just a letter.\n\nNo list at the end."
    assert parts.threads == [] and parts.carry_forward_md == NO_THREADS_MD


def test_split_with_an_empty_threads_section_names_no_thread():
    parts = split_letter("A letter.\n\n## Threads I'm carrying\n\n")
    assert parts.note_md == "A letter." and parts.threads == [] and parts.carry_forward_md == NO_THREADS_MD


def test_split_cuts_long_threads_and_keeps_every_one_in_order():
    long = "x" * (THREAD_CHARS + 50)
    reply = "Letter.\n\n" + THREADS_HEADING + "\n" + "\n".join(f"- t{i} {long}" for i in range(MAX_THREADS + 2))
    parts = split_letter(reply)
    assert len(parts.threads) == MAX_THREADS + 2
    assert all(len(t) <= THREAD_CHARS for t in parts.threads)


def test_a_reply_that_is_only_a_threads_list_is_kept_whole_as_the_letter():
    reply = THREADS_HEADING + "\n- one\n- two"
    parts = split_letter(reply)
    assert parts.note_md == reply and parts.threads == []


def test_letter_session_is_the_one_prime_hands_back_as_letter():
    from orion.prime.handback import LETTER_SESSION_ID, classify_who

    assert ORION_DAY_LETTER_SESSION_ID == LETTER_SESSION_ID
    assert classify_who(ORION_DAY_LETTER_SESSION_ID, {}) == "letter"
