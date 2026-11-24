"""Core pages: one length rule on every page, and an over-cap retry that names the word count."""
from __future__ import annotations

from datetime import date

from orion.memory.pages import juniper, lately, origin
from orion.memory.pages.fold import (
    TARGET_SENTENCES, TARGET_WORDS, WORD_CAP, length_rule, retry_note, with_retry_note,
)


def test_every_page_prompt_uses_the_shared_length_rule():
    assert TARGET_WORDS < WORD_CAP
    rule = length_rule()
    assert f"About {TARGET_WORDS} words, in at most {TARGET_SENTENCES} sentences" in rule and f"over {WORD_CAP} words is thrown away" in rule
    assert rule in juniper.build_fold_prompt([])
    assert rule in origin.build_fold_prompt([])
    assert rule in lately.build_fold_prompt([], today=date(2026, 11, 19))


def test_retry_note_only_for_over_cap_and_names_the_count():
    note = retry_note(["uncited: x", "over_cap: 354 words > 350"])
    assert note.startswith("Your last page was 354 words, over the 350-word cap")
    assert f"at most {TARGET_SENTENCES} sentences, about {TARGET_WORDS} words" in note
    # The budget scales from the last page: 17 sentences x 280 / 389 words -> 12; 19 x 280 / 435 -> 12;
    # 20 x 280 / 400 -> 14, capped at 12; 8 x 280 / 400 -> 5.
    assert "389 words in 17 sentences" in retry_note(["over_cap: 389 words > 350"], 17)
    assert "at most 5 sentences" in retry_note(["over_cap: 400 words > 350"], 8)
    assert "at most 12 sentences" in retry_note(["over_cap: 400 words > 350"], 20)
    assert retry_note(["uncited: x", "too_thin: 1 cited sentences < 3"], 9) is None
    assert retry_note([]) is None
    assert with_retry_note("P", None) == "P"
    assert with_retry_note("P", note) == f"P\n\n{note}"
