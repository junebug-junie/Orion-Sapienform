"""Core pages: one length rule on every page, and an over-cap retry that names the word count."""
from __future__ import annotations

from datetime import date

from orion.memory.pages import juniper, lately, origin
from orion.memory.pages.fold import TARGET_WORDS, WORD_CAP, length_rule, retry_note, with_retry_note


def test_every_page_prompt_uses_the_shared_length_rule():
    assert TARGET_WORDS < WORD_CAP
    rule = length_rule()
    assert f"About {TARGET_WORDS} words" in rule and f"over {WORD_CAP} words is thrown away" in rule
    assert rule in juniper.build_fold_prompt([])
    assert rule in origin.build_fold_prompt([])
    assert rule in lately.build_fold_prompt([], today=date(2026, 11, 19))


def test_retry_note_only_for_over_cap_and_names_the_count():
    note = retry_note(["uncited: x", "over_cap: 354 words > 350"])
    assert note.startswith("Your last page was 354 words, over the 350-word cap")
    assert f"about {TARGET_WORDS} words" in note
    assert retry_note(["uncited: x", "too_thin: 1 cited sentences < 3"]) is None
    assert retry_note([]) is None
    assert with_retry_note("P", None) == "P"
    assert with_retry_note("P", note) == f"P\n\n{note}"
