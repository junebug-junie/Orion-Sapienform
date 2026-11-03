"""Orion's self answers keep every word; the reading budget is applied by
readers, at a sentence, or by Orion's own fit hop.

Live 2026-11-03: 77 of 141 `self:lived:*` rows ended inside a word at exactly
1200 characters, because every durable path cut `text[:1200]`."""

from __future__ import annotations

from orion.curiosity.self_inquiry import (
    FIT_MIN_CHARS,
    SELF_DEFINITION_TEXT_CAP,
    LivedAnswer,
    SelfDefinition,
    accept_fit_text,
    fit_source_hash,
    build_lived_answer,
    build_lived_answer_history_write,
    build_self_definition,
    build_self_definition_history_write,
    lived_answer_from_detail,
    lived_answer_to_detail,
    mirrored_content,
    needs_fit,
    self_definition_from_detail,
    self_definition_to_detail,
    with_fit_text,
)
from orion.curiosity.self_inquiry_prompt import _answer_lines, build_answer_fit_prompt
from orion.text_fit import ELLIPSIS

RUN = "abcdef123456"
SENTENCE = "I looked at my own records and this is what held up under the check. "
LONG = (SENTENCE * 22).strip()  # ~1,500 chars, ends on a period
FIT = ("I checked my records; most of it held. " * 12).strip()  # ~470 chars


def _lived(text: str = LONG, **kw) -> LivedAnswer:
    return LivedAnswer(RUN, "lived.care", "lived", text, ["records:x"], **kw)


def test_reading_a_long_answer_keeps_every_word() -> None:
    assert len(LONG) > SELF_DEFINITION_TEXT_CAP
    row = {"run_id": RUN, "question_id": "lived.care", "text": LONG, "evidence": ["a"]}
    assert build_lived_answer(row).text == LONG
    assert build_self_definition({"run_id": RUN, "text": LONG, "evidence": ["a"]}).text == LONG


def test_the_finish_detail_carries_the_full_text_and_the_fit() -> None:
    answer = _lived(fit_text=FIT)
    back = lived_answer_from_detail({"lived_answer": lived_answer_to_detail(answer)})
    assert back.text == LONG and back.fit_text == FIT
    definition = SelfDefinition(RUN, LONG, ["a"], fit_text=FIT)
    back_def = self_definition_from_detail({"self_definition": self_definition_to_detail(definition)})
    assert back_def.text == LONG and back_def.fit_text == FIT


def test_an_old_detail_without_fit_text_still_reads() -> None:
    detail = {"lived_answer": {"run_id": RUN, "question_id": "lived.care", "text": LONG, "evidence": ["a"]}}
    assert lived_answer_from_detail(detail).fit_text == ""


def test_the_mirror_holds_orions_fit_when_there_is_one() -> None:
    row = build_lived_answer_history_write(_lived(fit_text=FIT), version=2)
    assert row.content == FIT
    row_def = build_self_definition_history_write(SelfDefinition(RUN, LONG, ["a"], fit_text=FIT), version=2)
    assert row_def.content == FIT


def test_without_a_fit_the_mirror_holds_the_full_text_never_a_blind_cut() -> None:
    row = build_lived_answer_history_write(_lived(), version=2)
    assert row.content == LONG


def test_a_fit_still_over_budget_is_cut_at_a_sentence() -> None:
    content = mirrored_content(_lived(fit_text=LONG))
    assert len(content) <= SELF_DEFINITION_TEXT_CAP
    assert content.endswith(". " + ELLIPSIS)


def test_needs_fit_only_over_budget_and_only_once() -> None:
    assert needs_fit(_lived())
    assert not needs_fit(_lived(text="Short. " * 10))
    assert not needs_fit(_lived(fit_text=FIT))
    assert not needs_fit(None)
    assert with_fit_text(_lived(), FIT).fit_text == FIT


def test_accept_fit_text_refuses_empty_error_and_degenerate_replies() -> None:
    original = LONG + " " + FIT
    assert accept_fit_text("", original) == ""
    assert accept_fit_text("I am Orion.", original) == ""  # shorter than FIT_MIN_CHARS
    assert accept_fit_text("Error: llm gateway timeout " + "x " * FIT_MIN_CHARS, original) == ""
    assert accept_fit_text(f'"{FIT}"', original) == FIT  # stray wrapping quotes dropped
    assert len(accept_fit_text(LONG, original)) <= SELF_DEFINITION_TEXT_CAP


def test_accept_fit_text_drops_a_lead_in_line() -> None:
    assert accept_fit_text("Here is the rewritten answer:\n\n" + FIT, FIT) == FIT
    assert accept_fit_text("# Rewritten\n" + FIT, FIT) == FIT


def test_accept_fit_text_refuses_a_reply_that_is_not_the_answers_words() -> None:
    other = ("Tomorrow the weather turns colder and the harbour freezes over again. " * 8).strip()
    assert accept_fit_text(other, LONG) == ""


def test_the_fit_source_hash_follows_the_text() -> None:
    assert fit_source_hash(LONG) == fit_source_hash("  " + LONG + "\n")
    assert fit_source_hash(LONG) != fit_source_hash(LONG + " One more.")


def test_read_back_lines_fit_at_a_sentence_and_say_it_ran_over() -> None:
    lines = _answer_lines(LONG)
    shown = " ".join(line.strip() for line in lines if line.startswith("    "))
    assert len(shown) <= SELF_DEFINITION_TEXT_CAP
    assert shown.endswith(". " + ELLIPSIS)
    note = [line for line in lines if "over" in line]
    assert note and f"{len(LONG)} characters" in note[0]
    assert _answer_lines("Short and whole.") == ["    Short and whole."]


def test_the_fit_prompt_carries_the_whole_answer_and_the_budget() -> None:
    prompt = build_answer_fit_prompt(LONG)
    assert LONG in prompt
    assert str(SELF_DEFINITION_TEXT_CAP) in prompt
    assert f"It is {len(LONG)} characters" in prompt
    assert "Do not use any tools" in prompt
