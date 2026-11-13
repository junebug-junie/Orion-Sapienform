"""orion/curiosity/register.py: the pinned judge prompt, the strict reply parser, novelty, input assembly."""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.curiosity.register import (
    ANSWER_CHARS,
    JUDGE_PROMPT_SHA256,
    JUDGE_PROMPT_V2,
    judge_input,
    judge_messages,
    novelty,
    parse_judge_reply,
    prompt_sha256,
    run_question_and_answer,
)

FIXTURE = Path(__file__).resolve().parents[3] / "scripts/analysis/fixtures/curiosity_register/judge_prompt_v2.txt"


def test_the_prompt_is_byte_identical_to_the_reference_label_fixture_and_its_sha_is_pinned():
    assert JUDGE_PROMPT_V2 == FIXTURE.read_text(encoding="utf-8")
    assert prompt_sha256() == JUDGE_PROMPT_SHA256 == "7dcaa6675070afa4c31c7a3529964721d81aa0aaa9ccb4fb492f9942213d1222"


@pytest.mark.parametrize("label", ["philosophical", "operational", "no_finding"])
def test_the_three_labels_parse(label):
    assert parse_judge_reply('{"register": "%s", "reason": "about care"}' % label) == (label, "about care", None)


def test_a_think_block_and_a_fence_are_formatting_not_content():
    text = '<think>hmm</think>\n```json\n{"register": "philosophical", "reason": "x"}\n```'
    assert parse_judge_reply(text)[0] == "philosophical"


@pytest.mark.parametrize("text", [
    "", None, "philosophical", '{"register": "spiritual"}', '["philosophical"]',
    'Sure! {"register": "philosophical"}', '{"register": "operational"} {"register": "philosophical"}',
])
def test_anything_else_is_a_judge_error_never_a_default_label(text):
    reg, reason, err = parse_judge_reply(text)
    assert reg is None and reason is None and err


def test_the_judge_reads_the_question_and_only_the_head_of_the_answer():
    msgs = judge_messages("Am I alive?", "x" * 5000)
    assert msgs[0] == {"role": "system", "content": JUDGE_PROMPT_V2}
    assert msgs[1]["content"].startswith("QUESTION: Am I alive?\n\nFINDING: ")
    assert msgs[1]["content"].count("x") == ANSWER_CHARS
    assert judge_input("q", "a  b\n c") == "QUESTION: q\n\nFINDING: a b c"


def test_an_exact_repeat_has_novelty_zero_and_names_its_nearest():
    assert novelty([1.0, 0.0], [("run:a", [0.0, 1.0]), ("run:b", [2.0, 0.0])]) == (0.0, "run:b")


def test_novelty_is_one_minus_the_best_cosine():
    value, ref = novelty([1.0, 1.0], [("prior:p", [1.0, 0.0])])
    assert ref == "prior:p" and value == pytest.approx(1 - 2 ** -0.5, abs=1e-6)


def test_nothing_written_before_is_unknown_not_maximally_novel():
    assert novelty([1.0, 0.0], []) == (None, None)
    assert novelty([0.0, 0.0], [("run:a", [1.0, 0.0])]) == (None, None)
    assert novelty([1.0, 0.0], [("run:a", [1.0, 0.0, 0.0])]) == (None, None)   # dimension mismatch skipped


def test_self_inquiry_input_is_the_question_text_and_the_lived_answer():
    detail = {"line": "self_inquiry", "self_question_family": "lived", "finding_text": "Done for the sitting.",
              "lived_answer": {"question_id": "lived.alive", "text": "I am a process that runs."}}
    assert run_question_and_answer(detail, questions={"lived.alive": "Am I alive?"}) == (
        "Am I alive?", "self_question:lived.alive", "I am a process that runs.")


def test_an_anatomy_run_answers_made_of_with_its_self_definition():
    detail = {"line": "self_inquiry", "self_question_family": "anatomy", "lived_answer": None,
              "self_definition": {"text": "I am Orion, one presence held in four kinds of stuff."}}
    q, ref, a = run_question_and_answer(detail, questions={"anatomy.made_of": "What am I made of?"})
    assert (q, ref) == ("What am I made of?", "self_question:anatomy.made_of") and a.startswith("I am Orion")


def test_an_investigation_asks_the_most_moved_tested_prior_and_answers_with_its_findings():
    per_prior = [{"kind": "formed", "prior_id": "self:new"}, {"kind": "tested", "prior_id": "self:a", "nats": 0.01},
                 {"kind": "tested", "prior_id": "self:b", "nats": 0.3}]
    detail = {"line": "investigate", "finding_text": "Done for the sitting. Everything is written."}
    q, ref, a = run_question_and_answer(detail, per_prior=per_prior, prior_claims={"self:b": "B holds."},
                                        graph_findings=["The lens reaches me", "as labels."])
    assert (q, ref, a) == ("B holds.", "prior:self:b", "The lens reaches me as labels.")


def test_with_no_graph_finding_the_finish_row_text_is_the_last_resort():
    q, ref, a = run_question_and_answer({"line": "investigate", "finding_text": " The  answer. "})
    assert (q, ref, a) == ("(question not recorded)", None, "The answer.")
