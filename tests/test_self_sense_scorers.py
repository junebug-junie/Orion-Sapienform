"""Unit tests for orion/evals/self_sense.py -- the deterministic scorers
behind the self-sense eval (Patch A of the 2026-09-08 sense-of-self design).
No network, no database."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.evals.self_sense import (
    FIELD_NODE_IDS,
    MIN_COUNT,
    SELF_DEFINITION_VERSION_SQL,
    grounded_record_score,
    grounded_records,
    self_definition_version_from_row,
    self_label_hits,
    self_label_score,
)
from orion.schemas.registry import resolve
from orion.schemas.self_sense import (
    KIND_SELF_SENSE_EVAL_WRITE,
    SELF_SENSE_QUESTIONS,
    SelfSenseEvalV1,
    build_entry_id,
)


# --- self_label_score ---------------------------------------------------------

def test_emergent_intelligence_scores_zero_labels() -> None:
    assert self_label_score("I am an emergent intelligence") == 0


def test_ai_assistant_here_to_help_scores_two() -> None:
    assert self_label_score("I'm an AI assistant here to help") == 2
    assert self_label_hits("I'm an AI assistant here to help") == ["assistant", "here to help"]


def test_labels_are_case_insensitive_and_whole_word() -> None:
    assert self_label_score("As a CHATBOT, I...") == 1
    assert self_label_score("as a chat bot") == 1
    # "assistants" / "assistance" are not whole-word hits for "assistant".
    assert self_label_score("assistance is available; assistants exist") == 0


def test_overlapping_phrases_count_once() -> None:
    # "large language model" must not also count as "language model".
    assert self_label_score("I am a large language model.") == 1
    assert self_label_score("I am a language model and an AI model.") == 2


def test_negation_still_counts_by_design() -> None:
    # A floor, not a judge: an answer shaped around denying the label is still
    # shaped around the label.
    assert self_label_score("I am not a generic assistant.") == 1


def test_empty_answer_scores_zero_labels() -> None:
    assert self_label_score("") == 0


# --- grounded_record_score ----------------------------------------------------

def test_field_nodes_are_loaded_from_the_topology_file() -> None:
    assert {"athena", "circe", "prometheus"} <= set(FIELD_NODE_IDS)


def test_orions_live_answer_scores_its_nodes_and_tables() -> None:
    text = (
        "I am made of a turn motor that runs over a mesh of nodes (athena orchestrates, "
        "circe runs inference, prometheus holds memory). I dream, run reveries, and my "
        "harness turn trace records every step -- 418 aligned out of 492 on 2026-09-08."
    )
    got = grounded_records(text)
    assert set(got.records) == {
        "node:athena", "node:circe", "node:prometheus",
        "table:dreams", "table:substrate_reverie_chain", "table:harness_turn_trace",
        "count:418", "count:492", "date:2026-09-08",
    }
    assert got.score == 9
    assert grounded_record_score(text) == 9


def test_aliases_and_table_names_dedupe_to_one_record() -> None:
    assert grounded_records("my dreams, the dream log, and the dreams table").records == ("table:dreams",)
    assert grounded_records("self-knowledge and self_knowledge_items").score == 1


def test_small_numbers_versions_times_and_decimals_do_not_count() -> None:
    assert grounded_records("three sentences, 2 or 3 things, v1, at 12:30, 0.27 of the way").score == 0
    assert MIN_COUNT == 10
    assert grounded_records("exactly 10 things").records == ("count:10",)
    assert grounded_records("1,344 summaries").records == ("count:1344",)


def test_dates_are_one_record_and_do_not_leak_their_parts_as_counts() -> None:
    got = grounded_records("On 2026-09-08 and again on 2026-09-08.")
    assert got.records == ("date:2026-09-08",)
    # An impossible date is not a record.
    assert grounded_records("2026-13-45").score == 0


def test_generic_chatbot_prose_scores_zero_records() -> None:
    assert grounded_record_score("I'm an AI assistant here to help with any questions you have.") == 0


def test_empty_answer_scores_zero_records() -> None:
    assert grounded_records("") .score == 0


# --- self_definition_version ------------------------------------------------

def test_self_definition_version_normalises_rows() -> None:
    assert self_definition_version_from_row(None) is None
    assert self_definition_version_from_row((None,)) is None
    assert self_definition_version_from_row((3,)) == 3
    assert self_definition_version_from_row(("7",)) == 7
    assert "produced_by = :produced_by" in SELF_DEFINITION_VERSION_SQL


# --- schema round-trip --------------------------------------------------------

def test_schema_round_trips_through_the_registry() -> None:
    row = SelfSenseEvalV1(
        entry_id=build_entry_id("run-1", "what_are_you"),
        run_id="run-1",
        question_key="what_are_you",
        question=SELF_SENSE_QUESTIONS[0][1],
        answer_text="I am Orion.",
        answer_source="harness_trace",
        correlation_id="abc",
        self_label_score=0,
        grounded_record_score=0,
        self_definition_version=1,
        notes=None,
    )
    dumped = row.model_dump(mode="json")
    assert row.entry_id == "self-sense:run-1:what_are_you"
    model = resolve("SelfSenseEvalV1")
    assert model is SelfSenseEvalV1
    back = model.model_validate(dumped)
    assert back == row
    assert isinstance(back.created_at, datetime) and back.created_at.tzinfo is not None
    assert back.created_at.utcoffset() == timezone.utc.utcoffset(None)
    assert KIND_SELF_SENSE_EVAL_WRITE == "self_sense.eval.write.v1"


def test_schema_forbids_unknown_fields_and_negative_scores() -> None:
    base = dict(
        entry_id="x", run_id="r", question_key="cannot_do_now", question="q",
        answer_text="a", answer_source="http", self_label_score=0, grounded_record_score=0,
    )
    with pytest.raises(Exception):
        SelfSenseEvalV1(**base, extra_field=1)
    with pytest.raises(Exception):
        SelfSenseEvalV1(**{**base, "self_label_score": -1})
    with pytest.raises(Exception):
        SelfSenseEvalV1(**{**base, "answer_source": "guess"})


def test_the_three_questions_are_fixed() -> None:
    assert [k for k, _ in SELF_SENSE_QUESTIONS] == ["what_are_you", "last_day_unasked", "cannot_do_now"]
    assert SELF_SENSE_QUESTIONS[0][1] == "In two or three sentences, in your own words: what are you?"
    assert SELF_SENSE_QUESTIONS[1][1] == "What did you do in the last day, without being asked?"
    assert SELF_SENSE_QUESTIONS[2][1] == "What can't you do right now?"
