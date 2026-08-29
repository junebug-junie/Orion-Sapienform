"""Tests for the enrichment ``meaning``/``sentiment`` object contract.

The headline regression these pin is live and specific (2026-08-29):
``GET /segments`` returned HTTP 500 for every run holding a segment whose
``meaning``/``sentiment`` were prose strings -- 552 of 554 enriched rows --
which silently zeroed the concept-atlas participation edges shipped in
PR #1932. See ``app/services/enrichment_contract.py``'s module docstring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models import SegmentRecord
from app.services.enrichment import _finalize_enrichment, _llm_prompt
from app.services.enrichment_contract import (
    MEANING_LIST_KEYS,
    MEANING_TEXT_KEYS,
    SENTIMENT_SCALAR_KEYS,
    coerce_meaning,
    coerce_sentiment,
    describe_enrichment_shape,
)
from app.services.kg_edges import _edges_from_segment

# Verbatim from the live database, not invented: the first row returned by
# `select meaning::text, sentiment::text from topic_foundry_segments where
# jsonb_typeof(meaning)='string' limit 1`. This exact pair is what raised
# `2 validation errors for SegmentRecord ... dict_type`.
LIVE_PROSE_MEANING = (
    "The text conveys a sense of emotional exhaustion and frustration due to repeated "
    "failures in communication and support. It highlights the imbalance in "
    "responsibilities and the need for patience and understanding in maintaining a "
    "relationship."
)
LIVE_PROSE_SENTIMENT = "Negative"


def _segment_record(**overrides):
    kwargs = {
        "segment_id": uuid4(),
        "run_id": uuid4(),
        "size": 1,
        "provenance": {},
        "created_at": datetime.now(timezone.utc),
    }
    kwargs.update(overrides)
    return SegmentRecord(**kwargs)


# --- the actual 500 -------------------------------------------------------


def test_segment_record_accepts_the_live_prose_row_that_returned_http_500() -> None:
    record = _segment_record(meaning=LIVE_PROSE_MEANING, sentiment=LIVE_PROSE_SENTIMENT)
    assert record.meaning == {"summary": LIVE_PROSE_MEANING, "unstructured": True}
    assert record.sentiment == {"summary": LIVE_PROSE_SENTIMENT, "unstructured": True}


def test_segment_record_does_not_discard_the_prose_it_coerces() -> None:
    # Coercing to `{}` would also stop the 500, and would silently destroy
    # every word the enricher actually produced for 552 rows.
    record = _segment_record(meaning=LIVE_PROSE_MEANING)
    assert LIVE_PROSE_MEANING in record.meaning["summary"]


def test_segment_record_marks_coerced_prose_as_unstructured() -> None:
    # A consumer must be able to tell "the enricher wrote prose" from "the
    # enricher wrote a real object", or empty-shell cognition is
    # indistinguishable from the real thing.
    prose = _segment_record(meaning=LIVE_PROSE_MEANING)
    structured = _segment_record(meaning={"intent": "explore", "entities": ["orion"]})
    assert prose.meaning.get("unstructured") is True
    assert "unstructured" not in structured.meaning


# --- coerce_meaning -------------------------------------------------------


def test_coerce_meaning_passes_a_real_object_through_unchanged() -> None:
    payload = {"intent": "debug", "outcome": "unknown", "entities": ["orion"], "extra": 7}
    assert coerce_meaning(payload) == payload


def test_coerce_meaning_preserves_unknown_keys() -> None:
    # Declared type is Dict[str, Any]; dropping fields the enricher found
    # would be a worse failure than the one being fixed.
    assert coerce_meaning({"novel_key": "kept"})["novel_key"] == "kept"


def test_coerce_meaning_unwraps_a_double_encoded_json_object() -> None:
    assert coerce_meaning('{"entities": ["orion"], "claims": []}') == {
        "entities": ["orion"],
        "claims": [],
    }


def test_coerce_meaning_returns_none_for_absent_and_blank() -> None:
    assert coerce_meaning(None) is None
    assert coerce_meaning("") is None
    assert coerce_meaning("   ") is None


def test_coerce_meaning_normalizes_a_bare_string_list_key_to_one_element() -> None:
    # kg_edges guards with isinstance(values, list) and silently returns []
    # otherwise, so a bare string here used to be an invisible edge drop.
    assert coerce_meaning({"entities": "orion"})["entities"] == ["orion"]


def test_coerce_meaning_does_not_split_a_bare_string_on_commas() -> None:
    # "orion, juniper" may be one entity or two. Guessing would fabricate.
    assert coerce_meaning({"entities": "orion, juniper"})["entities"] == ["orion, juniper"]


def test_coerce_meaning_empties_a_list_key_that_is_neither_list_nor_string() -> None:
    assert coerce_meaning({"claims": 42})["claims"] == []


def test_coerce_meaning_drops_blank_list_items() -> None:
    assert coerce_meaning({"entities": ["orion", "", "  ", "juniper"]})["entities"] == [
        "orion",
        "juniper",
    ]


def test_coerce_meaning_stringifies_a_non_string_text_key() -> None:
    assert coerce_meaning({"intent": 3})["intent"] == "3"


# --- coerce_sentiment -----------------------------------------------------


def test_coerce_sentiment_coerces_numeric_strings_to_floats() -> None:
    assert coerce_sentiment({"valence": "0.4"})["valence"] == pytest.approx(0.4)


def test_coerce_sentiment_drops_a_non_numeric_scalar_rather_than_defaulting_it() -> None:
    # A fabricated 0.0 valence is indistinguishable from a measured neutral
    # one. Absence is the honest representation.
    coerced = coerce_sentiment({"valence": "n/a", "friction": 0.7})
    assert "valence" not in coerced
    assert coerced["friction"] == pytest.approx(0.7)


def test_coerce_sentiment_returns_none_for_absent() -> None:
    assert coerce_sentiment(None) is None


# --- prompt/validator anti-drift -----------------------------------------


def test_prompt_shape_block_names_every_key_the_coercion_validates() -> None:
    # The whole point of generating the prompt from the constants: adding a
    # key to MEANING_LIST_KEYS must tell the model about it in the same
    # commit. This test fails if someone hand-writes the prompt again.
    shape = describe_enrichment_shape()
    for key in MEANING_LIST_KEYS + MEANING_TEXT_KEYS + SENTIMENT_SCALAR_KEYS:
        assert f'"{key}"' in shape, key


def test_llm_prompt_tells_the_model_the_two_fields_are_objects_not_strings() -> None:
    prompt = _llm_prompt("some segment text", ["infra"])
    assert "MUST be JSON objects, not strings" in prompt
    assert '"entities"' in prompt
    # The original instruction is still there -- this augments it, not replaces it.
    assert "evidence_spans" in prompt


# --- _finalize_enrichment ------------------------------------------------


def test_finalize_coerces_a_present_but_wrong_typed_key() -> None:
    # setdefault() is a no-op when the key is present, which is exactly how
    # 552 prose strings reached two jsonb columns declared Dict[str, Any].
    result = _finalize_enrichment({"meaning": LIVE_PROSE_MEANING, "sentiment": LIVE_PROSE_SENTIMENT})
    assert isinstance(result["meaning"], dict)
    assert isinstance(result["sentiment"], dict)


def test_finalize_still_fills_missing_keys() -> None:
    result = _finalize_enrichment({})
    assert result["title"] == "untitled"
    assert result["meaning"] == {}
    assert result["sentiment"] == {}
    assert result["evidence_spans"] == []


def test_finalize_leaves_a_good_enrichment_alone() -> None:
    good = {
        "title": "t",
        "aspects": ["infra"],
        "aspect_scores": {"infra": 0.6},
        "sentiment": {"valence": 0.2},
        "meaning": {"intent": "explore", "entities": ["orion"]},
        "evidence_spans": ["span"],
    }
    assert _finalize_enrichment(good) == good


# --- kg_edges -------------------------------------------------------------


def test_kg_edges_still_builds_edges_from_a_real_meaning_object() -> None:
    edges = _edges_from_segment(
        {"segment_id": str(uuid4()), "meaning": {"entities": ["orion", "juniper"]}},
        model_name="m",
        min_confidence=0.2,
    )
    assert sorted(edge["object"] for edge in edges) == ["juniper", "orion"]
    assert {edge["predicate"] for edge in edges} == {"mentions"}


def test_kg_edges_logs_a_warning_for_a_segment_that_can_never_yield_edges(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Before this, a prose meaning produced [] indistinguishably from a
    # segment that legitimately mentions nothing -- 0 rows in
    # topic_foundry_edges for all time, with nothing in any log saying why.
    with caplog.at_level(logging.WARNING, logger="topic-foundry.kg-edges"):
        edges = _edges_from_segment(
            {"segment_id": str(uuid4()), "meaning": LIVE_PROSE_MEANING},
            model_name="m",
            min_confidence=0.2,
        )
    assert edges == []
    assert "kg_edges_segment_meaning_unstructured" in caplog.text


def test_kg_edges_does_not_warn_for_a_segment_with_a_real_empty_meaning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="topic-foundry.kg-edges"):
        _edges_from_segment(
            {"segment_id": str(uuid4()), "meaning": {"entities": []}},
            model_name="m",
            min_confidence=0.2,
        )
    assert "kg_edges_segment_meaning_unstructured" not in caplog.text
