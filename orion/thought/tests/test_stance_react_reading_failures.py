"""Regression: real stance outputs from failed reading turns (live 2026-10-23..28).

The fixture holds the raw stance text from cognition_traces for five failed
``reading.turn`` runs; live, every one deferred as ``turn_deferred:empty_imperative``.
Only the ``imperial_with_text`` cases change here (regression tests). The others
must keep deferring (guards against laundering): their fix is upstream, in the
stance view of the reading prompt (orion/world_pulse_read/reader_contract.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1
from orion.thought.stance_react import (
    apply_stance_react_pipeline,
    normalize_stance_react_raw,
    parse_stance_react_payload,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stance_react_reading_failures_2026_10.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _case(family: str) -> list[dict]:
    return [c for c in CASES if c["family"] == family]


def _decide(case: dict):
    corr = case["correlation_id"]
    thought = parse_stance_react_payload(case["stance_text"], correlation_id=corr, session_id=case["session_id"])
    request = StanceReactRequestV1(
        correlation_id=corr,
        session_id=case["session_id"],
        user_message="reading task",
        association=HubAssociationBundleV1(
            correlation_id=corr, broadcast=None, broadcast_stale=False, read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "reading task"},
    )
    return apply_stance_react_pipeline(thought, request)


def test_fixture_holds_every_family():
    assert {c["family"] for c in CASES} == {
        "imperial_with_text", "imperial_null", "reader_summary_key", "reader_answer",
    }
    assert all(c["live_error"] == "turn_deferred:empty_imperative" for c in CASES)


@pytest.mark.parametrize("case", _case("imperial_with_text"), ids=lambda c: c["correlation_id"][:8])
def test_misspelled_imperative_key_now_carries_the_stance(case):
    """Bonsai-2 wrote "imperial" for "imperative"; its text is the real stance."""
    written = json.loads(case["stance_text"])["imperial"]
    decided = _decide(case)
    assert decided.imperative == written.strip()[:400]
    assert "empty_imperative" not in decided.disposition_reasons
    assert decided.disposition == "proceed"


@pytest.mark.parametrize("case", _case("imperial_null"), ids=lambda c: c["correlation_id"][:8])
def test_misspelled_key_with_null_value_still_defers(case):
    """No text, no stance: the empty-imperative gate still holds."""
    decided = _decide(case)
    assert decided.imperative == ""
    assert decided.disposition == "defer"
    assert "empty_imperative" in decided.disposition_reasons


@pytest.mark.parametrize(
    "case", _case("reader_summary_key") + _case("reader_answer"), ids=lambda c: c["correlation_id"][:8],
)
def test_reader_shaped_output_is_never_laundered_into_an_imperative(case):
    """Qwen3.6-35B wrote the reader's keys. That is not a stance; it must still defer.
    The fix for this family is upstream (the stance no longer sees the reader contract)."""
    decided = _decide(case)
    assert decided.imperative == ""
    assert decided.disposition == "defer"
    assert "empty_imperative" in decided.disposition_reasons


def test_correct_key_wins_over_misspelled_key():
    data = normalize_stance_react_raw({"imperative": "Read the page.", "imperial": "Something else."})
    assert data["imperative"] == "Read the page."
    assert "imperial" not in data


@pytest.mark.parametrize("value", [None, "", "   ", {"text": "x"}, ["x"], 3])
def test_misspelled_key_without_text_is_dropped(value):
    data = normalize_stance_react_raw({"imperial": value})
    assert data["imperative"] == ""
    assert "imperial" not in data


def test_misspelled_key_text_is_clamped_like_the_real_key():
    data = normalize_stance_react_raw({"imperial": "x" * 900})
    assert len(data["imperative"]) == 400
