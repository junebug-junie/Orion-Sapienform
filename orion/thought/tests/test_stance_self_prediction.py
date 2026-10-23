"""Self-prediction A1: the parser cleans stance's bet and never invents a value.

Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md.
"""
from __future__ import annotations

import pytest

from orion.schemas.thought import SELF_PREDICTION_COVERAGE
from orion.thought.stance_react import (
    SELF_PREDICTION_P_MAX,
    SELF_PREDICTION_P_MIN,
    normalize_self_prediction,
    parse_stance_react_payload,
)


def _payload(**overrides: object) -> dict:
    base: dict = {
        "imperative": "Stay present; one situated question.",
        "tone": "warm",
        "strain_refs": [],
        "evidence_refs": ["hub:turn"],
        "stance_harness_slice": {
            "task_mode": "reflective_dialogue",
            "conversation_frame": "reflective",
            "answer_strategy": "presence_first",
        },
    }
    base.update(overrides)
    return base


FULL = {
    "asks_question_p": 0.7,
    "fetches_outside_p": 0.1,
    "steps_lo": 3,
    "steps_hi": 9,
    "reply_chars_lo": 200,
    "reply_chars_hi": 1200,
}


def test_full_bet_survives_parse_with_fixed_coverage() -> None:
    thought = parse_stance_react_payload(_payload(self_prediction=dict(FULL)), correlation_id="c-1")
    sp = thought.self_prediction
    assert sp is not None
    assert (sp.asks_question_p, sp.fetches_outside_p) == (0.7, 0.1)
    assert (sp.steps_lo, sp.steps_hi, sp.reply_chars_lo, sp.reply_chars_hi) == (3, 9, 200, 1200)
    assert sp.coverage == SELF_PREDICTION_COVERAGE


def test_missing_bet_is_none_and_turn_still_parses() -> None:
    thought = parse_stance_react_payload(_payload(), correlation_id="c-1")
    assert thought.self_prediction is None
    assert thought.imperative.startswith("Stay present")


def test_unknown_keys_do_not_fail_the_turn() -> None:
    # TurnSelfPredictionV1 is extra="forbid"; a chatty model must not take the whole stance down.
    thought = parse_stance_react_payload(
        _payload(self_prediction={**FULL, "notes": "I think it will be quick", "coverage": 0.5}),
        correlation_id="c-1",
    )
    assert thought.self_prediction is not None
    assert thought.self_prediction.coverage == SELF_PREDICTION_COVERAGE  # model's coverage ignored


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, SELF_PREDICTION_P_MIN),
        (1.0, SELF_PREDICTION_P_MAX),
        ("0.4", 0.4),
        (1, SELF_PREDICTION_P_MAX),
    ],
)
def test_probabilities_are_clamped(raw: object, expected: float) -> None:
    out = normalize_self_prediction({"asks_question_p": raw})
    assert out == {"asks_question_p": expected}


@pytest.mark.parametrize("raw", [70, -0.1, True, False, "likely", None, float("nan"), float("inf")])
def test_unusable_probabilities_are_dropped_not_rescaled(raw: object) -> None:
    assert normalize_self_prediction({"asks_question_p": raw}) is None


@pytest.mark.parametrize(
    "pair",
    [
        {"steps_lo": 9, "steps_hi": 3},  # inverted
        {"steps_lo": 3},  # one end
        {"steps_hi": 9},
        {"steps_lo": -1, "steps_hi": 4},  # negative
        {"steps_lo": 3.5, "steps_hi": 9},  # not whole
        {"steps_lo": "a", "steps_hi": 9},
    ],
)
def test_bad_ranges_drop_both_ends(pair: dict) -> None:
    assert normalize_self_prediction(pair) is None


def test_whole_floats_and_equal_ends_are_kept() -> None:
    assert normalize_self_prediction({"steps_lo": 4.0, "steps_hi": 4}) == {"steps_lo": 4, "steps_hi": 4}


def test_one_bad_pair_keeps_the_good_ones() -> None:
    out = normalize_self_prediction({**FULL, "reply_chars_lo": 900, "reply_chars_hi": 100})
    assert out is not None
    assert "reply_chars_lo" not in out and "reply_chars_hi" not in out
    assert out["steps_lo"] == 3 and out["asks_question_p"] == 0.7


@pytest.mark.parametrize("raw", [None, "0.7", [], {}, {"notes": "x"}])
def test_nothing_usable_is_none(raw: object) -> None:
    assert normalize_self_prediction(raw) is None


def test_huge_json_int_is_dropped_not_raised() -> None:
    # Review finding: float() on a 400-digit int raises OverflowError, which used to fail the turn.
    huge = 10**400
    assert normalize_self_prediction({"steps_lo": huge, "steps_hi": huge}) is None
    assert normalize_self_prediction({"asks_question_p": huge}) is None


def test_any_bet_error_still_parses_the_turn(monkeypatch) -> None:
    from orion.thought import stance_react

    def boom(_value: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(stance_react, "normalize_self_prediction", boom)
    thought = parse_stance_react_payload(_payload(self_prediction=dict(FULL)), correlation_id="c-1")
    assert thought.self_prediction is None
    assert thought.imperative.startswith("Stay present")
