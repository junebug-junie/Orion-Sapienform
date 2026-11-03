"""Self-prediction A1: the parser cleans stance's bet and never invents a value.

Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md.
"""
from __future__ import annotations

import pytest

from orion.schemas.thought import TurnSelfPredictionV2
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
    "more_steps_than_usual_p": 0.6,
    "longer_reply_than_usual_p": 0.3,
}


def test_full_bet_survives_parse_as_v2() -> None:
    thought = parse_stance_react_payload(_payload(self_prediction=dict(FULL)), correlation_id="c-1")
    sp = thought.self_prediction
    assert isinstance(sp, TurnSelfPredictionV2)
    assert (sp.asks_question_p, sp.fetches_outside_p) == (0.7, 0.1)
    assert (sp.more_steps_than_usual_p, sp.longer_reply_than_usual_p) == (0.6, 0.3)


def test_missing_bet_is_none_and_turn_still_parses() -> None:
    thought = parse_stance_react_payload(_payload(), correlation_id="c-1")
    assert thought.self_prediction is None
    assert thought.imperative.startswith("Stay present")


def test_unknown_and_retired_keys_do_not_fail_the_turn() -> None:
    # TurnSelfPredictionV2 is extra="forbid"; a chatty model, or one still writing v1's size ranges,
    # must not take the whole stance down.
    thought = parse_stance_react_payload(
        _payload(self_prediction={**FULL, "notes": "quick", "coverage": 0.5, "steps_lo": 3, "steps_hi": 9,
                                  "schema_version": "turn.self_prediction.v1"}),
        correlation_id="c-1",
    )
    assert isinstance(thought.self_prediction, TurnSelfPredictionV2)
    assert thought.self_prediction.more_steps_than_usual_p == 0.6


def test_only_retired_keys_is_no_bet() -> None:
    assert normalize_self_prediction({"steps_lo": 3, "steps_hi": 9, "reply_chars_lo": 1, "reply_chars_hi": 2}) is None


@pytest.mark.parametrize("key", list(FULL))
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, SELF_PREDICTION_P_MIN),
        (1.0, SELF_PREDICTION_P_MAX),
        ("0.4", 0.4),
        (1, SELF_PREDICTION_P_MAX),
    ],
)
def test_probabilities_are_clamped(key: str, raw: object, expected: float) -> None:
    out = normalize_self_prediction({key: raw})
    assert out == {"schema_version": "turn.self_prediction.v2", key: expected}


@pytest.mark.parametrize("raw", [70, -0.1, True, False, "likely", None, float("nan"), float("inf"), 10**400])
def test_unusable_probabilities_are_dropped_not_rescaled(raw: object) -> None:
    assert normalize_self_prediction({"more_steps_than_usual_p": raw}) is None


@pytest.mark.parametrize("raw", [None, "0.7", [], {}, {"notes": "x"}])
def test_nothing_usable_is_none(raw: object) -> None:
    assert normalize_self_prediction(raw) is None


def test_any_bet_error_still_parses_the_turn(monkeypatch) -> None:
    from orion.thought import stance_react

    def boom(_value: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(stance_react, "normalize_self_prediction", boom)
    thought = parse_stance_react_payload(_payload(self_prediction=dict(FULL)), correlation_id="c-1")
    assert thought.self_prediction is None
    assert thought.imperative.startswith("Stay present")
