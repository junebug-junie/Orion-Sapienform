"""Unit tests for System One frame access and curiosity admission."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.system_one_appraisal import (
    SystemOneAnswerV1,
    SystemOneAppraisalFrameV1,
    SystemOneInputStateV1,
    SystemOneQuestionV1,
)
from orion.substrate.system_one_access import (
    access_score_question,
    argmax_score_level,
    decide_curiosity_admission,
)
from orion.substrate.system_one_appraisal import QUESTION_SET_ID, SYSTEM_ONE_QUESTIONS


NOW = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)


def _frame(*, probabilities: dict[str, float], generated_at: datetime = NOW) -> SystemOneAppraisalFrameV1:
    answers = {}
    for question_id, question in SYSTEM_ONE_QUESTIONS.items():
        if question_id == "curiosity_pull":
            probs = probabilities
        else:
            probs = {"0": 0.7, "1": 0.2, "2": 0.1}
        score = sum(int(k) * float(v) for k, v in probs.items())
        answers[question_id] = SystemOneAnswerV1(
            question_id=question_id,
            type="score",
            score=score,
            confidence=0.6,
            probabilities=probs,
            legend={str(i): str(c) for i, c in enumerate(question.criteria or [])},
        )
    state = SystemOneInputStateV1(
        source_broadcast_projection_id="broadcast-1",
        source_broadcast_generated_at=generated_at,
        selected_action_type="reflect",
        coalition_stability_score=0.5,
    )
    return SystemOneAppraisalFrameV1(
        frame_id="frame-test",
        question_set_id=QUESTION_SET_ID,
        generated_at=generated_at,
        expires_at=generated_at + timedelta(seconds=90),
        provider="kev",
        model_id="kev-latest",
        source_refs=["attention.broadcast:broadcast-1"],
        input_state=state,
        questions=dict(SYSTEM_ONE_QUESTIONS),
        answers=answers,
    )


def test_argmax_score_level():
    assert argmax_score_level({"0": 0.5, "1": 0.3, "2": 0.2}) == "0"
    assert argmax_score_level({"0": 0.1, "1": 0.6, "2": 0.3}) == "1"
    assert argmax_score_level({"0": 0.1, "1": 0.2, "2": 0.7}) == "2"


def test_access_rejects_expired_frame():
    frame = _frame(probabilities={"0": 0.8, "1": 0.1, "2": 0.1})
    result = access_score_question(
        frame,
        question_id="curiosity_pull",
        now=NOW + timedelta(seconds=120),
    )
    assert result == (None, "expired")


def test_access_rejects_question_set_mismatch():
    frame = _frame(probabilities={"0": 0.8, "1": 0.1, "2": 0.1})
    mismatched = frame.model_copy(update={"question_set_id": "other.set"})
    result = access_score_question(
        mismatched,
        question_id="curiosity_pull",
        now=NOW,
    )
    assert result == (None, "question_set_mismatch")


def test_level_0_noops_evaluator():
    decision = decide_curiosity_admission(
        _frame(probabilities={"0": 0.7, "1": 0.2, "2": 0.1}),
        now=NOW,
    )
    assert decision.gate_result == "system_one_curiosity_noop"
    assert decision.admit_evaluator is False
    assert decision.selected_level == "0"
    assert decision.probabilities["0"] == 0.7


def test_level_1_and_2_both_admit_without_boost():
    d1 = decide_curiosity_admission(
        _frame(probabilities={"0": 0.2, "1": 0.55, "2": 0.25}),
        now=NOW,
    )
    d2 = decide_curiosity_admission(
        _frame(probabilities={"0": 0.1, "1": 0.2, "2": 0.7}),
        now=NOW,
    )
    assert d1.gate_result == "system_one_curiosity_admit"
    assert d2.gate_result == "system_one_curiosity_admit"
    assert d1.admit_evaluator is True
    assert d2.admit_evaluator is True
    assert d1.selected_level == "1"
    assert d2.selected_level == "2"


def test_missing_frame_fails_open():
    decision = decide_curiosity_admission(None, now=NOW)
    assert decision.gate_result == "system_one_unavailable_fallback"
    assert decision.admit_evaluator is True
    assert decision.fallback_reason == "no_frame"


def test_kill_switch_fails_open():
    decision = decide_curiosity_admission(
        _frame(probabilities={"0": 0.9, "1": 0.05, "2": 0.05}),
        kill_switch=True,
        now=NOW,
    )
    assert decision.gate_result == "system_one_unavailable_fallback"
    assert decision.admit_evaluator is True
    assert decision.fallback_reason == "consumer_kill_switch"


def test_max_age_fails_open():
    decision = decide_curiosity_admission(
        _frame(probabilities={"0": 0.2, "1": 0.6, "2": 0.2}),
        now=NOW + timedelta(seconds=200),
        max_age_sec=120.0,
    )
    assert decision.gate_result == "system_one_unavailable_fallback"
    assert decision.admit_evaluator is True
    assert decision.fallback_reason == "expired"
