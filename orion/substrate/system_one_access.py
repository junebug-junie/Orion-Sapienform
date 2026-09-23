"""Reusable System One frame access + curiosity admission gate.

This is an inference-access path over ``SystemOneAppraisalFrameV1``, not a
new state ontology. Consumers receive a typed answer plus provenance, or a
fallback reason that restores pre–System-One behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from orion.schemas.system_one_appraisal import (
    SystemOneAnswerV1,
    SystemOneAppraisalFrameV1,
)
from orion.substrate.system_one_appraisal import QUESTION_SET_ID

CuriosityGateResult = Literal[
    "system_one_curiosity_noop",
    "system_one_curiosity_admit",
    "system_one_unavailable_fallback",
]

FallbackReason = Literal[
    "no_frame",
    "expired",
    "question_set_mismatch",
    "missing_question",
    "answer_type_mismatch",
    "incomplete_probabilities",
    "invalid_probabilities",
    "consumer_kill_switch",
    "load_failed",
]


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def argmax_score_level(probabilities: Mapping[str, float]) -> str | None:
    """Return the declared level key with the highest probability."""
    if not probabilities:
        return None
    try:
        return max(
            ((str(k), float(v)) for k, v in probabilities.items()),
            key=lambda item: item[1],
        )[0]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SystemOneQuestionView:
    """Typed answer + provenance for one question on one frame."""

    frame_id: str
    question_set_id: str
    question_id: str
    provider: str
    model_id: str
    generated_at: datetime
    expires_at: datetime
    appraisal_age_sec: float
    source_refs: tuple[str, ...]
    answer: SystemOneAnswerV1
    selected_level: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class CuriosityAdmissionDecision:
    """Admission judgment only — never invents curiosity candidates."""

    gate_result: CuriosityGateResult
    admit_evaluator: bool
    selected_level: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    frame_id: str | None = None
    question_set_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    appraisal_age_sec: float | None = None
    source_refs: tuple[str, ...] = ()
    fallback_reason: FallbackReason | None = None
    consumer: str = "endogenous_curiosity"

    def to_telemetry(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "gate_result": self.gate_result,
            "admit_evaluator": self.admit_evaluator,
            "selected_level": self.selected_level,
            "probabilities": dict(self.probabilities),
            "frame_id": self.frame_id,
            "question_set_id": self.question_set_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "appraisal_age_sec": self.appraisal_age_sec,
            "source_refs": list(self.source_refs),
            "fallback_reason": self.fallback_reason,
        }


def access_score_question(
    frame: SystemOneAppraisalFrameV1 | None,
    *,
    question_id: str,
    expected_question_set_id: str = QUESTION_SET_ID,
    now: datetime | None = None,
    max_age_sec: float | None = None,
) -> SystemOneQuestionView | tuple[None, FallbackReason]:
    """Validate a frame and return one score answer, or a fallback reason."""
    if frame is None:
        return None, "no_frame"

    clock = _ensure_utc(now or datetime.now(timezone.utc))
    generated_at = _ensure_utc(frame.generated_at)
    expires_at = _ensure_utc(frame.expires_at)
    age_sec = (clock - generated_at).total_seconds()

    if expires_at <= clock:
        return None, "expired"
    if max_age_sec is not None and age_sec > float(max_age_sec):
        return None, "expired"
    if frame.question_set_id != expected_question_set_id:
        return None, "question_set_mismatch"
    if question_id not in frame.answers or question_id not in frame.questions:
        return None, "missing_question"

    question = frame.questions[question_id]
    answer = frame.answers[question_id]
    if question.type != "score" or answer.type != "score":
        return None, "answer_type_mismatch"

    criteria = question.criteria
    if not isinstance(criteria, list) or not criteria:
        return None, "incomplete_probabilities"
    expected_keys = {str(i) for i in range(len(criteria))}
    probs = {str(k): float(v) for k, v in answer.probabilities.items()}
    if set(probs) != expected_keys:
        return None, "incomplete_probabilities"
    total = sum(probs.values())
    if abs(total - 1.0) > 0.05:
        return None, "invalid_probabilities"
    selected = argmax_score_level(probs)
    if selected is None or selected not in expected_keys:
        return None, "invalid_probabilities"

    return SystemOneQuestionView(
        frame_id=frame.frame_id,
        question_set_id=frame.question_set_id,
        question_id=question_id,
        provider=frame.provider,
        model_id=frame.model_id,
        generated_at=generated_at,
        expires_at=expires_at,
        appraisal_age_sec=age_sec,
        source_refs=tuple(frame.source_refs),
        answer=answer,
        selected_level=selected,
        probabilities=probs,
    )


def decide_curiosity_admission(
    frame: SystemOneAppraisalFrameV1 | None,
    *,
    kill_switch: bool = False,
    now: datetime | None = None,
    max_age_sec: float | None = None,
    expected_question_set_id: str = QUESTION_SET_ID,
) -> CuriosityAdmissionDecision:
    """Categorical curiosity admission: level 0 noop; levels 1 and 2 admit.

    Levels 1 and 2 have the same admission effect. Distinction is retained in
    telemetry only. Kill switch and invalid frames fail open to the legacy
    evaluator path.
    """
    if kill_switch:
        return CuriosityAdmissionDecision(
            gate_result="system_one_unavailable_fallback",
            admit_evaluator=True,
            fallback_reason="consumer_kill_switch",
        )

    accessed = access_score_question(
        frame,
        question_id="curiosity_pull",
        expected_question_set_id=expected_question_set_id,
        now=now,
        max_age_sec=max_age_sec,
    )
    if isinstance(accessed, tuple):
        _, reason = accessed
        return CuriosityAdmissionDecision(
            gate_result="system_one_unavailable_fallback",
            admit_evaluator=True,
            fallback_reason=reason,
        )

    level = accessed.selected_level
    if level == "0":
        gate: CuriosityGateResult = "system_one_curiosity_noop"
        admit = False
    else:
        # Levels 1 and 2: identical admission. No priority bump for level 2.
        gate = "system_one_curiosity_admit"
        admit = True

    return CuriosityAdmissionDecision(
        gate_result=gate,
        admit_evaluator=admit,
        selected_level=level,
        probabilities=dict(accessed.probabilities),
        frame_id=accessed.frame_id,
        question_set_id=accessed.question_set_id,
        provider=accessed.provider,
        model_id=accessed.model_id,
        appraisal_age_sec=accessed.appraisal_age_sec,
        source_refs=accessed.source_refs,
    )
