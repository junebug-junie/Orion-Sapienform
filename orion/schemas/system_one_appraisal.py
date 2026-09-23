from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


SystemOneQuestionTypeV1 = Literal["noul", "choice", "score"]


class SystemOneQuestionV1(BaseModel):
    """One bounded System One question. This is the provider-neutral wire shape
    shared by TypeSafe System One and Kev's compatible /v1/systemone endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    type: SystemOneQuestionTypeV1
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneOpenLoopInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loop_id: str
    target_type: str
    description: str
    why_it_matters: str = ""
    salience: float = Field(ge=0.0, le=1.0)
    combined_salience: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_refs: list[str] = Field(default_factory=list)


class SystemOneFieldTargetInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    target_kind: str
    salience_score: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    dominant_channels: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class SystemOneInputStateV1(BaseModel):
    """The exact bounded state sent to System One.

    Deliberately excludes raw chat text and raw graph payloads. It compiles two
    already-live substrate artifacts: the GWT/broadcast projection and the
    field-attention frame. This is input evidence, not a new state authority.
    """

    model_config = ConfigDict(extra="forbid")

    source_broadcast_projection_id: str
    source_broadcast_generated_at: datetime
    source_field_attention_frame_id: str | None = None
    source_field_attention_generated_at: datetime | None = None

    selected_action_type: str
    selected_open_loop_id: str | None = None
    selected_description: str | None = None
    attended_node_ids: list[str] = Field(default_factory=list)
    dwell_ticks: int = Field(default=0, ge=0)
    coalition_stability_score: float = Field(ge=0.0, le=1.0)
    effort_budget_used: float = Field(default=0.0, ge=0.0)
    voluntary_override_present: bool = False
    live_unknowns: list[str] = Field(default_factory=list)
    deferred_items: list[str] = Field(default_factory=list)
    open_loops: list[SystemOneOpenLoopInputV1] = Field(default_factory=list)

    field_overall_salience: float | None = Field(default=None, ge=0.0, le=1.0)
    field_dominant_targets: list[SystemOneFieldTargetInputV1] = Field(default_factory=list)

    @field_validator(
        "source_broadcast_generated_at",
        "source_field_attention_generated_at",
    )
    @classmethod
    def _ensure_tz(cls, value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


class SystemOneAnswerV1(BaseModel):
    """Normalized answer while preserving the provider's probability surface."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    type: SystemOneQuestionTypeV1
    choice: str | None = None
    noul: float | None = Field(default=None, ge=0.0, le=1.0)
    score: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def _bounded_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        for key, probability in value.items():
            if not 0.0 <= float(probability) <= 1.0:
                raise ValueError(f"probability for {key!r} must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def _answer_matches_type(self) -> "SystemOneAnswerV1":
        if self.type == "noul" and self.noul is None:
            raise ValueError("noul answer requires noul probability")
        if self.type == "choice":
            if self.choice is None:
                raise ValueError("choice answer requires selected choice")
            if not self.probabilities:
                raise ValueError("choice answer requires option probabilities")
        if self.type == "score":
            if self.score is None:
                raise ValueError("score answer requires score")
            if not self.probabilities:
                raise ValueError("score answer requires level probabilities")
        return self


class SystemOneUsageV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class SystemOneAppraisalFrameV1(BaseModel):
    """Compiled, shadow-only System One appraisal.

    This is a typed frame derived from existing substrate artifacts. It does not
    itself mutate field state, attention, autonomy, reverie, curiosity, memory,
    or policy. Promotion into a behavioral consumer requires a separate
    live-data/calibration gate.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["system_one.appraisal.frame.v1"] = "system_one.appraisal.frame.v1"
    frame_id: str
    question_set_id: str
    generated_at: datetime = Field(default_factory=_utc_now)
    expires_at: datetime
    provider: str
    model_id: str
    request_id: str | None = None

    source_refs: list[str] = Field(default_factory=list)
    input_state: SystemOneInputStateV1
    questions: dict[str, SystemOneQuestionV1]
    answers: dict[str, SystemOneAnswerV1]

    latency_ms: float | None = Field(default=None, ge=0.0)
    usage: SystemOneUsageV1 = Field(default_factory=SystemOneUsageV1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("generated_at", "expires_at")
    @classmethod
    def _frame_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def _complete_answer_set(self) -> "SystemOneAppraisalFrameV1":
        missing = set(self.questions) - set(self.answers)
        extra = set(self.answers) - set(self.questions)
        if missing:
            raise ValueError(f"missing System One answers: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected System One answers: {sorted(extra)}")

        for question_id, question in self.questions.items():
            answer = self.answers[question_id]
            if answer.type != question.type:
                raise ValueError(
                    f"answer type mismatch for {question_id!r}: "
                    f"{answer.type!r} != {question.type!r}"
                )

            if answer.type in {"choice", "score"}:
                total = sum(answer.probabilities.values())
                if abs(total - 1.0) > 0.05:
                    raise ValueError(
                        f"probabilities for {question_id!r} must sum to ~1; got {total}"
                    )

            if answer.type == "score":
                criteria = question.criteria
                if not isinstance(criteria, list) or not criteria:
                    raise ValueError(
                        f"score question {question_id!r} requires ordered criteria"
                    )
                expected_keys = {str(index) for index in range(len(criteria))}
                if set(answer.probabilities) != expected_keys:
                    raise ValueError(
                        f"score probability keys for {question_id!r} do not match criteria"
                    )
                if answer.score is None or not 0.0 <= answer.score <= len(criteria) - 1:
                    raise ValueError(
                        f"score for {question_id!r} is outside declared criteria range"
                    )
                if answer.legend and set(answer.legend) != expected_keys:
                    raise ValueError(
                        f"score legend keys for {question_id!r} do not match criteria"
                    )

            if answer.type == "choice":
                criteria = question.criteria
                if not isinstance(criteria, dict) or not criteria:
                    raise ValueError(
                        f"choice question {question_id!r} requires option criteria"
                    )
                expected_keys = set(criteria)
                if answer.choice not in expected_keys:
                    raise ValueError(
                        f"choice for {question_id!r} is not a declared option"
                    )
                if set(answer.probabilities) != expected_keys:
                    raise ValueError(
                        f"choice probability keys for {question_id!r} do not match options"
                    )

        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be after generated_at")
        return self
