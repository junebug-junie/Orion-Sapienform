"""Consult contracts: what a consult between Orion and Claude must hand Juniper.

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.

Not bus payloads. Hub owns a consult end to end and stores these as JSON in
``orion_consult.brief_json``. Claude's turns travel on the existing
``orion:room:claude:request`` / ``:utterance`` channels (a consult room id
starts with ``consult:``), so no new channel exists for them.

The brief gate is this module: a brief that does not validate is never sent,
and the validation error names the part that is missing.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ConsultEnding = Literal["continue", "answered", "agreed_proposal", "disagree"]
ConsultSourceKind = Literal["curiosity_run", "self_monitor", "stuck_prior"]
ConsultDecision = Literal["yes", "no", "no_because"]

FINAL_ENDINGS = ("answered", "agreed_proposal", "disagree")
BRIEF_ENDINGS = ("agreed_proposal", "disagree")


def _text(value: Optional[str]) -> str:
    return (value or "").strip()


class CheckedClaimV1(BaseModel):
    """One claim and its proof. No proof means it says so: ``not_checked=True``."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., min_length=1)
    proof: Optional[str] = None
    not_checked: bool = False

    @model_validator(mode="after")
    def _proof_or_label(self) -> "CheckedClaimV1":
        if not _text(self.claim):
            raise ValueError("claim is empty")
        has_proof = bool(_text(self.proof))
        if has_proof == self.not_checked:
            raise ValueError("a claim needs a proof (file:line or dig output) or not_checked=true, not both")
        return self


class CostLineV1(BaseModel):
    """A cost with its source, or ``unknown`` with the reason. An unsourced number fails."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[str] = None
    source: Optional[str] = None
    unknown_reason: Optional[str] = None

    @model_validator(mode="after")
    def _sourced_or_unknown(self) -> "CostLineV1":
        value, source, reason = _text(self.value), _text(self.source), _text(self.unknown_reason)
        if value and value.lower() != "unknown":
            if not source:
                raise ValueError("a cost number needs a source (dig command, file:line, or page)")
            return self
        if not reason:
            raise ValueError("an unknown cost needs unknown_reason")
        return self


class ImpactV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what_changes: str = Field(..., min_length=1)
    number_to_watch: str = Field(..., min_length=1)
    how_we_see_it: str = Field(..., min_length=1)

    @field_validator("what_changes", "number_to_watch", "how_we_see_it")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty")
        return value.strip()


class CostV1(BaseModel):
    """Doing the work and doing nothing. The consult's own cost is filled by Hub, not the model."""

    model_config = ConfigDict(extra="forbid")

    doing_the_work: CostLineV1
    doing_nothing: CostLineV1


class ConsultBriefV1(BaseModel):
    """What Juniper gets. Every part required; Hub adds where it came from and the consult cost."""

    model_config = ConfigDict(extra="forbid")

    why_it_matters: str = Field(..., min_length=1)
    checked: List[CheckedClaimV1] = Field(..., min_length=1)
    disagreement: str = Field(..., min_length=1)
    orion_position: Optional[str] = None
    claude_position: Optional[str] = None
    impact: ImpactV1
    cost: CostV1
    recommendation: str = Field(..., min_length=1)
    question_for_juniper: str = Field(..., min_length=1)

    @field_validator("why_it_matters", "disagreement", "recommendation", "question_for_juniper")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty")
        return value.strip()

    @field_validator("question_for_juniper")
    @classmethod
    def _yes_no(cls, value: str) -> str:
        if not value.rstrip().endswith("?"):
            raise ValueError("must be one yes/no question ending in '?'")
        return value


class ConsultAnswerV1(BaseModel):
    """An ``answered`` ending: what Orion keeps. Nothing goes to Juniper."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
