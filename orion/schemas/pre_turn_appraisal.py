from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.substrate.appraisal.models import EvidenceKind, RepairEvidenceV1


class TurnWindowMessageV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class PreTurnAppraisalOptionsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fail_closed: bool = True
    timeout_ms: int = Field(default=800, ge=100, le=5000)
    max_turns: int = Field(default=8, ge=1, le=32)


class PreTurnAppraisalRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    session_id: str
    turn_window: list[TurnWindowMessageV1]
    paradigms_requested: list[str] = Field(default_factory=lambda: ["repair_pressure"])
    contract_before: dict[str, Any] = Field(default_factory=lambda: {"mode": "default"})
    options: PreTurnAppraisalOptionsV1 = Field(default_factory=PreTurnAppraisalOptionsV1)


class TurnAppraisalParadigmSliceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appraisal_kind: Literal["repair_pressure"]
    level: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    dimensions: dict[str, float] = Field(default_factory=dict)
    evidence: list[RepairEvidenceV1] = Field(default_factory=list)
    contract_delta: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TurnAppraisalBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    paradigms: dict[str, TurnAppraisalParadigmSliceV1] = Field(default_factory=dict)
    metadata_attachments: dict[str, Any] = Field(default_factory=dict)
    grammar_scalars: dict[str, dict[str, float]] = Field(default_factory=dict)
    failed_paradigms: list[str] = Field(default_factory=list)
