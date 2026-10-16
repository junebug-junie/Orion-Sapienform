from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from orion.schemas.repair_evidence import RepairEvidenceV1


class TurnWindowMessageV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    # When the message was said, if a real timestamp exists for it (the Hub's
    # receive time for a live websocket turn, or chat_history_log.created_at
    # for a rehydrated one). None means "unknown" -- never invented. Rendered
    # into the harness RECENT CONVERSATION block (orion/harness/prefix.py) so
    # the motor can tell order and elapsed time apart.
    at: datetime | None = None

    @model_serializer(mode="wrap")
    def _omit_unknown_at(self, handler):  # type: ignore[no-untyped-def]
        # Every consumer of this model validates with extra="forbid". Dropping
        # `at` when it is None keeps payloads that carry no timestamp (every
        # pre-turn appraisal request) byte-identical to before this field
        # existed, so an un-rebuilt consumer never sees a key it would reject.
        data = handler(self)
        if isinstance(data, dict) and data.get("at") is None:
            data.pop("at", None)
        return data


class PreTurnAppraisalOptionsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fail_closed: bool = True
    timeout_ms: int = Field(default=60000, ge=100, le=180000)
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
