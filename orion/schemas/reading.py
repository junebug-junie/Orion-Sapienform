"""Reading ingress contracts. Provenance is supplied by runtime bindings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

ReadingContext = Literal["unified_chat", "curiosity", "world_pulse"]
ReadingStatus = Literal[
    "queued",
    "started",
    "stage1_completed",
    "stage2_started",
    "landing_pending",
    "completed",
    "failed",
    "skipped",
    "not_found",
]


class ReadingRequestedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    requested_by: Literal["juniper", "orion", "world_pulse"]
    invocation_context: ReadingContext
    why_now: str = Field(default="", max_length=4000)
    title: str = Field(default="", max_length=1000)
    parent_run_id: str | None = None
    parent_trace_id: str | None = None
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    root_request_id: UUID | None = None
    parent_request_id: UUID | None = None

    @model_validator(mode="after")
    def coherent_provenance(self):
        expected = {"unified_chat": "juniper", "curiosity": "orion", "world_pulse": "world_pulse"}
        if self.requested_by != expected[self.invocation_context]:
            raise ValueError("requester does not match the runtime binding")
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must include a timezone")
        return self


class ReadingToolBindingV1(BaseModel):
    """Server-authored turn context; never part of model tool arguments."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    invocation_context: Literal["unified_chat", "curiosity"]
    parent_run_id: str
    parent_trace_id: str


class RecommendReadingArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=8192)
    why_now: str = Field(min_length=1, max_length=4000)


class ReadingStatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID


class ReadingToolRequestV1(BaseModel):
    """Internal ephemeral RPC. Only its post-commit reply proves acceptance."""
    model_config = ConfigDict(extra="forbid")
    operation: Literal["recommend_reading", "reading_status"]
    request: ReadingRequestedV1 | None = None
    request_id: UUID | None = None

    @model_validator(mode="after")
    def operation_arguments(self):
        if self.operation == "recommend_reading":
            if self.request is None or self.request_id is not None:
                raise ValueError("recommend_reading requires only request")
        elif self.request_id is None or self.request is not None:
            raise ValueError("reading_status requires only request_id")
        return self


class ReadingToolResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ReadingStatusReceiptV1(BaseModel):
    """Minimum typed status returned by the server-owned Postgres queue."""

    model_config = ConfigDict(extra="allow")
    request_id: UUID
    status: ReadingStatus


class DurableReadingReceiptV1(ReadingStatusReceiptV1):
    """A row-backed receipt. ``not_found`` can never prove acceptance."""

    status: Literal[
        "queued",
        "started",
        "stage1_completed",
        "stage2_started",
        "landing_pending",
        "completed",
        "failed",
        "skipped",
    ]
    seed_id: str = Field(min_length=1)


class ReadingRecommendationOutcomeV1(BaseModel):
    """Deterministic FCC grounding derived from one recommendation tool round-trip."""

    model_config = ConfigDict(extra="forbid")
    tool_use_ids: list[str] = Field(min_length=1)
    attempt_count: int = Field(ge=1)
    url: str
    acceptance: Literal["accepted", "unknown"]
    request_id: UUID | None = None
    status: ReadingStatus | None = None
    source_read: bool = False
    failure_kind: Literal[
        "tool_error", "rpc_timeout", "malformed_receipt", "missing_result"
    ] | None = None

    @model_validator(mode="after")
    def coherent_outcome(self):
        if self.attempt_count != len(self.tool_use_ids):
            raise ValueError("reading attempt count must match tool use IDs")
        if self.acceptance == "accepted":
            if self.request_id is None or self.status in (None, "not_found"):
                raise ValueError("accepted reading outcome requires a durable receipt")
            if self.failure_kind is not None:
                raise ValueError("accepted reading outcome cannot carry a failure kind")
        elif self.request_id is not None or self.status is not None or self.failure_kind is None:
            raise ValueError("unknown reading outcome requires only a failure kind")
        return self


class ReadingLifecycleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: ReadingRequestedV1
    seed_id: str
    stage: Literal["started", "stage1_completed", "stage1_failed", "stage2_started", "stage2_completed", "stage2_failed", "landing_completed"]
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    error: str | None = None
