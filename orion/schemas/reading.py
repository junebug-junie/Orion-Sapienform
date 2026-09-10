"""Reading ingress contracts. Provenance is supplied by runtime bindings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

ReadingContext = Literal["unified_chat", "curiosity", "world_pulse"]


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


class ReadingLifecycleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: ReadingRequestedV1
    seed_id: str
    stage: Literal["started", "stage1_completed", "stage1_failed", "stage2_started", "stage2_completed", "stage2_failed", "landing_completed"]
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    error: str | None = None
