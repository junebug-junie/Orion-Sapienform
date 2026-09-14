"""Versioned exclusive-run admission contracts; no physical hosts in demands."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESOURCE_EVENT_CHANNEL = "orion:durable:resource:event"
RESOURCE_EVENT_KIND = "durable.resource.event.v1"


class ResourceRequirementV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource: str = "llm.route.agent"
    mode: Literal["exclusive"] = "exclusive"
    lease_scope: Literal["run"] = "run"
    priority: Literal["background"] = "background"
    preferred_lane: str = "agent"
    allow_elastic_activation: bool = False
    alternatives: list[str] = Field(default_factory=list)
    requirements: dict[str, Any] = Field(default_factory=dict)
    operator_override: str | None = None
    pinned_lane: str | None = None
    deadline_at: datetime | None = None

    @model_validator(mode="after")
    def logical_resource(self):
        if self.resource != f"llm.route.{self.preferred_lane}":
            raise ValueError("resource must name the requested preferred logical lane")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must include a timezone")
        for key, value in self.requirements.items():
            if key.startswith("minimum_") and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0
            ):
                raise ValueError("minimum capability requirements must be finite nonnegative numbers")
        return self


class ResourceLeaseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    demand_id: str
    lease_id: str
    resource_key: str
    lane: str
    backend_key: str
    generation: int = Field(ge=1)
    granted_at: datetime
    expires_at: datetime
    heartbeat_at: datetime
    status: Literal["active", "released", "expired"] = "active"


class ResourceEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["durable.resource.event.v1"] = RESOURCE_EVENT_KIND
    entry_id: str = Field(default_factory=lambda: uuid4().hex)
    event: str
    run_id: str
    thread_id: str
    correlation_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: dict[str, Any] = Field(default_factory=dict)


class CapacityAcquireV1(BaseModel):
    """Internal HTTP request permit; no model content or workflow state."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=256)
    lane: str = Field(min_length=1, max_length=128)
    backend_key: str = Field(min_length=1, max_length=2048)
    max_inflight: int = Field(ge=1, le=128)
    budget_sec: float = Field(gt=0, le=86400)
    lease: ResourceLeaseV1 | None = None


class CapacityTokenV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128)
    permit_id: str = Field(min_length=1, max_length=128)


class CapacityPermitV1(CapacityTokenV1):
    correlation_id: str
    lane: str
    backend_key: str
    lease_id: str | None = None
    generation: int | None = None
    granted_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    status: Literal["active", "released", "expired"]


class CapacityAcquireResultV1(BaseModel):
    acquired: bool
    reason: str
    permit: CapacityPermitV1 | None = None


class CapacityRenewResultV1(BaseModel):
    valid: bool
    permit: CapacityPermitV1 | None = None


class CapacityReleaseResultV1(BaseModel):
    released: bool
