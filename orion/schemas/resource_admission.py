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
    # "system" (2026-10-02, memory episode redesign): ahead of background holds in the pool's
    # queue and in durable-runs' driver order, below urgent. ADDITIVE on a forbid model: only
    # orion-durable-runs submits it (memory.episode_distill), so deploy durable-runs first.
    priority: Literal["background", "system", "urgent"] = "background"
    preferred_lane: str = "agent"
    requirements: dict[str, Any] = Field(default_factory=dict)
    deadline_at: datetime | None = None

    @model_validator(mode="after")
    def logical_resource(self):
        # service.route.<lane>: a hold on a non-LLM pool role (gpu_pool.yaml hold_routes).
        if self.resource not in (f"llm.route.{self.preferred_lane}", f"service.route.{self.preferred_lane}"):
            raise ValueError("resource must name the requested preferred logical lane")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must include a timezone")
        for key, value in self.requirements.items():
            if key.startswith("minimum_") and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0
            ):
                raise ValueError("minimum capability requirements must be finite nonnegative numbers")
        return self


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
