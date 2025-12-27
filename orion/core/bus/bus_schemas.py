"""orion.core.bus.bus_schemas

Titanium bus contracts for Orion.

Design goals:
  - One universal envelope (versioned, strict, and future-proof)
  - Business logic sees typed payloads only (Pydantic V2)
  - Backwards-compatible parsing for legacy envelopes seen across the mesh

This module intentionally does **not** contain any Redis loops.
The redis/async orchestration lives in `bus_service_chassis.py`.
"""

from __future__ import annotations

import os
import socket
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Iterable, List, Literal, Optional, Tuple, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic.aliases import AliasChoices


PayloadT = TypeVar("PayloadT", bound=BaseModel)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _env_first(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default


class SourceInfo(BaseModel):
    """Where did this envelope come from?"""

    model_config = ConfigDict(extra="forbid")

    service: str
    node: str
    host: str
    pid: int
    instance_id: str
    version: Optional[str] = None

    @classmethod
    def detect(cls, *, service: str | None = None, version: str | None = None) -> "SourceInfo":
        svc = service or _env_first("ORION_SERVICE_NAME", "SERVICE_NAME", default="unknown-service") or "unknown-service"
        ver = version or _env_first("ORION_SERVICE_VERSION", "SERVICE_VERSION")
        node = _env_first("ORION_NODE", "NODE_NAME", default=socket.gethostname()) or socket.gethostname()
        host = socket.gethostname()
        pid = os.getpid()
        instance_id = _env_first("ORION_INSTANCE_ID", default=str(uuid4())) or str(uuid4())
        return cls(service=svc, node=node, host=host, pid=pid, instance_id=instance_id, version=ver)


class EnvelopeMeta(BaseModel):
    """Future-proof headroom.

    Keep this small and typed, but extensible.
    """

    model_config = ConfigDict(extra="forbid")

    # Scheduling / retries
    priority: int = 0
    attempt: int = 0
    max_attempts: int = 0

    # TTL/deadline hints (best-effort; pubsub can't guarantee)
    ttl_ms: Optional[int] = None
    deadline_at: Optional[datetime] = None

    # Routing hints
    shard_key: Optional[str] = None
    target: Optional[str] = None

    # Freeform tags (kept as a list of strings to stay strict)
    tags: List[str] = Field(default_factory=list)


class BaseEnvelope(BaseModel, Generic[PayloadT]):
    """Universal envelope for Orion Bus events.

    Strict by default. If you need to accept the messy real world, use
    `parse_legacy_to_v2(...)` helpers below and then validate.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # Contract versioning
    contract: Literal["orion.envelope"] = Field(
        default="orion.envelope",
        serialization_alias="schema",
        validation_alias=AliasChoices("schema", "contract"),
    )
    schema_version: str = "2.0.0"

    # Identity
    envelope_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        validation_alias=AliasChoices("correlation_id", "trace_id", "corr_id"),
    )

    # Lineage
    root_id: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)
    causality_chain: List[str] = Field(default_factory=list)

    # Routing
    event: str = Field(validation_alias=AliasChoices("event", "name", "type"))
    # NOTE: do not alias from "source" here; v2 uses `source: SourceInfo`.
    service: str = Field(default="unknown", validation_alias=AliasChoices("service", "service_name"))
    reply_to: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "reply_to",
            "reply_channel",
            "response_channel",
            "result_channel",
        ),
    )

    # Time
    ts: datetime = Field(default_factory=utcnow, validation_alias=AliasChoices("ts", "timestamp", "created_at"))

    # Metadata
    source: SourceInfo = Field(default_factory=SourceInfo.detect)
    meta: EnvelopeMeta = Field(default_factory=EnvelopeMeta)

    # Typed payload
    payload: PayloadT

    def spawn_child(self, *, event: str, payload: BaseModel, reply_to: str | None = None) -> "BaseEnvelope[BaseModel]":
        """Create a child envelope with preserved lineage.

        We specialize the generic envelope at runtime using the payload's class,
        so the returned envelope stays typed.
        """
        root_id = self.root_id or self.envelope_id
        chain = list(self.causality_chain)
        chain.append(self.envelope_id)

        payload_cls = payload.__class__
        EnvCls = BaseEnvelope[payload_cls]  # type: ignore[index]
        return EnvCls(
            event=event,
            service=self.service,
            correlation_id=self.correlation_id,
            root_id=root_id,
            parent_id=self.envelope_id,
            causality_chain=chain,
            reply_to=reply_to,
            payload=payload,
        )

    def to_bus_dict(self) -> Dict[str, Any]:
        """JSON-safe representation for Redis bus publishing."""
        return self.model_dump(mode="json", by_alias=True)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy compatibility helpers
# ──────────────────────────────────────────────────────────────────────────────


class LegacyEnvelope(BaseModel):
    """A permissive parser for the mess currently on the Orion bus.

    This accepts extra fields and multiple alias variants so we can coerce into
    the strict `BaseEnvelope`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True, str_strip_whitespace=True)

    event: str = Field(validation_alias=AliasChoices("event", "name", "type"))
    service: str = Field(default="unknown", validation_alias=AliasChoices("service", "source"))
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        validation_alias=AliasChoices("correlation_id", "trace_id", "corr_id"),
    )
    reply_to: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "reply_to",
            "reply_channel",
            "response_channel",
            "result_channel",
        ),
    )
    ts: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("ts", "timestamp", "created_at"))
    payload: Dict[str, Any] = Field(default_factory=dict)


def coerce_legacy_envelope(raw: Dict[str, Any]) -> LegacyEnvelope:
    """Parse raw dict as LegacyEnvelope, never raising non-validation errors."""
    return LegacyEnvelope.model_validate(raw)


def legacy_to_v2(
    raw: Dict[str, Any],
    *,
    payload_model: Type[PayloadT],
    default_event: str | None = None,
    default_service: str | None = None,
) -> BaseEnvelope[PayloadT]:
    """Coerce a legacy bus message into a strict v2 BaseEnvelope.

    - Extracts core fields (event/service/correlation/reply_to/payload)
    - Validates payload against a provided model
    """
    legacy = coerce_legacy_envelope(raw)

    event = legacy.event or (default_event or "unknown")
    service = legacy.service or (default_service or "unknown")
    ts = legacy.ts or utcnow()

    # Prefer nested payload; if caller sent body-only, treat the whole dict as payload.
    payload_raw: Dict[str, Any] = legacy.payload or {}
    if not payload_raw:
        # Some callers put business fields at top-level AND include a payload.
        # If payload is empty, try to use top-level as payload fallback.
        payload_raw = {k: v for k, v in raw.items() if k not in {"event", "name", "type", "service", "source", "correlation_id", "trace_id", "corr_id", "reply_channel", "response_channel", "result_channel", "reply_to", "ts", "timestamp", "created_at"}}

    payload_obj = payload_model.model_validate(payload_raw)

    env = BaseEnvelope[payload_model](
        event=event,
        service=service,
        correlation_id=str(legacy.correlation_id),
        reply_to=legacy.reply_to,
        ts=ts,
        payload=payload_obj,
        # lineage fields are optional; set minimal compatible defaults
        root_id=str(legacy.correlation_id),
        parent_id=None,
        causality_chain=[],
    )
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Canonical "v1" envelope still common across the mesh
# ──────────────────────────────────────────────────────────────────────────────


class ExecutionEnvelopeV1(BaseModel):
    """Common Orion bus message shape used by many existing services.

    This is intentionally *compat-oriented* (extra ignored) so we can
    incrementally migrate services to `BaseEnvelope` without breakage.

    Shape:
      { event, service, correlation_id|trace_id, reply_channel, payload }
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)

    event: str = Field(validation_alias=AliasChoices("event", "name", "type"))
    service: str = Field(default="unknown", validation_alias=AliasChoices("service", "source"))
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        validation_alias=AliasChoices("correlation_id", "trace_id", "corr_id"),
    )
    reply_channel: str = Field(
        validation_alias=AliasChoices(
            "reply_channel",
            "reply_to",
            "response_channel",
            "result_channel",
        )
    )
    payload: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Example payloads / envelopes (how services should define their contracts)
# ──────────────────────────────────────────────────────────────────────────────


class CollapseMirrorPayload(BaseModel):
    """Minimal canonical Collapse Mirror payload.

    Keep this strict. Add optional fields as you learn.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    observer: str
    kind: Literal["solo", "shared", "external"] = "solo"
    trigger: Optional[str] = None
    reflection: Optional[str] = None

    # Pillars / tags
    pillars: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # Freeform structured details (kept typed but flexible)
    fields: Dict[str, Any] = Field(default_factory=dict)


class CollapseMirrorEnvelope(BaseEnvelope[CollapseMirrorPayload]):
    """Strict envelope for a Collapse Mirror log event."""

    event: Literal["collapse_mirror.log"] = "collapse_mirror.log"
    payload: CollapseMirrorPayload


class VerbStep(BaseModel):
    """A single step inside a Verb execution."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    verb: str
    step: str
    order: int
    prompt_template: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    requires_gpu: bool = False
    requires_memory: bool = False


class OrchestrateVerbPayload(BaseModel):
    """Canonical orchestrate_verb request payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    verb_name: str
    origin_node: str
    context: Dict[str, Any] = Field(default_factory=dict)
    steps: List[VerbStep] = Field(default_factory=list)
    timeout_ms: Optional[int] = None


class OrchestrateVerbEnvelope(BaseEnvelope[OrchestrateVerbPayload]):
    """Strict envelope for requesting an orchestration."""

    event: Literal["cortex.orch.verb"] = "cortex.orch.verb"
    payload: OrchestrateVerbPayload


class SystemHealthPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["starting", "ok", "stopping", "error"] = "ok"
    uptime_s: float
    detail: Dict[str, Any] = Field(default_factory=dict)


class SystemErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    error_message: str
    traceback: str
    when: datetime = Field(default_factory=utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)


def build_error_payload(exc: BaseException, *, context: Dict[str, Any] | None = None) -> SystemErrorPayload:
    return SystemErrorPayload(
        error_type=exc.__class__.__name__,
        error_message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        context=context or {},
    )
