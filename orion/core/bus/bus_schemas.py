# orion/core/bus/bus_schemas.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ServiceRef(BaseModel):
    """
    Identity for the producer/consumer of a message.

    Keep this tiny + stable; you can always add optional fields later.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Logical service name (e.g. 'orion-hub').")
    node: Optional[str] = Field(None, description="Node/host identifier (e.g. 'athena', 'atlas').")
    version: Optional[str] = Field(None, description="Service version (semantic version recommended).")
    instance: Optional[str] = Field(None, description="Ephemeral instance id (container id / pod id).")


class CausalityLink(BaseModel):
    """
    A single step in a causality chain. Used for Conjourney lineage.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: UUID
    kind: str
    source: ServiceRef
    created_at: datetime = Field(default_factory=utcnow)


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    message: str
    stack: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class BaseEnvelope(BaseModel):
    """
    Titanium, universal, versioned envelope.

    - Strict (extra fields forbidden)
    - Versioned (schema_version)
    - Self-identifying (schema, kind)
    - Traceable (id, correlation_id, causality_chain)
    - Auditable (source, created_at)
    - RPC-friendly (reply_to)

    NOTE: We intentionally use `schema_id` with alias "schema" to avoid
    shadowing BaseModel.schema (pydantic warning).
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
        populate_by_name=True,  # allow using field names even when aliases exist
    )

    # Self-identification
    schema_id: Literal["orion.envelope"] = Field(
        "orion.envelope",
        alias="schema",
        description="Envelope schema identifier.",
    )
    schema_version: str = Field("2.0.0", description="Envelope schema version.")

    # Message identity
    id: UUID = Field(default_factory=uuid4, description="Unique message id.")
    correlation_id: UUID = Field(default_factory=uuid4, description="Stable id for a request/flow.")

    kind: str = Field(..., description="Canonical message kind (e.g. 'llm.chat.request').")

    # Conjourney lineage
    causality_chain: List[CausalityLink] = Field(default_factory=list)

    # Producer identity + timing
    source: ServiceRef
    created_at: datetime = Field(default_factory=utcnow)

    # Optional RPC return address
    reply_to: Optional[str] = Field(
        None,
        description="If set, the callee publishes the response to this channel.",
    )

    # Payload is defined by subclasses (typed). Keep as Dict here for backwards compatibility.
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def derive_child(
        self,
        *,
        kind: str,
        source: ServiceRef,
        payload: Any,
        reply_to: str | None = None,
    ) -> "BaseEnvelope":
        """
        Create a child envelope that extends the causality chain with this message as a parent.
        """
        parent_link = CausalityLink(
            correlation_id=self.correlation_id,
            kind=self.kind,
            source=self.source,
            created_at=self.created_at,
        )
        payload_dict = payload if isinstance(payload, dict) else payload.model_dump(mode="json")
        return BaseEnvelope(
            kind=kind,
            source=source,
            correlation_id=self.correlation_id,
            causality_chain=[*self.causality_chain, parent_link],
            reply_to=reply_to,
            payload=payload_dict,
        )


PayloadT = TypeVar("PayloadT", bound=BaseModel)


class Envelope(BaseEnvelope, Generic[PayloadT]):
    """
    Typed envelope: payload is a Pydantic model.

    This is the preferred new path for all V2 services.
    """
    payload: PayloadT


# ─────────────────────────────────────────────────────────────
# Example payloads (extend as your platform grows)
# ─────────────────────────────────────────────────────────────

class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatRequestPayload(BaseModel):
    """
    Request payload for LLM chat.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Optional[str] = None
    profile: Optional[str] = None
    messages: List[LLMMessage]
    options: Dict[str, Any] = Field(default_factory=dict)

    # Optional provenance
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResultPayload(BaseModel):
    # Avoid pydantic protected namespace warning for fields that include "model_"
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_used: Optional[str] = None
    content: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)


class RecallRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    max_items: int = 8
    time_window_days: int = 90
    mode: str = "hybrid"
    tags: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None


class RecallResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class CollapseMirrorPayload(BaseModel):
    """
    Example: collapse mirror journal entry intake (skeleton).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    observer: str
    trigger: str
    summary: str
    pillar: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)
