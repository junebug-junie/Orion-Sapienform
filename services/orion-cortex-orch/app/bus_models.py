"""Bus-facing schemas for `orion-cortex-orch`.

Compatibility notes:
- `orion-hub` publishes a top-level envelope with a nested `payload` dict.
- Hub may use `result_channel` instead of `reply_channel`.

These models accept both and normalize before validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, model_validator


class CortexOrchBusRequestV1(BaseModel):
    """Envelope received on `orion-cortex:request`.

    This is intentionally tolerant:
    - accepts top-level fields or nested inside `payload`
    - accepts `trace_id` or `correlation_id`
    - accepts `result_channel` or `reply_channel`
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event: str = Field(default="orchestrate_verb", validation_alias=AliasChoices("event", "name", "type"))

    # RPC identity
    trace_id: str = Field(validation_alias=AliasChoices("trace_id", "correlation_id", "corr_id"))
    reply_channel: str = Field(
        validation_alias=AliasChoices("reply_channel", "reply_to", "result_channel", "response_channel")
    )

    # Request fields
    verb_name: str

    # Optional chat context
    text: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None

    # Generic args/context
    args: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Optional override (rare)
    steps: List[Dict[str, Any]] = Field(default_factory=list)

    timeout_ms: Optional[int] = Field(default=None)

    # Bookkeeping
    origin_node: str = Field(default="unknown")

    @model_validator(mode="before")
    @classmethod
    def _flatten_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = data.get("payload")
        if isinstance(payload, dict):
            # Only fill missing fields from payload.
            for k, v in payload.items():
                data.setdefault(k, v)

        # Some callers put origin inside payload.
        if "origin_node" not in data:
            data["origin_node"] = data.get("source") or data.get("origin") or "unknown"

        return data


class CortexOrchBusReplyV1(BaseModel):
    """Reply published to the request's `reply_channel`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ok: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
