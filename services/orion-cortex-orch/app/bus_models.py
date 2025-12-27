from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CortexOrchBusRequest(BaseModel):
    """Backward-compatible bus request model for Cortex Orchestrator.

    Hub currently sends a *legacy* envelope:
      { trace_id, result_channel, verb_name, origin_node, context, steps?, timeout_ms?, payload? }

    This model validates the fields we care about, ignores extras, and flattens
    nested `payload` (if present) so downstream code only sees one shape.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Hub uses `result_channel`; chassis looks for reply_channel/reply_to/etc.
    reply_channel: Optional[str] = Field(default=None, alias="result_channel")

    verb_name: str
    origin_node: str = "unknown"
    context: Dict[str, Any] = Field(default_factory=dict)

    # Optional explicit steps override YAML verb definition.
    steps: Optional[list[dict]] = None
    timeout_ms: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_payload(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        payload = values.get("payload")
        if isinstance(payload, dict):
            # Legacy senders sometimes include duplicated fields both at root
            # and under payload. Prefer explicit top-level keys if set.
            merged = {**payload, **values}
            merged.pop("payload", None)
            return merged

        return values


class CortexOrchBusReply(BaseModel):
    """What Cortex Orch publishes back to the requester."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    ok: bool
    kind: str = "cortex_orch_result"
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
