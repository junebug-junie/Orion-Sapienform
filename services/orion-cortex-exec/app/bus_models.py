"""orion-cortex-exec bus contracts.

Cortex Exec sits *between* Cortex Orch and downstream exec-target services.

- Orch sends an RPC request describing a single step and its target service calls.
- Exec fans out to `EXEC_REQUEST_PREFIX:{service}` channels.
- Exec collects `exec_step_result` messages on `EXEC_RESULT_PREFIX:{trace_id}`.
- Exec returns an aggregated reply to Orch on `reply_channel`.

These are intentionally forgiving (extra fields ignored) to support gradual
migration across the mesh.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, model_validator


class ExecServiceCallV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    service: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class CortexExecStepRequestV1(BaseModel):
    """RPC request Orch → Exec."""

    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field(
        ...,
        validation_alias=AliasChoices("trace_id", "exec_id", "correlation_id"),
    )
    reply_channel: str = Field(
        ...,
        validation_alias=AliasChoices(
            "reply_channel",
            "result_channel",
            "response_channel",
            "reply_to",
        ),
    )

    verb_name: str = Field(..., validation_alias=AliasChoices("verb_name", "verb"))
    step_name: str = Field(..., validation_alias=AliasChoices("step_name", "step"))
    order: int = 0
    origin_node: str = "unknown"

    timeout_ms: Optional[int] = None
    calls: List[ExecServiceCallV1] = Field(default_factory=list)

    # Back-compat: some callers wrap everything under payload.
    payload: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        pl = data.get("payload")
        if isinstance(pl, dict):
            for k, v in pl.items():
                data.setdefault(k, v)
        return data


class CortexExecStepReplyV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trace_id: str
    ok: bool = True
    elapsed_ms: Optional[int] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
