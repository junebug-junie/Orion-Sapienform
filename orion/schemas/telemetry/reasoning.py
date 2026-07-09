"""Reasoning telemetry — per-call cognition metadata + windowed activity.

`ReasoningCallV1` is emitted per LLM call at cortex-exec ingress: metadata ONLY
(booleans + token counts + verb/mode). The reasoning trace / thinking text is
NEVER carried — `reasoning_trace_present` is a bool. `ReasoningActivityV1` is the
rolling-window aggregate orion-thought materializes and exposes for φ to read,
mirroring the substrate `execution_trajectory` projection shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReasoningCallV1(BaseModel):
    """One LLM call's reasoning metadata. No trace text — privacy-preserving."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    turn_id: Optional[str] = None
    verb: str = "unknown"
    mode: str = "unknown"
    node_id: str = "unknown"
    reasoning_present: bool = False
    reasoning_trace_present: bool = False
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    emitted_at: datetime
    # --- Reserved / not yet wired by the cortex-exec producer -----------------
    # These fields are part of the contract but the current producer cannot
    # populate them, so they stay at their defaults until wired. Consumers must
    # treat them as absent, not as measured zeros.
    #   thinking_enabled: needs the `enable_thinking` template flag threaded from
    #     pre_turn_appraisal into run_plan's ctx (a follow-on wire).
    #   prompt_tokens: not surfaced by `_extract_final_text` diagnostics today.
    #   thinking_tokens: no provider exposes a separate thinking-token count yet.
    thinking_enabled: bool = False
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    thinking_tokens: Optional[int] = Field(default=None, ge=0)


class ReasoningActivityV1(BaseModel):
    """Rolling-window aggregate of ReasoningCallV1, read by φ (spark-introspector)."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    window_sec: float = Field(ge=0.0)
    call_count: int = Field(default=0, ge=0)
    reasoning_call_count: int = Field(default=0, ge=0)
    reasoning_present_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    completion_tokens_sum: int = Field(default=0, ge=0)
    completion_tokens_p50: float = Field(default=0.0, ge=0.0)
    by_mode: Dict[str, int] = Field(default_factory=dict)
    # Derived from ReasoningCallV1's reserved thinking_* fields — stay 0 / None
    # until the producer wires `thinking_enabled` / `thinking_tokens` (see above).
    thinking_call_count: int = Field(default=0, ge=0)
    thinking_tokens_sum: Optional[int] = Field(default=None, ge=0)
