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
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    # True when the request set chat_template_kwargs.enable_thinking — read live
    # from run_plan's ctx (executor.py forwards the same key to the LLM gateway).
    # No caller in this codebase sets it True today, so this is currently always
    # False in practice, but it is a live read, not a hardcoded literal.
    thinking_enabled: bool = False
    emitted_at: datetime
    # --- Reserved / not yet wired by the cortex-exec producer -----------------
    # thinking_tokens: no provider in this stack (local vLLM via
    # orion-llm-gateway) returns a separate reasoning-token count in `usage`
    # (no `completion_tokens_details` field is ever parsed). Stays None until a
    # provider actually surfaces one. Consumers must treat it as absent, not 0.
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
    thinking_call_count: int = Field(default=0, ge=0)
    # thinking_tokens_sum stays None — see ReasoningCallV1.thinking_tokens (reserved).
    thinking_tokens_sum: Optional[int] = Field(default=None, ge=0)
