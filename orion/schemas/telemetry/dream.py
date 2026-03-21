# orion/schemas/telemetry/dream.py
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DreamRequest(BaseModel):
    """
    Legacy payload shape for historical `dream.log` messages.

    Prefer `DreamResultV1` / `dream.result.v1` for durable dream artifacts.
    (Preserves prior behavior: extra ignored for forward-compat)
    """
    model_config = ConfigDict(extra="ignore")

    context_text: str
    mood: Optional[str] = None
    duration_seconds: int = 60
    integration_mode: str = "visual"  # "text", "visual", "audio"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DreamTriggerPayload(BaseModel):
    """Compatibility trigger payload on `dream.trigger` (minimal)."""
    model_config = ConfigDict(extra="forbid")

    mode: str = Field("standard", description="Dream mode / profile")


class DreamInternalTriggerV1(BaseModel):
    """
    Internal dream initiation contract (no final dream content).
    Used when normalizing schedulers/state into the canonical dream path.
    """
    model_config = ConfigDict(extra="ignore")

    trigger_id: Optional[str] = Field(default=None, description="Stable id for this trigger event")
    mode: str = Field("standard", description="Dream mode")
    profile: Optional[str] = Field(
        default=None,
        description="Recall profile override; default dream.v1 when unset",
    )
    source: Optional[str] = Field(default=None, description="Logical publisher (service or scheduler)")
    reason: Optional[str] = Field(default=None, description="Why the dream was scheduled")
    scheduled_for: Optional[str] = Field(default=None, description="ISO8601 or opaque schedule hint")
    state_overrides: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = Field(default=None, description="Correlation id when not envelope-level")


class DreamFragmentV1(BaseModel):
    """Fragment entry aligned with SQL writer DreamFragmentMeta."""
    model_config = ConfigDict(extra="ignore")

    id: str = "fragment-0"
    kind: str = "memory"
    salience: float = 0.0
    tags: List[str] = Field(default_factory=list)


class DreamMetricsV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_w: Optional[float] = None
    util: Optional[float] = None
    mem_mb: Optional[float] = None
    cpu_c: Optional[float] = None


class DreamResultV1(BaseModel):
    """
    Canonical dream artifact (envelope kind `dream.result.v1`).
    Source of truth for dream semantics; SQL projection derives from this model.
    """
    model_config = ConfigDict(extra="ignore")

    dream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dream_date: date = Field(default_factory=date.today)
    mode: str = "standard"
    profile: str = "dream.v1"
    trigger: Dict[str, Any] = Field(default_factory=dict)
    tldr: Optional[str] = None
    themes: List[str] = Field(default_factory=list)
    symbols: Dict[str, str] = Field(default_factory=dict)
    narrative: Optional[str] = None
    fragments: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    recall_debug: Dict[str, Any] = Field(default_factory=dict)
    source_context: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None

    def merged_metrics_for_sql(self) -> Dict[str, Any]:
        """Fold extended telemetry into metrics JSON for SQL without new columns."""
        base = dict(self.metrics or {})
        audit = {
            "dream_id": self.dream_id,
            "profile": self.profile,
            "mode": self.mode,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "trigger": self.trigger,
            "recall_debug": self.recall_debug,
            "source_context": self.source_context,
        }
        base["_dream_audit"] = audit
        return base
