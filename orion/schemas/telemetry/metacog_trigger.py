from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetacogTriggerV1(BaseModel):
    trigger_kind: str = Field(
        ...,
        description=(
            "baseline | dense | manual | pulse | relational | llm_surface_instability | telemetry_anomaly | "
            "chat_turn "
            "(advisory: language-surface instability from logprob summary, not factual confidence). "
            "relational = a live repair_pressure_v2 appraisal (see repair_pressure_metacog_gate.py). "
            "telemetry_anomaly = a field_channel_corpus.v1 reconstruction-loss anomaly from a trained "
            "orion/mood_arc/fit_encoder.py encoder (see telemetry_anomaly_metacog_gate.py). "
            "chat_turn = a completed (or governor-timed-out) chat turn whose correlated "
            "ThoughtEventV1 + HarnessRunV1 (or exec_turn_timeout GrammarEventV1) evidence "
            "trips a gate condition (see chat_turn_metacog_gate.py)."
        ),
    )
    reason: str
    zen_state: str = Field("unknown", description="zen | not_zen | unknown")
    pressure: float = Field(0.0, ge=0.0, le=1.0)
    window_sec: int = 15
    frame_refs: List[str] = Field(default_factory=list)
    signal_refs: List[str] = Field(default_factory=list)
    upstream: Dict[str, Any] = Field(default_factory=dict, description="Compact summary of upstream trigger event")
    recall_enabled: Optional[bool] = Field(
        default=None,
        description="Override for downstream recall usage (true/false), None to use defaults.",
    )
    timestamp: str = Field(default_factory=_utc_now_iso)
