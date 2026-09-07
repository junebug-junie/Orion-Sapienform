from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

COCKPIT_HOP_CHANNEL = "orion:cockpit:hop"

CockpitStageV1 = Literal[
    "ingress",
    "pre_turn_appraisal",
    "association",
    "thought_rpc",
    "mind_enrichment",
    "stance_inputs",
    "stance_decision",
    "harness_dispatch",
    "motor_boot",
    "motor_hop",
    "draft_appraisal",
    "finalize",
    "closure",
]

CockpitHopStatusV1 = Literal["started", "ok", "failed", "skipped", "gap"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CockpitHopV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cockpit.hop.v1"] = "cockpit.hop.v1"
    correlation_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    ts: datetime = Field(default_factory=_utc_now)
    stage: CockpitStageV1
    visor_line: str = Field(min_length=1, max_length=512)
    status: CockpitHopStatusV1
    summary: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    producer: str = Field(min_length=1, max_length=128)
