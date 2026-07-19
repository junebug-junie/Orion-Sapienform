# orion/schemas/repair_pressure_appraisal.py
"""Durable log of every repair_pressure_v2 appraisal, gated or not.

Built after a live-data check found the relational metacog trigger's only
observed appraisal in a week showed confidence=0.000 -- and that there was no
way to look further back, because repair_pressure_v2's result only ever lived
in ephemeral docker logs (wiped on container restart) with no Postgres
persistence at all. This schema is that persistence: every appraisal
published on `orion:repair_pressure:appraisal` gets logged here, whether or
not it crosses any gate/threshold, so the real level/confidence distribution
can be checked against actual outcomes instead of a single ephemeral sample.

See docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _appraisal_log_id() -> str:
    return f"repair_pressure_appraisal_{uuid4().hex}"


class RepairPressureAppraisalV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_appraisal_log_id)
    correlation_id: str
    created_at: str = Field(default_factory=_utc_now_iso)

    level: float
    level_label: str
    confidence: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    behavior_applied: Optional[str] = None
