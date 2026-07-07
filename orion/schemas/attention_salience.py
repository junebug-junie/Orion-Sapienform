"""Telemetry + operator-surface contracts for computed salience.

- AttentionSalienceTraceV1: every scored loop's feature vector + score (learning
  telemetry; the feature-distribution + input half of the label join).
- AttentionLoopOutcomeV1: the human verdict (Resolve/Dismiss) or implicit
  decay — the sparse-but-clean label the refit later trains on.
- PendingAttentionCardV1: operator-legible card. Never id-only (hard UX rule).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


AttentionOutcomeVerdictV1 = Literal["resolved", "dismissed", "decayed_unattended"]
PendingCardStatusV1 = Literal["pending", "resolved", "dismissed"]

MAX_FEATURE_LIST = 16


class AttentionSalienceTraceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.salience.trace.v1"] = "attention.salience.trace.v1"
    trace_id: str
    loop_id: str
    theme_key: str
    correlation_id: str | None = None
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    weights_version: str = "seed-v1"
    features: dict[str, Any] = Field(default_factory=dict)
    scope: str = "reverie"  # reverie | chat | broadcast
    created_at: datetime = Field(default_factory=_utc_now)


class AttentionLoopOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.loop.outcome.v1"] = "attention.loop.outcome.v1"
    outcome_id: str
    loop_id: str
    theme_key: str
    verdict: AttentionOutcomeVerdictV1
    actor: str = "juniper"
    note: str = Field(default="", max_length=500)
    salience_at_close: float = Field(default=0.0, ge=0.0, le=1.0)
    features_at_close: dict[str, Any] = Field(default_factory=dict)
    weights_version: str = "seed-v1"
    created_at: datetime = Field(default_factory=_utc_now)


class PendingAttentionCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.pending.card.v1"] = "attention.pending.card.v1"
    loop_id: str
    theme_key: str
    title: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    what_triggered: str = ""
    narrative: str = ""
    age_seconds: float = Field(default=0.0, ge=0.0)
    recurrence_count: int = Field(default=0, ge=0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    weights_version: str = "seed-v1"
    top_contributing_features: list[str] = Field(default_factory=list, max_length=MAX_FEATURE_LIST)
    source: Literal["cognitive_loop"] = "cognitive_loop"
    status: PendingCardStatusV1 = "pending"
