"""InnerStateFeaturesV1 — Orion's honest, decontaminated inner-state vector.

Felt+cognitive `features` are the only signals φ reads. `infra` signals are
retained for provenance/equilibrium and are NEVER read by φ. Proven-dead
signals (`policy_pressure`, `uncertainty`) are excluded entirely.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InnerFeatureV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    raw_value: float
    scaled_value: float
    source: str


class InnerStateFeaturesV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    features_version: str = "seed-v1"
    generated_at: datetime
    self_state_id: Optional[str] = None
    source_service: str = "spark-introspector"

    # felt + cognitive signals — the ONLY inputs φ reads
    features: List[InnerFeatureV1] = Field(default_factory=list)
    # infra provenance only — NEVER read by φ
    infra: List[InnerFeatureV1] = Field(default_factory=list)

    # cold-start honest headline (arithmetic; no geometric floor)
    headline: float = 0.5
    headline_source: str = "cold_start_aggregate"  # or "encoder" (Plan 2)

    # GIGO guard
    phi_health: str = "ok"  # ok | degenerate | frozen
    phi_degenerate_streak: int = 0
    grammar_truth_degraded: bool = False

    liveness: Dict[str, bool] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
