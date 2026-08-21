"""Phi encoder manifest schemas (Plan 2).

PhiIntrinsicRewardV1 and AttributionV1 (the intrinsic-reward telemetry half of
Plan 2) were removed 2026-08-21: orion-spark-introspector, their sole
producer, was retired outright 2026-07-28, and nothing else in the repo ever
constructed either class. See orion/inner_state_registry.py's
phi_intrinsic_reward.v1 entry for the full history.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CorpusStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_path: str
    row_count: int
    excluded_degenerate: int
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None


class TrainingStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int
    final_loss: float
    held_out_loss: float
    recon_error_p50: float
    recon_error_p95: float


class PhiEncoderManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_id: str
    encoder_version: str
    parent_version: Optional[str] = None
    status: Literal["candidate", "active", "retired"]
    architecture: str
    features_version: str
    input_features: List[str]
    hidden_dim: int
    latent_dim: int
    corpus: CorpusStatsV1
    training: TrainingStatsV1
    probes: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    git_sha: str
    trained_at: datetime
    promoted_at: Optional[datetime] = None
