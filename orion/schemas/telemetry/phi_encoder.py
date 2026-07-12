"""Phi encoder manifest + intrinsic reward schemas (Plan 2)."""
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


class AttributionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    raw_value: float
    scaled_value: float
    attribution: float


class PhiIntrinsicRewardV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    self_state_id: Optional[str] = None
    encoder_version: str
    features_version: str
    phi: float
    delta_phi: float
    recon_error: float
    delta_recon_error: float
    latent: Dict[str, float] = Field(default_factory=dict)
    tick_metadata_window_sec: int = 2
    phi_health: str = "ok"
    grammar_truth_degraded: bool = False
    attribution_top: List[AttributionV1] = Field(default_factory=list)
    # 2026-07-12, inner-state unification Phase 2: node-attributed embodiment.
    # Sourced from SelfStateV1.dominant_attention_target_details, filtered to
    # real hardware nodes only (excludes synthetic pseudo-nodes and non-node
    # target kinds like "field:recent_perturbations" -- see
    # services/orion-spark-introspector/app/worker.py's
    # _dominant_hardware_node()). None when no qualifying node is present
    # this tick.
    dominant_node: Optional[str] = None
    dominant_node_reason: Optional[str] = None
