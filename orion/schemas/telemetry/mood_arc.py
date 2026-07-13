"""Mood-arc corpus row schema -- Item 1 of docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict


class MoodArcCorpusRowV1(BaseModel):
    """One per-tick training-data row for the not-yet-built windowed felt-
    state autoencoder (roadmap item 2). REHEARSAL status -- no cognition
    consumer by design, see orion/self_state/inner_state_registry.py.

    dominant_node is null on any tick where the phi encoder didn't run
    (disabled/degraded/failed) -- _dominant_hardware_node() is only called
    inside handle_self_state()'s encoder-success branch, an existing
    characteristic inherited from Phase 2/3, not introduced here. Do not
    read a null dominant_node here as "no salient node this tick"; it may
    just mean the encoder was down.

    No rotation or retention limit on the sink this writes through
    (InnerStateCorpusSink, shared with INNER_FEATURES_CORPUS_PATH) -- that
    sink has already grown unbounded in this deployment (confirmed
    2026-07-13: ~98MB/36k rows in 5 days for its existing use). Do not
    leave MOOD_ARC_CORPUS_PATH set indefinitely without a manual retention
    plan; this project has a prior incident from exactly this class of
    unbounded-write bug.
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    self_state_id: str
    coherence: float
    energy: float
    novelty: float
    valence: float
    valence_source: Literal["proxy", "heuristic"]
    dominant_node: Optional[str] = None
