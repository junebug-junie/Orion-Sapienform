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

    Rotation (2026-07-13): the sink this writes through
    (InnerStateCorpusSink, shared with INNER_FEATURES_CORPUS_PATH) now
    rotates at CORPUS_SINK_MAX_BYTES (default 200MB) and keeps at most
    CORPUS_SINK_ROTATED_KEEP (default 5) rotated siblings -- see
    services/orion-spark-introspector/app/inner_state_sink.py. Unlike
    InnerStateFeaturesV1 (recoverable via scripts/backfill_phi_corpus.py
    from Postgres), there is NO backfill path for pruned mood-arc rows --
    once a rotated file ages past the retention count, that slice of
    history is genuinely gone, not just archived. At the default policy
    (200MB x 5 = up to ~1GB retained) this is generous relative to the
    "weeks, not months" scope roadmap item 2 needs, but is a real,
    permanent-not-recoverable loss if collection runs far longer than
    that unattended.
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
