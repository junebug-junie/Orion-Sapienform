from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorldPulseReadSeedV1(_Base):
    seed_id: str = Field(min_length=1)
    kind: Literal["finding", "digest_item"]
    run_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    section: str = ""
    item_id: str | None = None  # digest_item only


class WorldPulseReadConceptCandidateV1(_Base):
    label: str = Field(min_length=1)
    definition: str | None = None
    link_hints: list[str] = Field(default_factory=list)


class WorldPulseReadPriorCandidateV1(_Base):
    claim: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class WorldPulseReadHandoffV1(_Base):
    """Stage 1 → Stage 2 (and Concept Atlas) artifact."""

    seed_ref: WorldPulseReadSeedV1
    what_i_learned: str = Field(min_length=1)
    candidate_priors: list[WorldPulseReadPriorCandidateV1] = Field(default_factory=list)
    concept_candidates: list[WorldPulseReadConceptCandidateV1] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)
    created_at: datetime
    producer_hint: Literal["world_pulse_read_pipeline"] = "world_pulse_read_pipeline"


class WorldPulseReadStage2ResultV1(_Base):
    """Stage 2 FCC result. ``need_stage1_urls`` may trigger Stage 1 re-entry."""

    summary: str = Field(min_length=1)
    need_stage1_urls: list[str] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)
    created_at: datetime
    seed_id: str = ""
    producer_hint: Literal["world_pulse_read_stage2"] = "world_pulse_read_stage2"
