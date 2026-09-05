"""Append-only, versioned history of Orion's own self-concept claims --
the identity.yaml-replacement store named in the self-model rebuild arc's
design doc (docs/superpowers/specs/2026-09-03-orion-endogenous-self-model-
and-journal-design.md).

Additive to identity.yaml, not a replacement of it (explicit non-goal in
that doc). Two producers write here: Layer 3's real LLM reflection
(SelfReflectiveFindingV1) and Self Atlas's per-cluster LLM-written
descriptions (topic-foundry pointed at self-facts). Every claim is
diffable/revertable against its evidence, unlike an in-place identity.yaml
edit -- append-only means rollback never loses history, it only changes
what "current" resolves to (latest row per concept_id by created_at).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SelfConceptHistoryProducer = Literal["layer3_reflect", "self_atlas_cluster"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SelfConceptHistoryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    concept_id: str
    # Monotonically increasing per concept_id, computed by the producer at
    # write time (query current max + 1) -- informational, not load-bearing
    # for correctness: "current" is defined as latest row by created_at per
    # concept_id, matching this repo's other append-only history tables
    # (journal_entries, chat_stance_belief_log, self_knowledge_items).
    version: int = 1
    content: str
    evidence_refs: list[str] = Field(default_factory=list)
    produced_by: SelfConceptHistoryProducer
