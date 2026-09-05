"""Durable, multi-run history of self_study.py's Layer-1 SelfKnowledgeItemV1
items.

Part of the Orion self-model rebuild arc, Patch 2 (prerequisite for Patch 3:
pointing topic-foundry's real clustering pipeline at self-facts instead of
chat). Confirmed live 2026-09-05: a self_repo_inspect run's Layer-1 items
were never durably stored across runs -- the journal write is an explicit
one-off summary, "not treated as storage of record"
(services/orion-cortex-exec/app/self_study.py::build_self_study_summary).
Topic-foundry's DatasetSpec needs a real, growing, queryable source_table
with stable id/time/text columns to cluster over -- this is that table.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SelfKnowledgeItemLogV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    item_id: str
    run_id: str
    category: str
    name: str
    trust_tier: str
    observed_at: str
    source_path: str
    symbol_name: str | None = None
    # Flattened, plain-text rendering of the item's metadata dict -- what
    # topic-foundry's DatasetSpec.text_columns actually clusters over. Kept
    # separate from a raw JSONB metadata column on purpose: the clustering
    # pipeline wants text, not a dict it would have to know how to flatten
    # itself.
    metadata_text: str | None = None
