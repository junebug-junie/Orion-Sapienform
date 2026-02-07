from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TypedDict
from uuid import UUID


@dataclass
class RowBlock:
    row_ids: List[str]
    timestamps: List[str]
    doc_id: str
    text: str
    conversation_id: Optional[UUID] = None
    block_index: Optional[int] = None


@dataclass
class BoundaryContext:
    run_id: Optional[UUID]
    spec_hash: Optional[str]
    dataset_id: Optional[UUID]
    model_id: Optional[UUID]
    run_dir: Optional[str]


class BoundaryDecision(TypedDict, total=False):
    split: bool
    confidence: float
    reason: str
    topic_left: str
    topic_right: str
