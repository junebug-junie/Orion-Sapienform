"""Episodic continuity contracts (self-modeling loop, rung 4).

An episode is a proposal-marked rollup of one time-window of reduction
receipts: derived autobiographical memory, never authoritative truth. It is
excluded from execution context by default and must go through review before
anything treats it as accepted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

# Hard cap on receipts folded into a single episode — same bound discipline as
# the evidence-id caps in the pressure/execution reducers.
EPISODE_RECEIPT_CAP = 64


class EpisodeSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.episode_summary.v1"] = "substrate.episode_summary.v1"
    episode_id: str
    status: Literal["proposal"] = "proposal"
    window_start: datetime
    window_end: datetime
    window_seconds: int = Field(ge=1)
    receipt_refs: List[str] = Field(default_factory=list, max_length=EPISODE_RECEIPT_CAP)
    receipt_count_total: int = Field(default=0, ge=0)
    receipt_count_capped: bool = False
    organ_counts: Dict[str, int] = Field(default_factory=dict)
    reducer_counts: Dict[str, int] = Field(default_factory=dict)
    accepted_event_count: int = Field(default=0, ge=0)
    rejected_event_count: int = Field(default=0, ge=0)
    merged_event_count: int = Field(default=0, ge=0)
    noop_event_count: int = Field(default=0, ge=0)
    state_delta_count: int = Field(default=0, ge=0)
    projection_update_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    sample_warnings: List[str] = Field(default_factory=list, max_length=8)
    notes: List[str] = Field(default_factory=list, max_length=16)
    created_at: datetime
