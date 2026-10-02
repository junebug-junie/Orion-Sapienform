"""Memory episode contracts (spec 2026-09-30-memory-episode-redesign-design.md, Stage 1).

An episode is a maximal run of consecutive turns on one ``source_platform``
with no conversation boundary inside it. orion-memory-consolidation decides
boundaries (Rule 3, shadow) and publishes ``MemoryEpisodeClosedV1`` when one
closes. Consumers: the durable submitter for ``memory.episode_distill`` and the
shadow old-vs-new report.

Deploy order: these models are ``extra="forbid"``, so a consumer must be
deployed before a producer adds a field (consumer-first).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MEMORY_EPISODE_CLOSED_KIND = "memory.episode.closed.v1"

EpisodeStatus = Literal["closed", "skipped"]


class MemoryEpisodeClosedV1(BaseModel):
    """One shadow episode closed under boundary Rule 3.

    ``close_lag_sec`` is the time from the episode's last turn to the arrival
    of the turn that closed it. There is no idle timer (Juniper's decision), so
    this is the latency cost of waiting for the next turn, measured.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    source_platform: Optional[str] = None
    started_at: datetime
    ended_at: datetime
    closed_at: datetime
    turn_ids: List[str] = Field(default_factory=list)
    juniper_turn_count: int = 0
    command_turn_count: int = 0
    close_reason: str
    phase_at_close: Optional[str] = None
    boundary_score_at_close: Optional[float] = None
    close_lag_sec: Optional[float] = None
    closing_turn_id: Optional[str] = None
    episode_status: EpisodeStatus = "closed"
    skip_reason: Optional[str] = None
    boundary_rule: Literal["v2"] = "v2"
