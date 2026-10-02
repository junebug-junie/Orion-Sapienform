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


# --- Stage 1 PR 2: the shadow distiller (memory.episode_distill) ---------------------------

MEMORY_EPISODE_DISTILL_WORKFLOW = "memory.episode_distill"
MEMORY_EPISODE_DISTILL_PROMPT_VERSION = "memory_episode_distill.v1"

Purpose = Literal["happened", "about_juniper", "orion_view", "follow_up"]
Voice = Literal["juniper_said", "worked_out_together", "orion_thought", "orion_read", "orion_self_knowledge"]
Channel = Literal[
    "chat", "reverie", "curiosity", "dream", "reading", "journal", "topic_model", "graphify", "legacy_crystallization"
]
Stakes = Literal["low", "high"]
StakesReason = Literal[
    "health", "family", "identity_conclusion_about_juniper", "relationship", "safety_location", "orion_self_conclusion"
]
EvidenceField = Literal["prompt", "response"]


class EpisodeDistillBriefV1(BaseModel):
    """What a memory.episode_distill run needs, built from a MemoryEpisodeClosedV1.

    Turns are NOT carried here: the run's first node reads their full, untruncated
    text from chat_history_log by id (the revision-1 lesson: never judge against
    truncated text) and checkpoints it.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    source_platform: Optional[str] = None
    turn_ids: List[str] = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    close_reason: str
    llm_route: str = "memory_distill"
    timeout_sec: float = Field(gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    prompt_version: str = MEMORY_EPISODE_DISTILL_PROMPT_VERSION


class DistillEvidenceV1(BaseModel):
    """One quote. ``turn`` is the episode-local label the prompt shows (t1, t2, ...)."""

    model_config = ConfigDict(extra="ignore")

    turn: str
    field: EvidenceField
    quote: str


class DistillReferentV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    role: str = "about"
    aliases: List[str] = Field(default_factory=list)


class DistilledMemoryV1(BaseModel):
    """One memory as the distiller proposes it. Lenient (``extra="ignore"``): this is LLM
    output, and every field that matters is re-checked by orion.memory.episode.validate."""

    model_config = ConfigDict(extra="ignore")

    purpose: Purpose
    voice: Voice
    channel: Channel = "chat"
    statement: str
    occurred_at: Optional[str] = None
    stakes: Stakes = "low"
    stakes_reason: Optional[StakesReason] = None
    asks_direction: bool = False
    referents: List[DistillReferentV1] = Field(default_factory=list)
    evidence: List[DistillEvidenceV1] = Field(default_factory=list)
    due_after: Optional[str] = None
    expires_at: Optional[str] = None


class DistilledQuestionV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    kind: Literal["question", "tension", "contradiction"] = "question"
    scope: Literal["self", "juniper", "relationship", "world"] = "juniper"
    answer_via: Literal["investigation", "conversation"] = "conversation"
    referents: List[DistillReferentV1] = Field(default_factory=list)
    evidence: List[DistillEvidenceV1] = Field(default_factory=list)


class EpisodeDistillationV1(BaseModel):
    """The distiller's whole answer for one episode (structured JSON)."""

    model_config = ConfigDict(extra="ignore")

    memories: List[DistilledMemoryV1] = Field(default_factory=list)
    questions: List[DistilledQuestionV1] = Field(default_factory=list)
