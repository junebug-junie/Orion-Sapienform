from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


JournalTriggerKind = Literal[
    "daily_summary",
    "collapse_response",
    "metacog_digest",
    "manual",
    "notify_summary",
]
JournalSourceKind = Literal[
    "notify",
    "collapse_mirror",
    "metacog",
    "manual",
    "scheduler",
    "self_study",
    "self_reflection",
]
JournalMode = Literal["daily", "response", "digest", "manual"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JournalTriggerV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_kind: JournalTriggerKind
    source_kind: JournalSourceKind
    source_ref: str | None = None
    summary: str
    prompt_seed: str | None = None


class JournalEntryDraftV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: JournalMode
    title: str | None = None
    body: str


class JournalEntryWriteV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    author: str
    mode: JournalMode
    title: str | None = None
    body: str
    source_kind: JournalSourceKind | None = None
    source_ref: str | None = None
    correlation_id: str | None = None


class JournalEntryIndexV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    created_at: datetime
    author: str
    mode: JournalMode
    title: str | None = None
    body: str

    source_kind: str | None = None
    source_ref: str | None = None
    correlation_id: str | None = None
    trigger_kind: str | None = None
    trigger_summary: str | None = None

    conversation_frame: str | None = None
    task_mode: str | None = None
    identity_salience: str | None = None
    answer_strategy: str | None = None
    stance_summary: str | None = None

    active_identity_facets: list[str] | None = None
    active_growth_axes: list[str] | None = None
    active_relationship_facets: list[str] | None = None
    social_posture: list[str] | None = None
    reflective_themes: list[str] | None = None
    active_tensions: list[str] | None = None
    dream_motifs: list[str] | None = None
    response_hazards: list[str] | None = None
