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
