from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.world_pulse import WorldPulseRunResultV1


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str | None = None
    dry_run: bool | None = None
    requested_by: str = "manual"
    fixtures: bool = False


class PublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    run_id: str
    detail: str = ""


def new_run_id() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RunCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    result: WorldPulseRunResultV1
    created_at: datetime = Field(default_factory=now_utc)
