from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from orion.mind.constants import MIND_RUN_ARTIFACT_SCHEMA_ID


class MindRunArtifactV1(BaseModel):
    """Bus + Postgres artifact for a completed Mind run (producer: orch, consumer: sql-writer)."""

    schema_id: str = Field(default=MIND_RUN_ARTIFACT_SCHEMA_ID)
    mind_run_id: UUID
    correlation_id: str
    session_id: str | None = None
    trigger: str
    ok: bool
    error_code: str | None = None
    snapshot_hash: str = ""
    router_profile_id: str = ""
    result_jsonb: dict[str, Any] = Field(default_factory=dict)
    request_summary_jsonb: dict[str, Any] = Field(default_factory=dict)
    redaction_profile_id: str | None = None
    created_at_utc: datetime
