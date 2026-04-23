from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AutonomyGoalHeadlineV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_id: str
    goal_statement: str
    drive_origin: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_until: datetime | None = None
    proposal_signature: str


class AutonomyStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    subject: str
    model_layer: str
    entity_id: str
    latest_identity_snapshot_id: str | None = None
    latest_drive_audit_id: str | None = None
    latest_goal_ids: list[str] = Field(default_factory=list)
    identity_summary: str | None = None
    anchor_strategy: str | None = None
    dominant_drive: str | None = None
    active_drives: list[str] = Field(default_factory=list)
    drive_pressures: dict[str, float] = Field(default_factory=dict)
    tension_kinds: list[str] = Field(default_factory=list)
    goal_headlines: list[AutonomyGoalHeadlineV1] = Field(default_factory=list)
    source: str
    generated_at: datetime | None = None


class DriveCompetitionSummaryV1(BaseModel):
    """When tension.drive_competition.v1 is active: which drives disagree and by how much."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    top_drive: str
    runner_drive: str
    spread: float = Field(ge=0.0, le=1.0)
    pressure_top: float = Field(ge=0.0, le=1.0)
    pressure_runner: float = Field(ge=0.0, le=1.0)


class AutonomySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stance_hint: str
    dominant_drive: str | None = None
    top_drives: list[str] = Field(default_factory=list)
    active_tensions: list[str] = Field(default_factory=list)
    proposal_headlines: list[str] = Field(default_factory=list)
    response_hazards: list[str] = Field(default_factory=list)
    raw_state_present: bool = False
    drive_competition: DriveCompetitionSummaryV1 | None = None
