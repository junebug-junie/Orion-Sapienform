"""Manual calibration profile adoption and rollback contracts (Phase 12)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationRolloutScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    invocation_surfaces: list[str] = Field(default_factory=list)
    workflow_types: list[str] = Field(default_factory=list)
    mentor_path: Literal["any", "mentor_enabled_only", "mentor_disabled_only"] = "any"
    sample_percent: int = Field(default=100, ge=1, le=100)
    operator_only: bool = False


class CalibrationProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    profile_id: str = Field(default_factory=lambda: f"calibration-profile-{uuid4()}")
    profile_version: int = Field(default=1, ge=1)
    source_recommendation_ids: list[str] = Field(default_factory=list)
    source_evaluation_request_id: Optional[str] = None
    overrides: dict[str, str] = Field(default_factory=dict)
    scope: CalibrationRolloutScopeV1 = Field(default_factory=CalibrationRolloutScopeV1)
    state: Literal["staged", "active", "rolled_back"] = "staged"
    previous_profile_id: Optional[str] = None
    created_by: str
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None


class CalibrationAdoptionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action: Literal["stage", "activate"]
    operator_id: str
    rationale: str
    profile: Optional[CalibrationProfileV1] = None
    profile_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "CalibrationAdoptionRequestV1":
        if self.action == "stage" and self.profile is None:
            raise ValueError("profile_required_for_stage")
        if self.action == "activate" and not self.profile_id:
            raise ValueError("profile_id_required_for_activate")
        return self


class CalibrationAdoptionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    accepted: bool
    action: Literal["stage", "activate"]
    profile: Optional[CalibrationProfileV1] = None
    active_profile_id: Optional[str] = None
    previous_profile_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationRollbackRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    rollback_mode: Literal["to_previous", "to_baseline"] = "to_previous"
    operator_id: str
    rationale: str


class CalibrationRollbackResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    accepted: bool
    active_profile_id: Optional[str] = None
    rolled_back_profile_id: Optional[str] = None
    restored_profile_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationProfileAuditV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    audit_id: str = Field(default_factory=lambda: f"calibration-audit-{uuid4()}")
    event_type: Literal["staged", "activated", "rolled_back", "reverted_to_baseline"]
    operator_id: str
    rationale: str
    profile_id: Optional[str] = None
    previous_profile_id: Optional[str] = None
    details: dict[str, str] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationProfileResolutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["baseline", "adopted"] = "baseline"
    profile_id: Optional[str] = None
    overrides: dict[str, str] = Field(default_factory=dict)
    reason: str = "baseline_default"
