"""Operator-controlled substrate policy profile adoption contracts (Phase 17)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeSurfaceV1

SubstratePolicyActivationStateV1 = Literal["staged", "active", "inactive", "rolled_back"]
SubstratePolicyAdoptionActionV1 = Literal["staged", "activated", "rejected"]
SubstratePolicyRollbackTargetV1 = Literal["previous", "baseline", "profile"]


class SubstratePolicyRolloutScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_surfaces: list[GraphReviewRuntimeSurfaceV1] = Field(default_factory=list, max_length=8)
    target_zones: list[FrontierTargetZoneV1] = Field(default_factory=list, max_length=8)
    operator_only: bool = False


class SubstratePolicyOverridesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cycles_world: Optional[int] = Field(default=None, ge=1, le=24)
    max_cycles_concept: Optional[int] = Field(default=None, ge=1, le=24)
    max_cycles_autonomy: Optional[int] = Field(default=None, ge=1, le=24)
    max_cycles_self_relationship: Optional[int] = Field(default=None, ge=1, le=12)
    urgent_revisit_seconds: Optional[int] = Field(default=None, ge=60, le=86400)
    normal_revisit_seconds: Optional[int] = Field(default=None, ge=300, le=172800)
    slow_revisit_seconds: Optional[int] = Field(default=None, ge=600, le=604800)
    suppress_after_low_value_cycles: Optional[int] = Field(default=None, ge=1, le=10)
    frontier_followup_allowed: Optional[bool] = None
    query_limit_nodes: Optional[int] = Field(default=None, ge=8, le=256)
    query_limit_edges: Optional[int] = Field(default=None, ge=16, le=512)
    query_cache_enabled: Optional[bool] = None


class SubstratePolicyProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(default_factory=lambda: f"substrate-policy-profile-{uuid4()}")
    profile_version: int = Field(default=1, ge=1)
    source_recommendation_id: Optional[str] = None
    source_summary_window: Optional[str] = None
    rollout_scope: SubstratePolicyRolloutScopeV1 = Field(default_factory=SubstratePolicyRolloutScopeV1)
    policy_overrides: SubstratePolicyOverridesV1 = Field(default_factory=SubstratePolicyOverridesV1)
    activation_state: SubstratePolicyActivationStateV1 = "staged"
    previous_profile_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    operator_id: Optional[str] = None
    rationale: str = ""
    notes: list[str] = Field(default_factory=list, max_length=64)


class SubstratePolicyAdoptionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"substrate-policy-adoption-{uuid4()}")
    source_recommendation_id: Optional[str] = None
    source_summary_window: Optional[str] = None
    rollout_scope: SubstratePolicyRolloutScopeV1 = Field(default_factory=SubstratePolicyRolloutScopeV1)
    policy_overrides: SubstratePolicyOverridesV1 = Field(default_factory=SubstratePolicyOverridesV1)
    activate_now: bool = False
    operator_id: Optional[str] = None
    rationale: str = ""
    notes: list[str] = Field(default_factory=list, max_length=32)


class SubstratePolicyAdoptionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    profile_id: Optional[str] = None
    action_taken: SubstratePolicyAdoptionActionV1
    active_scope_summary: dict[str, Any] = Field(default_factory=dict)
    previous_active_profile_id: Optional[str] = None
    notes: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)


class SubstratePolicyRollbackRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"substrate-policy-rollback-{uuid4()}")
    rollback_target: SubstratePolicyRollbackTargetV1 = "previous"
    explicit_profile_id: Optional[str] = None
    rollout_scope: SubstratePolicyRolloutScopeV1 = Field(default_factory=SubstratePolicyRolloutScopeV1)
    operator_id: Optional[str] = None
    rationale: str = ""
    notes: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _validate_target(self) -> "SubstratePolicyRollbackRequestV1":
        if self.rollback_target == "profile" and not self.explicit_profile_id:
            raise ValueError("explicit_profile_id_required_for_profile_rollback")
        return self


class SubstratePolicyRollbackResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    action_taken: Literal["rolled_back", "rejected"]
    active_profile_id: Optional[str] = None
    previous_profile_id: Optional[str] = None
    notes: list[str] = Field(default_factory=list, max_length=64)
    warnings: list[str] = Field(default_factory=list, max_length=64)


class SubstratePolicyAuditEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"substrate-policy-audit-{uuid4()}")
    event_type: Literal["staged", "activated", "rolled_back", "deactivated"]
    profile_id: Optional[str] = None
    previous_profile_id: Optional[str] = None
    operator_id: Optional[str] = None
    rationale: str = ""
    rollout_scope: SubstratePolicyRolloutScopeV1 = Field(default_factory=SubstratePolicyRolloutScopeV1)
    notes: list[str] = Field(default_factory=list, max_length=64)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SubstratePolicyResolutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["baseline", "adopted"] = "baseline"
    profile_id: Optional[str] = None
    rollout_scope: SubstratePolicyRolloutScopeV1 = Field(default_factory=SubstratePolicyRolloutScopeV1)
    overrides: dict[str, Any] = Field(default_factory=dict)
    reason: str = "baseline_default"


class SubstratePolicyInspectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profiles: list[SubstratePolicyProfileV1] = Field(default_factory=list)
    staged_profiles: list[SubstratePolicyProfileV1] = Field(default_factory=list)
    rolled_back_profiles: list[SubstratePolicyProfileV1] = Field(default_factory=list)
    recent_audit_events: list[SubstratePolicyAuditEventV1] = Field(default_factory=list)


class SubstratePolicyComparisonV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_summary: dict[str, Any] = Field(default_factory=dict)
    active_profile_id: Optional[str] = None
    active_overrides: dict[str, Any] = Field(default_factory=dict)
    comparison_notes: list[str] = Field(default_factory=list, max_length=32)
