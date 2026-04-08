from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CapabilityRecoveryReasonV1(str, Enum):
    selected_verb_missing = "selected_verb_missing"
    invalid_bound_input = "invalid_bound_input"
    no_compatible_capability = "no_compatible_capability"
    policy_blocked = "policy_blocked"
    capability_executor_unavailable = "capability_executor_unavailable"
    internal_contract_error = "internal_contract_error"


class CapabilityRecoveryDecisionV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: CapabilityRecoveryReasonV1
    allow_replan: bool = False
    replanned: bool = False
    detail: Optional[str] = None


class BoundCapabilityExecutionRequestV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_verb: str
    normalized_action_input: Dict[str, Any] = Field(default_factory=dict)
    execution_mode: Literal["capability_backed"] = "capability_backed"
    requires_capability_selector: bool = True
    preferred_skill_families: list[str] = Field(default_factory=list)
    side_effect_level: Optional[Literal["none", "low", "moderate", "high"]] = None
    planner_correlation_id: Optional[str] = None
    planner_metadata: Dict[str, Any] = Field(default_factory=dict)
    selected_tool_metadata: Dict[str, Any] = Field(default_factory=dict)
    policy_metadata: Dict[str, Any] = Field(default_factory=dict)
    recovery: CapabilityRecoveryDecisionV1 = Field(
        default_factory=lambda: CapabilityRecoveryDecisionV1(
            reason=CapabilityRecoveryReasonV1.internal_contract_error,
            allow_replan=False,
            replanned=False,
            detail="default_fail_closed",
        )
    )


class BoundCapabilityExecutionFailureV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["fail"] = "fail"
    reason: CapabilityRecoveryReasonV1
    selected_verb: str
    execution_path: Literal["direct_execute", "recovery_replan"] = "direct_execute"
    detail: Optional[str] = None
    recovery: CapabilityRecoveryDecisionV1
    capability_decision: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)


class BoundCapabilityExecutionResultV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok"] = "ok"
    selected_verb: str
    normalized_action_input: Dict[str, Any] = Field(default_factory=dict)
    selected_skill_family: Optional[str] = None
    selected_skill: Optional[str] = None
    policy_metadata: Dict[str, Any] = Field(default_factory=dict)
    capability_decision: Dict[str, Any] = Field(default_factory=dict)
    structured_skill_output: Dict[str, Any] = Field(default_factory=dict)
    execution_path: Literal["direct_execute", "recovery_replan"] = "direct_execute"
    recovery: CapabilityRecoveryDecisionV1
