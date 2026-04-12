"""Evidence-first answering contracts (parallel to legacy output_mode)."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


RequestKind = Literal["repo_technical", "runtime_debug", "conceptual", "personal", "mixed"]
PreferredRenderStyle = Literal["answer", "steps", "comparison", "recommendation"]
FindingEvidenceType = Literal[
    "repo_file",
    "runtime_log",
    "user_artifact",
    "user_statement",
    "inference",
]
FindingScope = Literal["fact", "interpretation", "proposal", "unknown"]
FindingsGroundedStatus = Literal["grounded_complete", "grounded_partial", "insufficient_grounding"]
InvestigationStatus = Literal["not_needed", "repo_required", "runtime_required", "mixed_required"]
AnswerGroundingStatusKind = Literal["grounded_complete", "grounded_partial", "insufficient_grounding"]


class AnswerContractDraft(BaseModel):
    """Hub-supplied partial contract; orch normalizes to AnswerContract."""

    model_config = ConfigDict(extra="ignore")

    request_kind: Optional[RequestKind] = None
    asks_for_explanation: Optional[bool] = None
    asks_for_steps: Optional[bool] = None
    asks_for_comparison: Optional[bool] = None
    asks_for_recommendation: Optional[bool] = None
    asks_for_action: Optional[bool] = None
    requires_repo_grounding: Optional[bool] = None
    requires_runtime_grounding: Optional[bool] = None
    requires_user_artifact_grounding: Optional[bool] = None
    allow_inference: Optional[bool] = None
    allow_unverified_specifics: Optional[bool] = None
    max_unverified_claims: Optional[int] = None
    preferred_render_style: Optional[PreferredRenderStyle] = None


class AnswerContract(BaseModel):
    """Epistemic control surface; delivery is late (renderer), not early."""

    model_config = ConfigDict(extra="forbid")

    request_kind: RequestKind = "conceptual"
    asks_for_explanation: bool = False
    asks_for_steps: bool = False
    asks_for_comparison: bool = False
    asks_for_recommendation: bool = False
    asks_for_action: bool = False

    requires_repo_grounding: bool = False
    requires_runtime_grounding: bool = False
    requires_user_artifact_grounding: bool = False

    allow_inference: bool = True
    allow_unverified_specifics: bool = False
    max_unverified_claims: int = 0

    preferred_render_style: PreferredRenderStyle = "answer"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_type: FindingEvidenceType
    source_ref: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0
    scope: FindingScope = "fact"


class FindingsBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: List[Finding] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    unsupported_requests: List[str] = Field(default_factory=list)
    next_checks: List[str] = Field(default_factory=list)
    grounded_status: FindingsGroundedStatus = "grounded_partial"


class RenderedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    structured: dict[str, Any] = Field(default_factory=dict)
    grounded_status: str = "grounded_partial"
    claims_used: int = 0
    unverified_claims_used: int = 0


class InvestigationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InvestigationStatus = "not_needed"
    evidence_acquired: bool = False
    findings_count: int = 0


class AnswerGroundingStatus(BaseModel):
    """User-visible grounding summary (§9.2)."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerGroundingStatusKind = "grounded_partial"
    reason: Optional[str] = None


# Spec §9.2 name
GroundingStatus = AnswerGroundingStatus
