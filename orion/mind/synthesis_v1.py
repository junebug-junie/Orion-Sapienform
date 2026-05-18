"""Mind semantic synthesis, appraisal, and stance handoff contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimKindV1 = Literal[
    "current_turn_claim",
    "continuity_claim",
    "relationship_claim",
    "task_claim",
    "identity_boundary_claim",
    "autonomy_claim",
    "social_claim",
    "situation_claim",
    "hazard_claim",
    "curiosity_affordance_claim",
    "uncertainty_claim",
]

RecommendedEffectV1 = Literal[
    "answer_directly",
    "receive_warmly",
    "ask_one_situated_question",
    "suppress_question",
    "preserve_identity_boundary",
    "avoid_identity_sermon",
    "technical_triage",
    "surface_uncertainty",
    "no_effect",
]

ClaimAnchorV1 = Literal["orion", "juniper", "relationship", "none", "mixed", "unknown"]

SuppressedReasonV1 = Literal[
    "source_tag_not_semantic",
    "identity_background_not_turn_specific",
    "duplicate",
    "empty",
    "too_generic",
    "stale_or_ungrounded",
    "unsupported_or_weak",
    "evidence_too_weak",
]

ExtractionModeV1 = Literal["llm", "deterministic_fallback", "hybrid"]

MatterKindV1 = Literal[
    "turn_anchor",
    "continuity_memory",
    "relationship_opportunity",
    "autonomy_pressure",
    "social_posture",
    "identity_boundary",
    "task_directive",
    "hazard",
    "curiosity_affordance",
    "uncertainty",
]

HandoffMindQualityV1 = Literal[
    "empty",
    "fallback_contract_only",
    "shadow_synthesis",
    "meaningful_synthesis",
    "invalid_handoff",
    "error",
]


class SemanticClaimV1(BaseModel):
    claim_id: str
    label: str
    summary: str
    claim_kind: ClaimKindV1
    evidence_refs: list[str] = Field(default_factory=list)
    source_kinds: list[str] = Field(default_factory=list)
    anchor: ClaimAnchorV1 = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience_hint: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_effect: RecommendedEffectV1 = "no_effect"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuppressedMindCandidateV1(BaseModel):
    label: str
    source_kind: str
    source_ref: str | None = None
    reason: SuppressedReasonV1
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSynthesisDiagnosticsV1(BaseModel):
    evidence_item_count: int = 0
    projection_item_count_seen: int = 0
    recall_fragments_seen: int = 0
    autonomy_fields_seen: int = 0
    social_fields_seen: int = 0
    llm_ok: bool = False
    llm_error: str | None = None
    notes: list[str] = Field(default_factory=list)


class SemanticSynthesisV1(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["mind.semantic_synthesis.v1"] = "mind.semantic_synthesis.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_id: str = "unknown"
    extraction_mode: ExtractionModeV1 = "llm"
    claims: list[SemanticClaimV1] = Field(default_factory=list)
    suppressed: list[SuppressedMindCandidateV1] = Field(default_factory=list)
    diagnostics: SemanticSynthesisDiagnosticsV1 = Field(default_factory=SemanticSynthesisDiagnosticsV1)


class AppraisalFeatureVectorV1(BaseModel):
    current_turn_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    source_corroboration: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolvedness: float = Field(default=0.0, ge=0.0, le=1.0)
    actionability: float = Field(default=0.0, ge=0.0, le=1.0)
    relationship_leverage: float = Field(default=0.0, ge=0.0, le=1.0)
    identity_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    interaction_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    source_tag_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SelectedFrontierMatterV1(BaseModel):
    matter_id: str
    source_claim_id: str
    label: str
    summary: str
    matter_kind: MatterKindV1
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    features: AppraisalFeatureVectorV1 = Field(default_factory=AppraisalFeatureVectorV1)
    evidence_refs: list[str] = Field(default_factory=list)
    reason_selected: str = ""
    recommended_effect: RecommendedEffectV1 = "no_effect"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeferredFrontierMatterV1(BaseModel):
    source_claim_id: str
    label: str
    reason_deferred: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveFrontierDiagnosticsV1(BaseModel):
    selected_count: int = 0
    deferred_count: int = 0
    suppressed_count: int = 0
    score_notes: list[str] = Field(default_factory=list)
    llm_ok: bool = False
    llm_error: str | None = None
    notes: list[str] = Field(default_factory=list)


class ActiveCognitiveFrontierV1(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["mind.active_cognitive_frontier.v1"] = "mind.active_cognitive_frontier.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_id: str = "unknown"
    selected: list[SelectedFrontierMatterV1] = Field(default_factory=list)
    deferred: list[DeferredFrontierMatterV1] = Field(default_factory=list)
    suppressed: list[SuppressedMindCandidateV1] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    response_directives: list[str] = Field(default_factory=list)
    diagnostics: ActiveFrontierDiagnosticsV1 = Field(default_factory=ActiveFrontierDiagnosticsV1)


class MindStanceHandoffV1(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["mind.stance_handoff.v1"] = "mind.stance_handoff.v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_id: str = "unknown"
    mind_quality: HandoffMindQualityV1 = "empty"
    semantic_synthesis: SemanticSynthesisV1 | None = None
    active_frontier: ActiveCognitiveFrontierV1 | None = None
    stance_payload: dict[str, Any] = Field(default_factory=dict)
    authorized_for_stance_use: bool = False
    authorization_reasons: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class MindEvidenceItemV1(BaseModel):
    evidence_ref: str
    source_kind: str
    text: str
    label: str | None = None
    source_ref: str | None = None
    item_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    trust_hint: str | None = None
    freshness_hint: str | None = None


class MindEvidencePackV1(BaseModel):
    schema_version: Literal["mind.evidence_pack.v1"] = "mind.evidence_pack.v1"
    current_user_text: str = ""
    items: list[MindEvidenceItemV1] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
