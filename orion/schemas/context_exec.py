"""Bounded RLM context-exec investigation schemas."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.cognition.answer_contract import AnswerContract, FindingsBundle, Finding

ContextExecMode = Literal[
    "belief_provenance",
    "trace_autopsy",
    "repo_impact_analysis",
    "patch_proposal",
    "runtime_debug",
    "grammar_collision_audit",
    "memory_contradiction_review",
    "general_investigation",
]


class ContextExecPermissionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_memory: bool = True
    read_graph: bool = True
    read_recall: bool = True
    read_repo: bool = False
    read_runtime_logs: bool = False
    read_redis_traces: bool = True

    write_memory: bool = False
    write_graph: bool = False
    write_repo: bool = False
    mutate_runtime: bool = False
    network_enabled: bool = False
    shell_enabled: bool = False


class ContextExecBudgetV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_seconds: float = 45.0
    max_hops: int = 8
    max_subcalls: int = 6
    max_depth: int = 1
    max_repl_output_chars: int = 8192
    max_artifact_chars: int = 24000
    max_repo_file_chars: int = 12000
    max_trace_hits: int = 40


class ContextExecRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    text: str
    mode: ContextExecMode = "general_investigation"

    session_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None

    messages: list[LLMMessage] = Field(default_factory=list)
    answer_contract: AnswerContract | None = None

    allowed_verbs: list[str] = Field(default_factory=list)
    packs: list[str] = Field(default_factory=list)

    scopes: dict[str, Any] = Field(default_factory=dict)
    permissions: ContextExecPermissionV1 = Field(default_factory=ContextExecPermissionV1)
    budget: ContextExecBudgetV1 = Field(default_factory=ContextExecBudgetV1)

    expected_artifact_type: str | None = None


class ContextExecFindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_type: Literal[
        "repo_file", "runtime_log", "user_artifact", "user_statement", "inference"
    ]
    source_ref: str | None = None
    verified: bool = False
    confidence: float = 0.0
    scope: Literal["fact", "interpretation", "proposal", "unknown"] = "unknown"


class ContextExecVerbStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    verb: str
    callable: str | None = None
    input_summary: str | None = None
    output_handle: str | None = None
    output_summary: str | None = None
    status: Literal["ok", "error", "skipped", "blocked"] = "ok"
    duration_ms: int = 0


class TraceHitV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str
    source: str
    corr_id: str | None = None
    run_id: str | None = None
    kind: str
    timestamp: str
    snippet: str = ""
    payload_ref: str | None = None


class BeliefProvenanceReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    status: Literal[
        "supported", "unsupported", "contradicted", "stale", "inferred", "unknown"
    ]
    likely_origin: str | None = None
    first_seen: str | None = None
    source_chain: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    recommended_action: Literal[
        "keep", "mark_uncertain", "delete_candidate", "ask_user", "no_action"
    ] = "no_action"
    findings: list[ContextExecFindingV1] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class TraceAutopsyReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    status: Literal["explained", "partial", "unknown"]
    failure_chain: list[str] = Field(default_factory=list)
    root_cause: str | None = None
    contributing_factors: list[str] = Field(default_factory=list)
    evidence: list[ContextExecFindingV1] = Field(default_factory=list)
    recommended_patch: str | None = None


class RepoImpactAnalysisReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_change: str
    status: Literal["analyzed", "partial", "insufficient_grounding"]
    affected_paths: list[str] = Field(default_factory=list)
    breaking_surfaces: list[str] = Field(default_factory=list)
    compatibility_shims: list[str] = Field(default_factory=list)
    tests_to_add_or_update: list[str] = Field(default_factory=list)
    migration_steps: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    findings: list[ContextExecFindingV1] = Field(default_factory=list)


ProposalRiskLevel = Literal["low", "medium", "high", "unknown"]

ProposalReviewState = Literal[
    "draft",
    "pending_review",
    "approved",
    "rejected",
    "superseded",
    "executed",
]

ContextExecCreatableReviewState = Literal["draft", "pending_review"]

ProposalArtifactType = Literal[
    "patch_proposal",
    "memory_correction_proposal",
    "runtime_config_proposal",
    "test_plan_proposal",
]

# Context-exec may only create draft or pending_review envelopes.
CONTEXT_EXEC_CREATABLE_REVIEW_STATES: frozenset[str] = frozenset({"draft", "pending_review"})


class PatchProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str
    evidence: list[str] = Field(default_factory=list)
    files_to_change: list[str] = Field(default_factory=list)
    proposed_change_summary: str
    risk: ProposalRiskLevel = "unknown"
    tests_to_run: list[str] = Field(default_factory=list)
    rollback_plan: str
    open_questions: list[str] = Field(default_factory=list)
    mutation_allowed: bool = False


class ProposalEnvelopeV1(BaseModel):
    """Shared review wrapper for context-exec proposal artifacts."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    proposal_type: ProposalArtifactType
    source_mode: str
    source_run_id: str | None = None
    created_by: Literal["context-exec"] = "context-exec"
    created_at: str | None = None

    title: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    risk: ProposalRiskLevel = "unknown"

    requires_human_approval: bool = True
    mutation_allowed: bool = False
    review_status: ProposalReviewState = "draft"

    artifact_type: str
    artifact: dict[str, Any]

    open_questions: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


def assert_context_exec_proposal_safe(envelope: ProposalEnvelopeV1) -> None:
    """Context-exec may only emit draft/pending_review envelopes with mutation disallowed."""
    if envelope.review_status not in CONTEXT_EXEC_CREATABLE_REVIEW_STATES:
        raise ValueError(
            f"context-exec may only emit review_status in "
            f"{sorted(CONTEXT_EXEC_CREATABLE_REVIEW_STATES)}; got {envelope.review_status!r}"
        )
    if envelope.mutation_allowed:
        raise ValueError("context-exec proposals must set mutation_allowed=false")
    if not envelope.requires_human_approval:
        raise ValueError("context-exec proposals must set requires_human_approval=true")


def build_patch_proposal_envelope(
    patch: PatchProposalV1,
    *,
    source_mode: str,
    source_run_id: str | None = None,
    review_status: ContextExecCreatableReviewState = "draft",
) -> ProposalEnvelopeV1:
    """Wrap a PatchProposalV1 in a context-exec proposal envelope."""
    inner = patch.model_dump(mode="json")
    title = (patch.problem[:120] if patch.problem else "Patch proposal").strip()
    envelope = ProposalEnvelopeV1(
        proposal_id=f"prop_{uuid.uuid4().hex[:12]}",
        proposal_type="patch_proposal",
        source_mode=source_mode,
        source_run_id=source_run_id,
        title=title,
        summary=patch.proposed_change_summary,
        evidence=list(patch.evidence),
        risk=patch.risk,
        requires_human_approval=True,
        mutation_allowed=False,
        review_status=review_status,
        artifact_type="PatchProposalV1",
        artifact=inner,
        open_questions=list(patch.open_questions),
        safety_notes=[
            "Proposal artifacts are not actions.",
            "Proposal envelopes are review objects.",
            "Executors are separate.",
            "Cortex/human approval is required before mutation.",
            "Context-exec may draft proposals, but it may not approve or execute them.",
        ],
    )
    assert_context_exec_proposal_safe(envelope)
    return envelope


class ContextExecRunV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["ok", "error", "timeout", "schema_invalid", "policy_blocked"]

    mode: ContextExecMode
    text: str
    answer_contract: dict[str, Any] | None = None

    findings_bundle: FindingsBundle | None = None
    artifact_type: str | None = None
    artifact: dict[str, Any] = Field(default_factory=dict)

    final_text: str
    verb_trace: list[ContextExecVerbStepV1] = Field(default_factory=list)

    runtime_debug: dict[str, Any] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)


def finding_to_context_exec(f: Finding) -> ContextExecFindingV1:
    return ContextExecFindingV1(
        claim=f.claim,
        evidence_type=f.evidence_type,
        source_ref=f.source_ref,
        verified=f.verified,
        confidence=f.confidence,
        scope=f.scope,
    )
