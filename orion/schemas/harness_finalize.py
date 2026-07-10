from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.thought import CoalitionSnapshotV1, ThoughtEventV1


class GrammarReceiptV1(BaseModel):
    step_index: int
    tool_name: str | None = None
    summary: str
    grammar_event_id: str | None = None


class HarnessDraftMoleculeV1(BaseModel):
    schema_version: Literal["harness.draft.molecule.v1"] = "harness.draft.molecule.v1"
    correlation_id: str
    thought_event_id: str
    draft_text: str
    draft_hash: str
    thought_event: ThoughtEventV1
    grammar_receipts: list[GrammarReceiptV1] = Field(default_factory=list)
    coalition_snapshot: CoalitionSnapshotV1
    repair_overlay_mode: str | None = None


class SubstrateFinalizeAppraisalV1(BaseModel):
    schema_version: Literal["substrate.finalize.appraisal.v1"] = "substrate.finalize.appraisal.v1"
    correlation_id: str
    molecule_id: str
    draft_hash: str

    surprise_level: float = Field(ge=0.0, le=1.0)
    strain_shift_refs: list[str] = Field(default_factory=list)
    open_loop_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    prediction_error_refs: list[str] = Field(default_factory=list)
    learning_refs: list[str] = Field(min_length=1)
    alignment_hints: list[str] = Field(default_factory=list)

    tick_source: Literal["substrate_runtime_finalize_appraisal"] = (
        "substrate_runtime_finalize_appraisal"
    )


class FinalizeReflectionV1(BaseModel):
    schema_version: Literal["finalize.reflection.v1"] = "finalize.reflection.v1"
    correlation_id: str
    thought_event_id: str
    substrate_appraisal_id: str
    draft_hash: str

    imperative: str
    tone: str
    strain_refs: list[str]

    alignment_verdict: Literal["aligned", "misaligned", "uncertain"]
    alignment_notes: list[str]
    strain_unresolved: bool

    reflection_source: Literal[
        "substrate_informed_pass",
        "deterministic_quick_gate",
        "degraded_llm_failure_fallback",
    ] = (
        "substrate_informed_pass"
    )
    quick_lane_skipped_llm: bool = False
    finalize_changed: bool = False


class HarnessVerdictMoleculeV1(BaseModel):
    schema_version: Literal["harness.verdict.molecule.v1"] = "harness.verdict.molecule.v1"
    correlation_id: str
    reflection: FinalizeReflectionV1
    cortex_trace_id: str | None = None


class HarnessTurnOutcomeMoleculeV1(BaseModel):
    schema_version: Literal["harness.turn.outcome.v1"] = "harness.turn.outcome.v1"
    correlation_id: str
    thought_event_id: str
    substrate_appraisal_id: str
    reflection_id: str
    verdict_molecule_id: str
    draft_hash: str
    final_hash: str
    finalize_changed: bool
    alignment_verdict: Literal["aligned", "misaligned", "uncertain"]
    surprise_level_at_draft: float
    surprise_resolved: bool
    grammar_event_ids: list[str] = Field(default_factory=list)
    final_text: str
    finalize_failed: bool = False
    failure_reason: str | None = Field(default=None, max_length=500)
    draft_text_excerpt: str | None = Field(default=None, max_length=300)


class HarnessPostTurnClosureV1(BaseModel):
    schema_version: Literal["harness.post_turn.closure.v1"] = "harness.post_turn.closure.v1"
    correlation_id: str
    outcome_molecule_id: str
    verdict_molecule_id: str
    grammar_event_ids: list[str] = Field(default_factory=list)
    surprise_unresolved: bool
    closure_source: Literal["harness_post_turn_appraisal"] = "harness_post_turn_appraisal"
    user_message_excerpt: str = Field(default="", max_length=300)
    stance_imperative: str = Field(default="", max_length=300)
    thought_event_id: str | None = None


class HarnessRepairOverlayV1(BaseModel):
    schema_version: Literal["harness.repair.overlay.v1"] = "harness.repair.overlay.v1"
    mode: Literal["default", "concrete_bias", "repair_concrete"] = "default"
    rule_lines: list[str] = Field(default_factory=list)
    prefix_overlay: str = ""
    finalize_overlay: str = ""


class HarnessRunRequestV1(BaseModel):
    schema_version: Literal["harness.run.request.v1"] = "harness.run.request.v1"
    correlation_id: str
    thought_event: ThoughtEventV1
    user_message: str
    permissions: ContextExecPermissionV1
    answer_contract: AnswerContract
    repair_pressure_contract: dict[str, Any] | None = None
    fcc_model_label: str | None = None


class HarnessRunCancelV1(BaseModel):
    """Fire-and-forget cancel for an in-flight FCC motor turn (Hub disconnect / abort)."""

    schema_version: Literal["harness.run.cancel.v1"] = "harness.run.cancel.v1"
    correlation_id: str
    reason: str = "client_disconnect"


class HarnessRunStepV1(BaseModel):
    schema_version: Literal["harness.run.step.v1"] = "harness.run.step.v1"
    correlation_id: str
    step_index: int
    step: dict[str, Any]


class HarnessRunV1(BaseModel):
    schema_version: Literal["harness.run.v1"] = "harness.run.v1"
    correlation_id: str
    final_text: str | None
    draft_text: str | None = None
    substrate_appraisal: SubstrateFinalizeAppraisalV1 | None = None
    reflection: FinalizeReflectionV1 | None = None
    verdict_molecule_id: str | None = None
    finalize_ran: bool
    finalize_changed: bool = False
    quick_lane_skipped_5b: bool = False
    step_count: int
    exit_code: int | None = None
    compliance_verdict: Literal["completed", "partial", "failed", "refused"]
    grounding_status: str
    grammar_event_ids: list[str] = Field(default_factory=list)
    recall_debug: dict[str, Any] | None = None
    memory_digest: str | None = None
