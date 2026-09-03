from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1
from orion.schemas.thought import CoalitionSnapshotV1, ThoughtEventV1

# Bounded cap for HarnessRunRequestV1.recent_turns -- reused, not invented:
# matches PreTurnAppraisalOptionsV1.max_turns's existing default (orion/schemas/
# pre_turn_appraisal.py) and orion.substrate.appraisal.turn_window.build_turn_window's
# default max_turns, the same normalize+cap helper turn_orchestrator.py already
# calls one line above where recent_turns is built. Enforced here too (not just
# at the call site) so the invariant holds regardless of caller -- see
# CLAUDE.md's evidence_event_ids unbounded-growth precedent for why a turn-scoped
# list needs its own ceiling instead of trusting every producer to remember one.
HARNESS_RECENT_TURNS_MAX = 8


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
    # Set by orion.harness.tool_provenance_audit.detect_tool_provenance_mismatch
    # when draft_text uses live-immediacy language but grammar_receipts show a
    # fetch-shaped tool call this same turn. None on every normal turn.
    tool_provenance_audit: str | None = None


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

    # Loop-back tool-recall retry (see orion/harness/finalize.py::
    # maybe_run_finalize_tool_retry). Populated when alignment_verdict is
    # "misaligned" for a reason a specific, already-reviewed tool would fix,
    # OR when world_contact_opportunity is true (see below) -- the reflect
    # prompt is constrained to a small allowlist, so an unrecognized value
    # here is dropped by the harness, never dispatched.
    recommended_tool: str | None = None
    recommended_tool_reason: str | None = Field(default=None, max_length=300)

    # Independent of alignment_verdict: true when the user asked what is
    # currently visible/in the room right now, no matching tool was called
    # this turn, and the user did not instruct Orion not to use tools this
    # turn. An honest "I can't see, I'd need to check" draft can still set
    # this true -- honesty and having-a-live-tool-opportunity are unrelated
    # questions, and conflating them (the pre-2026-08-18 design) meant the
    # tool-recall retry could never fire for its own intended purpose: 3 live
    # turns confirmed a genuinely honest non-hallucinating draft is always
    # "aligned," so gating the retry on misalignment alone left it
    # structurally unreachable for a benign "what do you see" ask. See
    # orion/harness/finalize.py::maybe_run_finalize_tool_retry.
    world_contact_opportunity: bool = False


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

    # Loop-back tool-recall retry. When true, this turn's grammar_event_ids
    # legitimately includes one event from a finalize-stage correction, not
    # only motor-stage tool calls -- HarnessRunV1.step_count/grammar_event_ids
    # (published earlier, before finalize runs) do NOT reflect it, so a
    # retried turn's outcome-stage tool count can exceed its run-stage count.
    # That asymmetry is expected. Consumers reading grammar_event_ids as
    # evidence of deliberate motor tool-use (e.g. orion/memory/crystallization/
    # intake_autonomy_episode.py) should check this flag to tell "the motor
    # reached for this tool" apart from "finalize corrected a gap after the
    # fact" -- both are real evidence, but of different things.
    finalize_loop_retried: bool = False
    finalize_loop_tool: str | None = None


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


class HarnessAttachmentV1(BaseModel):
    """An image already staged into the FCC sandbox, addressed by path.

    Deliberately a *path*, not an ``AttachmentRefV1``: the harness-governor
    container cannot see ``/mnt/orion-chat-attachments`` at all (verified live
    2026-08-15 -- only the Hub mounts the attachment store), so a store
    reference would be unresolvable on the side that actually runs ``claude``.
    The Hub stages a copy into the shared sandbox and passes where it landed.

    The path is what lets Orion's *own* model look at the image with its own
    Read tool, first-person, rather than being handed a description produced by
    some other model. That distinction is the whole point -- see the rejected
    "captioner" option in
    ``docs/superpowers/specs/2026-08-14-hub-chat-vision-and-rendering-design.md``.
    """

    schema_version: Literal["harness.attachment.v1"] = "harness.attachment.v1"
    path: str = Field(..., description="Absolute path inside the FCC sandbox.")
    mime: str
    sha256: str
    filename: str | None = None


class HarnessRunRequestV1(BaseModel):
    schema_version: Literal["harness.run.request.v1"] = "harness.run.request.v1"
    correlation_id: str
    thought_event: ThoughtEventV1
    user_message: str
    permissions: ContextExecPermissionV1
    answer_contract: AnswerContract
    repair_pressure_contract: dict[str, Any] | None = None
    fcc_model_label: str | None = None
    # Which Hub chat mode ("orion" or "agent") initiated this run --
    # threaded through so orion-harness-governor's own grammar trace
    # (orion/harness/grammar_emit.py's HarnessGrammarCollector) can label
    # itself correctly instead of always saying "mode=orion" regardless of
    # caller. Optional/defaults to None (treated as "orion" downstream) so
    # a Hub build that predates this field doesn't break an older-vs-newer
    # rolling deploy against this schema, matching this model's existing
    # attachments/recent_turns/situation_prompt_fragment convention of
    # getattr(request, "...", None) at every read site.
    mode: str | None = None
    attachments: list[HarnessAttachmentV1] = Field(
        default_factory=list,
        description=(
            "Images staged into the sandbox for this turn. Empty for every "
            "text-only turn, which keeps the harness prompt byte-identical to "
            "its pre-attachment form."
        ),
    )
    situation_prompt_fragment: str | None = Field(
        default=None,
        description=(
            "Compact, already-length-capped situational-awareness text "
            "(local time-of-day/day-phase, conversation-phase, presence; "
            "see orion.situational.context.SituationPromptFragmentV1.compact_text) "
            "resolved by orion-hub BEFORE this request is built -- "
            "orion/hub/turn_orchestrator.py::execute_unified_turn -- and "
            "rendered verbatim into the harness prompt (orion/harness/"
            "prefix.py::compile_harness_prefix). None when situation "
            "context is disabled or failed to build; the prompt then omits "
            "the block entirely rather than showing a placeholder."
        ),
    )
    recent_turns: list[TurnWindowMessageV1] = Field(
        default_factory=list,
        max_length=HARNESS_RECENT_TURNS_MAX,
        description=(
            "Bounded recent user/assistant history for THIS session, oldest "
            "first, rendered into the harness prompt as a 'RECENT CONVERSATION' "
            "section (orion/harness/prefix.py::compile_harness_prefix). Empty "
            "for a genuinely fresh session -- never fabricated. Distinct from "
            "the single current `user_message` field: before this field "
            "existed, every unified-turn (mode='orion') `claude -p` prompt "
            "carried zero prior-turn history, so each turn was answered in "
            "isolation regardless of how long the conversation had run."
        ),
    )


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
    # Set only when finalize (5a substrate appraisal) could not be reached due to an
    # infra failure (e.g. substrate RPC timeout) and draft_text was used as final_text
    # instead. finalize_ran stays False (the real chain did not run) so this is the
    # signal Hub uses to treat the turn as a degraded success rather than a hard error.
    finalize_degraded_reason: str | None = None
    step_count: int
    exit_code: int | None = None
    compliance_verdict: Literal["completed", "partial", "failed", "refused"]
    grounding_status: str
    grammar_event_ids: list[str] = Field(default_factory=list)
    recall_debug: dict[str, Any] | None = None
    memory_digest: str | None = None
    # Real backend model discovered from the CLI's own stream-json "assistant"
    # events (see orion/harness/fcc_motor.py's _served_model_from_assistant),
    # distinct from HarnessRunRequestV1.fcc_model_label -- that's just the
    # ~/.fcc/.env route alias requested (e.g. "MODEL_SONNET"), which cannot
    # distinguish backends sharing one route. None when discovery never fired.
    fcc_served_model: str | None = None
    # Wall time for the FCC motor leg alone, seconds. The quantity
    # HARNESS_FCC_TIMEOUT_SEC (1600s) compares against, and therefore the one
    # that decides `grounding_status == "fcc_timeout"`. Optional because the
    # refusal/validation paths never run the motor -- absent there means "no
    # motor leg happened", NOT "it took no time".
    #
    # Hub previously had no way to see this: it could only time the whole
    # unified turn, which also spans the stance leg (<=400s) and the finalize
    # chain (<=485s). See services/orion-hub/scripts/curiosity_investigation.py.
    fcc_elapsed_sec: float | None = None
