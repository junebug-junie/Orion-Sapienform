from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from orion.cognition.cortex_payload_extract import cortex_exec_failure_detail, extract_cortex_payload_text
from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.llm_json import parse_json_object
from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import EmbodimentIntentV1
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    GrammarReceiptV1,
    HarnessDraftMoleculeV1,
    HarnessPostTurnClosureV1,
    HarnessRepairOverlayV1,
    HarnessTurnOutcomeMoleculeV1,
    HarnessVerdictMoleculeV1,
    SubstrateFinalizeAppraisalV1,
)
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.substrate.ids import stable_hash_id
from orion.thought.policy_refusal import TRUST_RUPTURE_DEFER_THRESHOLD

CortexClientFn = Callable[[PlanExecutionRequest], Awaitable[dict[str, Any]]]
SubstrateClientFn = Callable[[HarnessDraftMoleculeV1], Awaitable[SubstrateFinalizeAppraisalV1]]
PublishFn = Callable[..., Awaitable[None]]

logger = logging.getLogger("orion.harness.finalize")

DEFAULT_QUICK_GATE_EPSILON = 0.08
OPEN_LOOP_PRESSURE_MAX = 0.2
REPAIR_PRESSURE_MAX = 0.3

# Loop-back tool-recall retry (5b-prime): one bounded retry when reflection
# names a specific, fixable grounding gap. Fixed constant today -- a real
# candidate for Orion's own autonomy/drive-economy system to throttle or
# accelerate later (spend more retries when curiosity/resource pressure
# favors it, fewer when the turn is already expensive). Named here, not
# built, so the idea isn't lost when that system exists.
MAX_FINALIZE_LOOP_RETRIES = 1

# Semantic verb names the reflect prompt is allowed to recommend. Kept in
# lockstep with the TOOL RECOMMENDATION section of harness_finalize_reflect.j2
# -- an unrecognized name is logged and dropped, never dispatched.
FINALIZE_LOOP_TOOL_ALLOWLIST = frozenset({"look_at_camera"})

DEFAULT_GRAMMAR_EVENT_CHANNEL = "orion:grammar:event"

VERDICT_CHANNEL = "orion:harness:verdict:artifact"
OUTCOME_CHANNEL = "orion:substrate:turn_outcome"
POST_TURN_CLOSURE_CHANNEL = "orion:substrate:post_turn_closure"
SYSTEM_ERROR_CHANNEL = "orion:system:error"
EMBODIMENT_INTENT_CHANNEL = "orion:embodiment:intent"

# Interlocutor the finalized (relational) turn should approach in the town when
# EMBODIMENT_D_FINALIZE_ENABLED is set. Juniper is Orion's default human player.
DEFAULT_FINALIZE_INTERLOCUTOR = "Juniper"


def _quick_gate_epsilon() -> float:
    raw = os.environ.get("FINALIZE_QUICK_GATE_EPSILON", str(DEFAULT_QUICK_GATE_EPSILON))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_QUICK_GATE_EPSILON


def _embodiment_d_finalize_enabled() -> bool:
    raw = os.environ.get("EMBODIMENT_D_FINALIZE_ENABLED", "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _finalize_tool_loop_enabled() -> bool:
    raw = os.environ.get("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def build_finalize_embodiment_intent(
    *,
    correlation_id: str,
    interlocutor_ref: str | None,
    relational: bool,
) -> EmbodimentIntentV1 | None:
    """Pure (D) intent for a finalized relational turn.

    A relational turn with a known interlocutor produces a ``deliberate``
    ``approach_player`` intent carrying the turn ``correlation_id`` so the town
    body drifts toward whoever Orion just spoke with. Non-relational turns (no
    interlocutor / not relational) produce no intent.
    """
    if not relational or not interlocutor_ref:
        return None
    return build_intent(
        kind="approach_player",
        source="deliberate",
        reason=f"finalized relational turn -> approach {interlocutor_ref}",
        correlation_id=correlation_id,
        ref=interlocutor_ref,
    )


async def emit_finalize_embodiment_intent(
    *,
    correlation_id: str,
    interlocutor_ref: str | None,
    relational: bool,
    channel: str = EMBODIMENT_INTENT_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> EmbodimentIntentV1 | None:
    """Gated, fail-open emit of the finalize (D) intent. Never breaks a turn."""
    if not _embodiment_d_finalize_enabled():
        return None
    try:
        intent = build_finalize_embodiment_intent(
            correlation_id=correlation_id,
            interlocutor_ref=interlocutor_ref,
            relational=relational,
        )
        if intent is None:
            return None
        if publish_fn is not None:
            await publish_fn(intent, channel=channel)
        elif bus is not None:
            from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

            envelope = BaseEnvelope(
                kind="embodiment.intent.v1",
                source=ServiceRef(name="orion-harness-governor"),
                correlation_id=uuid.UUID(correlation_id) if _is_uuid(correlation_id) else uuid.uuid4(),
                payload=intent.model_dump(mode="json"),
            )
            await bus.publish(channel, envelope)
        else:
            return intent
        logger.info(
            "embodiment_finalize_intent_published corr=%s channel=%s ref=%s",
            correlation_id,
            channel,
            interlocutor_ref,
        )
        return intent
    except Exception:
        logger.warning("embodiment_finalize_intent_emit_failed corr=%s", correlation_id, exc_info=True)
        return None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def grammar_receipt_summaries(receipts: list[GrammarReceiptV1] | None) -> list[dict[str, str]]:
    return [
        {
            "step": str(receipt.step_index),
            "tool": receipt.tool_name or "",
            "summary": receipt.summary,
        }
        for receipt in (receipts or [])
    ]


def tools_called_this_turn(receipts: list[GrammarReceiptV1] | None) -> list[dict[str, str]]:
    """Deterministic list of tool calls executed this turn (from grammar receipts)."""
    return [
        {"step": str(r.step_index), "tool": r.tool_name}
        for r in (receipts or [])
        if r.tool_name
    ]


def format_tool_execution_digest(receipts: list[GrammarReceiptV1] | None) -> str:
    """Human-readable tool-execution digest for finalize prompts.

    Surfaces the deterministic fact that world-contact happened this turn so the
    reflect/voice passes cannot confabulate "no tool call succeeded" when one did.
    """
    calls = tools_called_this_turn(receipts)
    if not calls:
        return "none — no tools were called this turn"
    return "\n".join(
        f"- step {c['step']}: {c['tool']} (executed; results are in grammar_receipts)"
        for c in calls
    )


def quick_lane_block_reason(
    *,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    thought: ThoughtEventV1,
    repair_overlay: HarnessRepairOverlayV1,
    epsilon: float | None = None,
) -> str | None:
    """Return a block reason when quick lane is disallowed; None when eligible."""
    eps = epsilon if epsilon is not None else _quick_gate_epsilon()

    if substrate_appraisal.surprise_level >= eps:
        return "surprise_level"
    if substrate_appraisal.alignment_hints:
        return "alignment_hints"
    if substrate_appraisal.strain_shift_refs:
        return "strain_shift_refs"
    if substrate_appraisal.open_loop_pressure >= OPEN_LOOP_PRESSURE_MAX:
        return "open_loop_pressure"

    repair = thought.repair_pressure_level
    if repair is not None and repair >= REPAIR_PRESSURE_MAX:
        return "repair_pressure_level"

    trust = thought.trust_rupture_score
    if trust is not None and trust >= TRUST_RUPTURE_DEFER_THRESHOLD:
        return "trust_rupture_score"

    if thought.boundary_register:
        return "boundary_register"

    if repair_overlay.mode != "default":
        return "repair_overlay_mode"

    return None


def maybe_quick_lane_verdict(
    *,
    correlation_id: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    repair_overlay: HarnessRepairOverlayV1,
    epsilon: float | None = None,
) -> FinalizeReflectionV1 | None:
    if quick_lane_block_reason(
        substrate_appraisal=substrate_appraisal,
        thought=thought,
        repair_overlay=repair_overlay,
        epsilon=epsilon,
    ):
        return None

    return FinalizeReflectionV1(
        correlation_id=correlation_id,
        thought_event_id=thought.event_id,
        substrate_appraisal_id=substrate_appraisal.molecule_id,
        draft_hash=substrate_appraisal.draft_hash,
        imperative=thought.imperative,
        tone=thought.tone,
        strain_refs=list(thought.strain_refs),
        alignment_verdict="aligned",
        alignment_notes=[],
        strain_unresolved=False,
        reflection_source="deterministic_quick_gate",
        quick_lane_skipped_llm=True,
        finalize_changed=False,
    )


def build_finalize_reflect_context(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
) -> dict[str, Any]:
    return {
        "draft_text": draft_text,
        "thought_event": thought.model_dump(mode="json"),
        "substrate_appraisal": substrate_appraisal.model_dump(mode="json"),
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
        "tool_execution": format_tool_execution_digest(grammar_receipts),
        "repair_overlay": repair_overlay.model_dump(mode="json"),
        "finalize_overlay": repair_overlay.finalize_overlay,
        "user_message": user_message,
        # Route the 5b integrative reflection off the saturated `chat` worker onto the
        # background/metacog lane (unsaturated, long read timeout). This is an internal
        # reflection pass (not user-visible speech), so the lighter lane is acceptable.
        # allow_chat_fallback keeps resilience if the metacog route is ever absent.
        "llm_lane": "background",
        "allow_chat_fallback": True,
        "metadata": {
            "correlation_id": correlation_id,
            "mode": "brain",
        },
    }


def build_finalize_reflect_plan_request(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
) -> PlanExecutionRequest:
    plan = build_plan_for_verb("harness_finalize_reflect", mode="brain")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source="orion-harness-governor",
            extra={"llm_profile": "brain", "mode": "brain"},
        ),
        context=build_finalize_reflect_context(
            correlation_id=correlation_id,
            draft_text=draft_text,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            repair_overlay=repair_overlay,
            user_message=user_message,
            grammar_receipts=grammar_receipts,
        ),
    )


def parse_finalize_reflection_payload(raw: dict[str, Any] | str) -> FinalizeReflectionV1:
    if isinstance(raw, str):
        raw = parse_json_object(raw)
    return FinalizeReflectionV1.model_validate(raw)


def extract_finalize_reflection_payload(result: dict[str, Any]) -> dict[str, Any] | str:
    text = extract_cortex_payload_text(result)
    if text:
        return text

    detail = cortex_exec_failure_detail(result)
    if detail:
        raise ValueError(f"harness_finalize_reflect exec failed: {detail}")
    raise ValueError("harness_finalize_reflect exec result missing reflection payload")


async def run_finalize_reflection(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1 | None,
    repair_overlay: HarnessRepairOverlayV1 | None = None,
    user_message: str = "",
    grammar_receipts: list[GrammarReceiptV1] | None = None,
    cortex_client: CortexClientFn | None = None,
) -> tuple[FinalizeReflectionV1, bool, str | None]:
    if substrate_appraisal is None:
        raise ValueError("substrate_appraisal is required for harness finalize reflection")

    overlay = repair_overlay or HarnessRepairOverlayV1()
    quick = maybe_quick_lane_verdict(
        correlation_id=correlation_id,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        repair_overlay=overlay,
    )
    if quick is not None:
        return quick, True, None

    if cortex_client is None:
        raise ValueError("cortex_client is required when quick lane is blocked")

    plan_request = build_finalize_reflect_plan_request(
        correlation_id=correlation_id,
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        repair_overlay=overlay,
        user_message=user_message,
        grammar_receipts=grammar_receipts,
    )
    try:
        exec_result = await cortex_client(plan_request)
        raw_payload = extract_finalize_reflection_payload(exec_result)
        # extract_finalize_reflection_payload() returns model TEXT on the normal
        # path (extract_cortex_payload_text() is str-typed), so normalize to a
        # dict BEFORE injecting the four server-owned identity fields. The
        # injection below used to sit after this try block guarded by
        # `isinstance(raw_payload, dict)`, which is false for a str -- so the
        # fields were never injected on any real turn and FinalizeReflectionV1
        # validation failed with 4 missing required fields. Dead since
        # 40cf980bf (2026-07-05); only reachable when the quick lane is blocked,
        # which is why most turns never hit it.
        if isinstance(raw_payload, str):
            raw_payload = parse_json_object(raw_payload)
        if isinstance(raw_payload, dict):
            # Assignment, NOT setdefault: these four are server-owned identity.
            # setdefault let a model-supplied value win, which was harmless only
            # while this block was dead. _reflection_id()/_verdict_molecule_id()
            # hash the reflection's copies while the sibling fields on the same
            # outcome molecule come from the real appraisal object -- one
            # transcribed-wrong draft_hash and those two keys silently stop
            # correlating, with nothing visible at runtime.
            raw_payload["correlation_id"] = correlation_id
            raw_payload["thought_event_id"] = thought.event_id
            raw_payload["substrate_appraisal_id"] = substrate_appraisal.molecule_id
            raw_payload["draft_hash"] = substrate_appraisal.draft_hash
        reflection = parse_finalize_reflection_payload(raw_payload)
    except Exception as exc:
        degraded = maybe_quick_lane_verdict(
            correlation_id=correlation_id,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            repair_overlay=overlay,
        )
        if degraded is not None:
            logger.warning(
                "harness_finalize_reflect_llm_failed_using_quick_lane corr=%s err=%s",
                correlation_id,
                exc,
            )
            return degraded, True, None
        # Distinguish "the payload would not parse/validate" from "the gateway
        # call failed". Both degrade the same way, but only the first can be
        # caused by OUR bug (a FinalizeReflectionV1 field rename would degrade
        # every turn to misaligned instead of crashing loudly), so it must be
        # greppable on its own rather than hiding under an llm_failed label.
        unparseable = isinstance(exc, (ValidationError, ValueError))
        logger.warning(
            "%s corr=%s err=%s",
            (
                "harness_finalize_reflect_payload_unparseable_using_degraded_reflection"
                if unparseable
                else "harness_finalize_reflect_llm_failed_using_degraded_reflection"
            ),
            correlation_id,
            exc,
        )
        return (
            FinalizeReflectionV1(
                correlation_id=correlation_id,
                thought_event_id=thought.event_id,
                substrate_appraisal_id=substrate_appraisal.molecule_id,
                draft_hash=substrate_appraisal.draft_hash,
                imperative=thought.imperative,
                tone=thought.tone,
                strain_refs=list(thought.strain_refs),
                alignment_verdict="misaligned",
                alignment_notes=[f"reflect_llm_failed: {_excerpt(str(exc), max_len=200)}"],
                strain_unresolved=False,
                reflection_source="degraded_llm_failure_fallback",
                quick_lane_skipped_llm=False,
                finalize_changed=False,
            ),
            False,
            None,
        )
    cortex_trace_id = exec_result.get("trace_id") or exec_result.get("cortex_trace_id")
    if isinstance(cortex_trace_id, str):
        return reflection, False, cortex_trace_id
    return reflection, False, None


async def maybe_run_finalize_tool_retry(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None,
    cortex_client: CortexClientFn,
    bus: Any = None,
    grammar_channel: str = DEFAULT_GRAMMAR_EVENT_CHANNEL,
    grammar_publish_fn: Any = None,
) -> tuple[FinalizeReflectionV1, list[GrammarReceiptV1], bool, str | None, str | None]:
    """One bounded tool-recall retry (loop-back beat, "finalize 5b-prime").

    When 5b's reflection is misaligned specifically because a known,
    already-reviewed tool would fix it, call that tool once, fold its result
    into grammar_receipts, and re-run 5b so 5c's voice-finalize step (which
    already reads grammar_receipts/tool_execution) can incorporate it. The
    caller (run_harness_finalize_chain) loops this up to
    MAX_FINALIZE_LOOP_RETRIES times; within a single call, a tool already
    called this turn (motor stage or an earlier retry) is never called again,
    so a given tool cannot fire twice regardless of what a later reflection
    verdict says.

    Returns (reflection, grammar_receipts, retried, tool_name, cortex_trace_id).
    `retried` is True only when a tool call was actually attempted (whether or
    not it succeeded) -- the caller uses it, not a truthiness check on
    tool_name, to decide whether to record finalize_loop_retried and whether
    to keep looping. `cortex_trace_id` is the RETRY's own re-reflection trace
    id when one completed, else None -- the caller must not keep reusing the
    pre-retry trace id once a new reflection has actually replaced it, or the
    verdict molecule (durably persisted) points at the wrong LLM call.

    Fail-open by design, matching every other beat in this module. Before any
    tool call (flag off, no recommendation, unrecognized tool, already
    called) or if dispatching the tool itself fails, returns the ORIGINAL
    reflection/receipts unchanged -- EXCEPT that an unrecognized
    recommended_tool is scrubbed from the returned reflection (see below), not
    just skipped at the dispatch site. Once the tool has genuinely run,
    receipts always reflect that (real evidence, kept even if what follows
    fails) -- but run_finalize_reflection has its own internal fail-open (it
    never raises; an LLM/parse failure degrades to reflection_source=
    "degraded_llm_failure_fallback" internally), so the try/except around it
    below is defense-in-depth for a path that one already covers, not the
    normal way a re-reflection failure surfaces. Either way, a broken retry
    must never block or degrade a turn beyond what it would have been without
    this loop.
    """
    receipts = list(grammar_receipts or [])
    if not _finalize_tool_loop_enabled():
        return reflection, receipts, False, None, None
    if reflection.alignment_verdict != "misaligned":
        return reflection, receipts, False, None, None

    tool = str(reflection.recommended_tool or "").strip()
    if not tool:
        return reflection, receipts, False, None, None
    if tool not in FINALIZE_LOOP_TOOL_ALLOWLIST:
        logger.warning(
            "harness_finalize_tool_loop_unrecognized_tool corr=%s tool=%s",
            correlation_id,
            tool,
        )
        # Scrub, don't just skip: an unvetted name must not keep flowing into
        # the verdict molecule (published + persisted to harness_turn_trace)
        # or 5c's voice-finalize prompt context looking like a legitimate,
        # actionable recommendation -- the allowlist is enforced on the data,
        # not only at this one dispatch site.
        scrubbed = reflection.model_copy(
            update={"recommended_tool": None, "recommended_tool_reason": None}
        )
        return scrubbed, receipts, False, None, None
    if any(r.tool_name == tool for r in receipts):
        # Already called this turn -- do not call it again. Re-reflecting on
        # the same evidence a second time would not change the verdict, and
        # is exactly the runaway-retry shape the bound exists to prevent.
        logger.info(
            "harness_finalize_tool_loop_already_called corr=%s tool=%s",
            correlation_id,
            tool,
        )
        return reflection, receipts, False, None, None

    try:
        plan = build_plan_for_verb(tool, mode="brain")
        plan_request = PlanExecutionRequest(
            plan=plan,
            args=PlanExecutionArgs(request_id=correlation_id, trigger_source="orion-harness-governor"),
            context={"text": user_message, "metadata": {"correlation_id": correlation_id}},
        )
        exec_result = await cortex_client(plan_request)
        summary = extract_cortex_payload_text(exec_result) or "tool executed"
    except Exception as exc:
        logger.warning(
            "harness_finalize_tool_loop_dispatch_failed corr=%s tool=%s err=%s",
            correlation_id,
            tool,
            exc,
        )
        return reflection, receipts, False, None, None

    summary_excerpt = _excerpt(summary, max_len=500)
    try:
        from orion.harness.grammar_publish import publish_harness_step_grammar

        receipt = await publish_harness_step_grammar(
            bus,
            correlation_id=correlation_id,
            channel=grammar_channel,
            step_index=len(receipts),
            tool_name=tool,
            summary=summary_excerpt,
            publish_fn=grammar_publish_fn,
        )
    except Exception as exc:
        # Still usable for re-reflection even without a durable grammar event
        # -- an unpublished receipt (grammar_event_id=None) keeps the retry's
        # evidence visible to 5c's tool_execution digest without silently
        # dropping the retry. emit_turn_outcome_molecule already filters None
        # grammar_event_ids out of grammar_event_ids, so this degrades to
        # "evidence used, not durably recorded" rather than failing the retry.
        logger.warning(
            "harness_finalize_tool_loop_grammar_publish_failed corr=%s tool=%s err=%s",
            correlation_id,
            tool,
            exc,
        )
        receipt = GrammarReceiptV1(step_index=len(receipts), tool_name=tool, summary=summary_excerpt)
    receipts = receipts + [receipt]

    try:
        new_reflection, _quick_skipped, trace_id = await run_finalize_reflection(
            correlation_id=correlation_id,
            draft_text=draft_text,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            repair_overlay=repair_overlay,
            user_message=user_message,
            grammar_receipts=receipts,
            cortex_client=cortex_client,
        )
    except Exception as exc:
        logger.warning(
            "harness_finalize_tool_loop_re_reflect_failed corr=%s tool=%s err=%s",
            correlation_id,
            tool,
            exc,
        )
        return reflection, receipts, True, tool, None

    return new_reflection, receipts, True, tool, trace_id


async def emit_verdict_molecule(
    *,
    correlation_id: str,
    reflection: FinalizeReflectionV1,
    cortex_trace_id: str | None = None,
    channel: str = VERDICT_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessVerdictMoleculeV1:
    molecule = HarnessVerdictMoleculeV1(
        correlation_id=correlation_id,
        reflection=reflection,
        cortex_trace_id=cortex_trace_id,
    )
    if publish_fn is not None:
        await publish_fn(molecule, channel=channel)
        logger.info(
            "harness_verdict_published corr=%s channel=%s alignment=%s",
            correlation_id,
            channel,
            reflection.alignment_verdict,
        )
    elif bus is not None:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        envelope = BaseEnvelope(
            kind="harness.verdict.molecule.v1",
            source=ServiceRef(name="orion-harness-governor"),
            correlation_id=uuid.UUID(correlation_id) if _is_uuid(correlation_id) else uuid.uuid4(),
            payload=molecule.model_dump(mode="json"),
        )
        await bus.publish(channel, envelope)
        logger.info(
            "harness_verdict_published corr=%s channel=%s alignment=%s",
            correlation_id,
            channel,
            reflection.alignment_verdict,
        )
    return molecule


def build_voice_finalize_context(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    stance_harness_slice: StanceHarnessSliceV1,
    voice_contract: AnswerContract | dict[str, Any],
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
) -> dict[str, Any]:
    contract_dump = (
        voice_contract.model_dump(mode="json")
        if isinstance(voice_contract, AnswerContract)
        else dict(voice_contract)
    )
    return {
        "draft_text": draft_text,
        "thought_event": thought.model_dump(mode="json"),
        "substrate_appraisal": substrate_appraisal.model_dump(mode="json"),
        "reflection": reflection.model_dump(mode="json"),
        "stance_harness_slice": stance_harness_slice.model_dump(mode="json"),
        "grounding_capsule": (
            thought.grounding_capsule.model_dump(mode="json")
            if thought.grounding_capsule is not None
            else None
        ),
        "voice_contract": contract_dump,
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
        "tool_execution": format_tool_execution_digest(grammar_receipts),
        "finalize_overlay": repair_overlay.finalize_overlay,
        "user_message": user_message,
        "metadata": {
            "correlation_id": correlation_id,
            "mode": "brain",
        },
    }


def build_voice_finalize_plan_request(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    stance_harness_slice: StanceHarnessSliceV1,
    voice_contract: AnswerContract | dict[str, Any],
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
) -> PlanExecutionRequest:
    plan = build_plan_for_verb("orion_voice_finalize", mode="brain")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source="orion-harness-governor",
            extra={"llm_profile": "brain", "mode": "brain"},
        ),
        context=build_voice_finalize_context(
            correlation_id=correlation_id,
            draft_text=draft_text,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            reflection=reflection,
            stance_harness_slice=stance_harness_slice,
            voice_contract=voice_contract,
            repair_overlay=repair_overlay,
            user_message=user_message,
            grammar_receipts=grammar_receipts,
        ),
    )


def extract_voice_finalize_text(result: dict[str, Any]) -> str:
    text = extract_cortex_payload_text(result)
    if text:
        return text

    detail = cortex_exec_failure_detail(result)
    if detail:
        raise ValueError(f"orion_voice_finalize exec failed: {detail}")
    raise ValueError("orion_voice_finalize exec result missing final_text")


def _voice_finalize_changed(
    draft_text: str,
    final_text: str,
    reflection: FinalizeReflectionV1,
) -> bool:
    if reflection.alignment_verdict == "misaligned":
        return True
    return final_text.strip() != draft_text.strip()


async def run_orion_voice_finalize(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    voice_contract: AnswerContract | dict[str, Any] | None = None,
    repair_overlay: HarnessRepairOverlayV1 | None = None,
    user_message: str = "",
    grammar_receipts: list[GrammarReceiptV1] | None = None,
    cortex_client: CortexClientFn | None = None,
) -> tuple[str, dict[str, Any]]:
    """Orion capability: Orion's voiced final answer (beat 5c).

    Converts the motor draft into the user-visible text through the voice
    finalize Cortex plan (bus-mediated, not a local call), then records
    whether the result differs from the draft. This is the last stage allowed
    to change what Juniper reads.

    Runtime evidence: finalize_changed and alignment_verdict in the returned
    metadata, plus the cortex trace id. Start here when the final text
    diverged from the draft unexpectedly, or 5c failed after appraisal and
    reflection succeeded.
    """
    if cortex_client is None:
        raise ValueError("cortex_client is required for orion voice finalize")

    overlay = repair_overlay or HarnessRepairOverlayV1()
    contract = voice_contract or AnswerContract()
    plan_request = build_voice_finalize_plan_request(
        correlation_id=correlation_id,
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        reflection=reflection,
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=contract,
        repair_overlay=overlay,
        user_message=user_message,
        grammar_receipts=grammar_receipts,
    )
    exec_result = await cortex_client(plan_request)
    final_text = extract_voice_finalize_text(exec_result)
    finalize_changed = _voice_finalize_changed(draft_text, final_text, reflection)
    meta = {
        "finalize_changed": finalize_changed,
        "alignment_verdict": reflection.alignment_verdict,
        "cortex_trace_id": exec_result.get("trace_id") or exec_result.get("cortex_trace_id"),
    }
    return final_text, meta


def _reflection_id(reflection: FinalizeReflectionV1) -> str:
    return stable_hash_id(
        "reflection",
        [
            reflection.correlation_id,
            reflection.thought_event_id,
            reflection.substrate_appraisal_id,
            reflection.draft_hash,
            reflection.alignment_verdict,
        ],
    )


def _verdict_molecule_id(molecule: HarnessVerdictMoleculeV1) -> str:
    return stable_hash_id(
        "verdict",
        [
            molecule.correlation_id,
            molecule.reflection.thought_event_id,
            molecule.reflection.substrate_appraisal_id,
            molecule.reflection.draft_hash,
        ],
    )


async def emit_turn_outcome_molecule(
    *,
    correlation_id: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    verdict_molecule: HarnessVerdictMoleculeV1,
    draft_text: str,
    final_text: str,
    finalize_changed: bool,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
    finalize_failed: bool = False,
    failure_reason: str | None = None,
    finalize_loop_retried: bool = False,
    finalize_loop_tool: str | None = None,
    channel: str = OUTCOME_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessTurnOutcomeMoleculeV1:
    receipts = list(grammar_receipts or [])
    grammar_event_ids = [
        receipt.grammar_event_id for receipt in receipts if receipt.grammar_event_id
    ]
    surprise_resolved = (
        not finalize_failed
        and reflection.alignment_verdict == "aligned"
        and not reflection.strain_unresolved
        and (finalize_changed or substrate_appraisal.surprise_level < _quick_gate_epsilon())
    )
    molecule = HarnessTurnOutcomeMoleculeV1(
        correlation_id=correlation_id,
        thought_event_id=thought.event_id,
        substrate_appraisal_id=substrate_appraisal.molecule_id,
        reflection_id=_reflection_id(reflection),
        verdict_molecule_id=_verdict_molecule_id(verdict_molecule),
        draft_hash=substrate_appraisal.draft_hash,
        final_hash=_text_hash(final_text),
        finalize_changed=finalize_changed,
        alignment_verdict=reflection.alignment_verdict,
        surprise_level_at_draft=substrate_appraisal.surprise_level,
        surprise_resolved=surprise_resolved,
        grammar_event_ids=grammar_event_ids,
        final_text=final_text,
        finalize_failed=finalize_failed,
        failure_reason=failure_reason,
        draft_text_excerpt=_excerpt(draft_text) if finalize_failed else None,
        finalize_loop_retried=finalize_loop_retried,
        finalize_loop_tool=finalize_loop_tool,
    )
    if publish_fn is not None:
        await publish_fn(molecule, channel=channel)
        logger.info(
            "harness_turn_outcome_published corr=%s channel=%s surprise_resolved=%s finalize_failed=%s grammar_events=%d",
            correlation_id,
            channel,
            surprise_resolved,
            finalize_failed,
            len(grammar_event_ids),
        )
    elif bus is not None:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        envelope = BaseEnvelope(
            kind="harness.turn.outcome.v1",
            source=ServiceRef(name="orion-harness-governor"),
            correlation_id=uuid.UUID(correlation_id) if _is_uuid(correlation_id) else uuid.uuid4(),
            payload=molecule.model_dump(mode="json"),
        )
        await bus.publish(channel, envelope)
        logger.info(
            "harness_turn_outcome_published corr=%s channel=%s surprise_resolved=%s finalize_failed=%s grammar_events=%d",
            correlation_id,
            channel,
            surprise_resolved,
            finalize_failed,
            len(grammar_event_ids),
        )
    return molecule


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def _outcome_molecule_id(molecule: HarnessTurnOutcomeMoleculeV1) -> str:
    return stable_hash_id(
        "outcome",
        [
            molecule.correlation_id,
            molecule.thought_event_id,
            molecule.draft_hash,
            molecule.final_hash,
        ],
    )


async def run_substrate_finalize_appraisal(
    *,
    draft_molecule: HarnessDraftMoleculeV1,
    substrate_client: SubstrateClientFn,
) -> SubstrateFinalizeAppraisalV1:
    return await substrate_client(draft_molecule)


@dataclass
class HarnessFinalizeChainResult:
    final_text: str
    substrate_appraisal: SubstrateFinalizeAppraisalV1
    reflection: FinalizeReflectionV1
    verdict_molecule: HarnessVerdictMoleculeV1
    outcome_molecule: HarnessTurnOutcomeMoleculeV1
    finalize_changed: bool
    quick_lane_skipped_5b: bool
    verdict_molecule_id: str


@dataclass
class HarnessFinalizePartialState:
    substrate_appraisal: SubstrateFinalizeAppraisalV1
    reflection: FinalizeReflectionV1
    verdict_molecule: HarnessVerdictMoleculeV1
    outcome_molecule: HarnessTurnOutcomeMoleculeV1
    quick_lane_skipped_5b: bool
    verdict_molecule_id: str


class HarnessFinalizeFailedError(Exception):
    """Voice finalize (5c) failed after substrate appraisal + reflection completed."""

    def __init__(self, message: str, *, partial: HarnessFinalizePartialState) -> None:
        super().__init__(message)
        self.partial = partial


async def emit_harness_finalize_system_error(
    *,
    correlation_id: str,
    error: str,
    phase: str = "orion_voice_finalize",
    channel: str = SYSTEM_ERROR_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> None:
    payload = {
        "error": error,
        "phase": phase,
        "source": "orion-harness-governor",
        "correlation_id": correlation_id,
    }
    if publish_fn is not None:
        await publish_fn(payload, channel=channel)
        logger.info(
            "harness_finalize_system_error_published corr=%s channel=%s phase=%s",
            correlation_id,
            channel,
            phase,
        )
    elif bus is not None:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        envelope = BaseEnvelope(
            kind="system.error.v1",
            source=ServiceRef(name="orion-harness-governor"),
            correlation_id=uuid.UUID(correlation_id) if _is_uuid(correlation_id) else uuid.uuid4(),
            payload=payload,
        )
        await bus.publish(channel, envelope)
        logger.info(
            "harness_finalize_system_error_published corr=%s channel=%s phase=%s",
            correlation_id,
            channel,
            phase,
        )
    else:
        logger.warning(
            "harness_finalize_system_error_skipped corr=%s reason=no_bus_or_publish_fn phase=%s",
            correlation_id,
            phase,
        )


async def emit_finalize_failure_artifacts(
    *,
    correlation_id: str,
    error: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    reflection: FinalizeReflectionV1,
    verdict_molecule: HarnessVerdictMoleculeV1,
    verdict_molecule_id: str,
    grammar_receipts: list[GrammarReceiptV1] | None,
    user_message: str,
    quick_lane_skipped_5b: bool,
    bus: Any = None,
    outcome_publish_fn: PublishFn | None = None,
    closure_channel: str = POST_TURN_CLOSURE_CHANNEL,
    closure_publish_fn: PublishFn | None = None,
    system_error_channel: str = SYSTEM_ERROR_CHANNEL,
    system_error_publish_fn: PublishFn | None = None,
    finalize_loop_retried: bool = False,
    finalize_loop_tool: str | None = None,
) -> HarnessFinalizePartialState:
    """Emit degraded 6b outcome + 7 closure + homeostatic system error on 5c failure."""
    outcome_molecule = await emit_turn_outcome_molecule(
        correlation_id=correlation_id,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        reflection=reflection,
        verdict_molecule=verdict_molecule,
        draft_text=draft_text,
        final_text="",
        finalize_changed=False,
        grammar_receipts=grammar_receipts,
        finalize_failed=True,
        failure_reason=_excerpt(error, max_len=500),
        publish_fn=outcome_publish_fn,
        bus=bus,
        finalize_loop_retried=finalize_loop_retried,
        finalize_loop_tool=finalize_loop_tool,
    )
    closure = await emit_post_turn_closure(
        correlation_id=correlation_id,
        outcome_molecule=outcome_molecule,
        verdict_molecule_id=verdict_molecule_id,
        grammar_event_ids=outcome_molecule.grammar_event_ids,
        user_message=user_message,
        thought_event=thought,
        reflection_imperative=reflection.imperative,
        channel=closure_channel,
        publish_fn=closure_publish_fn,
        bus=bus,
    )
    logger.info(
        "harness_post_turn_closure_emitted corr=%s channel=%s surprise_unresolved=%s outcome_id=%s finalize_failed=true",
        correlation_id,
        closure_channel,
        closure.surprise_unresolved,
        closure.outcome_molecule_id,
    )
    await emit_harness_finalize_system_error(
        correlation_id=correlation_id,
        error=error,
        channel=system_error_channel,
        publish_fn=system_error_publish_fn,
        bus=bus,
    )
    return HarnessFinalizePartialState(
        substrate_appraisal=substrate_appraisal,
        reflection=reflection,
        verdict_molecule=verdict_molecule,
        outcome_molecule=outcome_molecule,
        quick_lane_skipped_5b=quick_lane_skipped_5b,
        verdict_molecule_id=verdict_molecule_id,
    )


async def run_harness_finalize_chain(
    *,
    correlation_id: str,
    draft_text: str,
    draft_molecule: HarnessDraftMoleculeV1,
    thought: ThoughtEventV1,
    grammar_receipts: list[GrammarReceiptV1] | None,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    voice_contract: AnswerContract | dict[str, Any] | None,
    cortex_client: CortexClientFn,
    substrate_client: SubstrateClientFn,
    bus: Any = None,
    verdict_publish_fn: PublishFn | None = None,
    outcome_publish_fn: PublishFn | None = None,
    embodiment_relational: bool = False,
    embodiment_interlocutor_ref: str | None = None,
    embodiment_publish_fn: PublishFn | None = None,
    closure_channel: str = POST_TURN_CLOSURE_CHANNEL,
    closure_publish_fn: PublishFn | None = None,
    system_error_channel: str = SYSTEM_ERROR_CHANNEL,
    system_error_publish_fn: PublishFn | None = None,
    grammar_channel: str = DEFAULT_GRAMMAR_EVENT_CHANNEL,
    grammar_publish_fn: Any = None,
) -> HarnessFinalizeChainResult:
    """Orion capability: unified-turn draft-to-voice finalization.

    Orchestrates finalize beats 5a → 5b → 5b-prime → 5c → 6b: substrate
    appraisal (5a), integrative reflection or its deterministic quick lane
    (5b), an optional bounded tool-recall retry when 5b is misaligned for a
    fixable reason (5b-prime, see maybe_run_finalize_tool_retry -- default-off
    via HARNESS_FINALIZE_TOOL_LOOP_ENABLED), verdict emission, Orion voice
    finalization (5c), turn-outcome emission (6b), and the optional embodiment
    intent. The motor draft is not the final Hub response; this chain is what
    may change it.

    Runtime evidence: substrate appraisal, verdict and outcome molecules
    (outcome carries finalize_loop_retried/finalize_loop_tool when 5b-prime
    fired), finalize_changed in voice metadata, post-turn closure, and failure
    artifacts when a beat raises. Start here when the motor completed but Hub
    received no final text or a finalize-phase error.
    """
    substrate_appraisal = await run_substrate_finalize_appraisal(
        draft_molecule=draft_molecule,
        substrate_client=substrate_client,
    )

    reflection, quick_lane_skipped_5b, cortex_trace_id = await run_finalize_reflection(
        correlation_id=correlation_id,
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        repair_overlay=repair_overlay,
        user_message=user_message,
        grammar_receipts=grammar_receipts,
        cortex_client=cortex_client,
    )

    # Loop-back tool-recall retry (5b-prime). MAX_FINALIZE_LOOP_RETRIES bounds
    # how many times this can fire per turn -- currently 1, so today this is
    # "one bounded retry" in practice, but the bound is real (a for-loop, not
    # just documentation) so a later change to that constant (e.g. Orion's own
    # autonomy/drive-economy system throttling it up) actually takes effect
    # without touching this function. Each iteration can only call a tool not
    # already in grammar_receipts (enforced inside maybe_run_finalize_tool_retry),
    # so the same tool is never dispatched twice regardless of the loop bound.
    finalize_loop_retried = False
    finalize_loop_tool: str | None = None
    for _attempt in range(MAX_FINALIZE_LOOP_RETRIES):
        reflection, grammar_receipts, attempt_retried, attempt_tool, attempt_trace_id = (
            await maybe_run_finalize_tool_retry(
                correlation_id=correlation_id,
                draft_text=draft_text,
                thought=thought,
                substrate_appraisal=substrate_appraisal,
                reflection=reflection,
                repair_overlay=repair_overlay,
                user_message=user_message,
                grammar_receipts=grammar_receipts,
                cortex_client=cortex_client,
                bus=bus,
                grammar_channel=grammar_channel,
                grammar_publish_fn=grammar_publish_fn,
            )
        )
        if not attempt_retried:
            break
        finalize_loop_retried = True
        finalize_loop_tool = attempt_tool
        if attempt_trace_id:
            # Only overwrite once a NEW reflection genuinely replaced the old
            # one -- a failed re-reflection (attempt_trace_id=None) must not
            # make the verdict molecule point at the wrong (pre-retry) trace.
            cortex_trace_id = attempt_trace_id
        if reflection.alignment_verdict != "misaligned":
            break

    verdict_molecule = await emit_verdict_molecule(
        correlation_id=correlation_id,
        reflection=reflection,
        cortex_trace_id=cortex_trace_id,
        publish_fn=verdict_publish_fn,
        bus=bus,
    )
    verdict_molecule_id = _verdict_molecule_id(verdict_molecule)

    try:
        final_text, voice_meta = await run_orion_voice_finalize(
            correlation_id=correlation_id,
            draft_text=draft_text,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            reflection=reflection,
            voice_contract=voice_contract,
            repair_overlay=repair_overlay,
            user_message=user_message,
            grammar_receipts=grammar_receipts,
            cortex_client=cortex_client,
        )
    except Exception as exc:
        partial = await emit_finalize_failure_artifacts(
            correlation_id=correlation_id,
            error=str(exc),
            draft_text=draft_text,
            thought=thought,
            substrate_appraisal=substrate_appraisal,
            reflection=reflection,
            verdict_molecule=verdict_molecule,
            verdict_molecule_id=verdict_molecule_id,
            grammar_receipts=grammar_receipts,
            user_message=user_message,
            quick_lane_skipped_5b=quick_lane_skipped_5b,
            bus=bus,
            outcome_publish_fn=outcome_publish_fn,
            closure_channel=closure_channel,
            closure_publish_fn=closure_publish_fn,
            system_error_channel=system_error_channel,
            system_error_publish_fn=system_error_publish_fn,
            finalize_loop_retried=finalize_loop_retried,
            finalize_loop_tool=finalize_loop_tool,
        )
        raise HarnessFinalizeFailedError(str(exc), partial=partial) from exc
    finalize_changed = bool(voice_meta.get("finalize_changed"))

    outcome_molecule = await emit_turn_outcome_molecule(
        correlation_id=correlation_id,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        reflection=reflection,
        verdict_molecule=verdict_molecule,
        draft_text=draft_text,
        final_text=final_text,
        finalize_changed=finalize_changed,
        grammar_receipts=grammar_receipts,
        publish_fn=outcome_publish_fn,
        bus=bus,
        finalize_loop_retried=finalize_loop_retried,
        finalize_loop_tool=finalize_loop_tool,
    )

    # (D) deliberate embodiment intent on the turn correlation_id — gated + fail-open.
    await emit_finalize_embodiment_intent(
        correlation_id=correlation_id,
        interlocutor_ref=embodiment_interlocutor_ref,
        relational=embodiment_relational,
        publish_fn=embodiment_publish_fn,
        bus=bus,
    )

    return HarnessFinalizeChainResult(
        final_text=final_text,
        substrate_appraisal=substrate_appraisal,
        reflection=reflection,
        verdict_molecule=verdict_molecule,
        outcome_molecule=outcome_molecule,
        finalize_changed=finalize_changed,
        quick_lane_skipped_5b=quick_lane_skipped_5b,
        verdict_molecule_id=verdict_molecule_id,
    )


def _excerpt(text: str, max_len: int = 300) -> str:
    t = (text or "").strip()
    return t[:max_len] if len(t) > max_len else t


async def emit_post_turn_closure(
    *,
    correlation_id: str,
    outcome_molecule: HarnessTurnOutcomeMoleculeV1,
    verdict_molecule_id: str,
    grammar_event_ids: list[str] | None = None,
    user_message: str = "",
    thought_event: ThoughtEventV1 | None = None,
    reflection_imperative: str = "",
    channel: str = POST_TURN_CLOSURE_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessPostTurnClosureV1:
    surprise_unresolved = not outcome_molecule.surprise_resolved
    referent_fields: dict[str, Any] = {}
    if surprise_unresolved:
        referent_fields = {
            "user_message_excerpt": _excerpt(user_message),
            "stance_imperative": _excerpt(
                reflection_imperative or (thought_event.imperative if thought_event else "")
            ),
            "thought_event_id": thought_event.event_id if thought_event else None,
        }
    closure = HarnessPostTurnClosureV1(
        correlation_id=correlation_id,
        outcome_molecule_id=_outcome_molecule_id(outcome_molecule),
        verdict_molecule_id=verdict_molecule_id,
        grammar_event_ids=list(grammar_event_ids or outcome_molecule.grammar_event_ids),
        surprise_unresolved=surprise_unresolved,
        **referent_fields,
    )
    if publish_fn is not None:
        await publish_fn(closure, channel=channel)
        logger.info(
            "harness_post_turn_closure_published corr=%s channel=%s surprise_unresolved=%s grammar_events=%d",
            correlation_id,
            channel,
            closure.surprise_unresolved,
            len(closure.grammar_event_ids),
        )
    elif bus is not None:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        envelope = BaseEnvelope(
            kind="harness.post_turn.closure.v1",
            source=ServiceRef(name="orion-harness-governor"),
            correlation_id=uuid.UUID(correlation_id) if _is_uuid(correlation_id) else uuid.uuid4(),
            payload=closure.model_dump(mode="json"),
        )
        await bus.publish(channel, envelope)
        logger.info(
            "harness_post_turn_closure_published corr=%s channel=%s surprise_unresolved=%s grammar_events=%d",
            correlation_id,
            channel,
            closure.surprise_unresolved,
            len(closure.grammar_event_ids),
        )
    return closure
