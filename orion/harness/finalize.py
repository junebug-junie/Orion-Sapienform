from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from orion.cognition.plan_loader import build_plan_for_verb
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

VERDICT_CHANNEL = "orion:harness:verdict:artifact"
OUTCOME_CHANNEL = "orion:substrate:turn_outcome"
POST_TURN_CLOSURE_CHANNEL = "orion:substrate:post_turn_closure"


def _quick_gate_epsilon() -> float:
    raw = os.environ.get("FINALIZE_QUICK_GATE_EPSILON", str(DEFAULT_QUICK_GATE_EPSILON))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_QUICK_GATE_EPSILON


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
        "repair_overlay": repair_overlay.model_dump(mode="json"),
        "finalize_overlay": repair_overlay.finalize_overlay,
        "user_message": user_message,
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
        raw = json.loads(raw)
    return FinalizeReflectionV1.model_validate(raw)


def extract_finalize_reflection_payload(result: dict[str, Any]) -> dict[str, Any] | str:
    final_text = result.get("final_text")
    if isinstance(final_text, str) and final_text.strip():
        return final_text

    steps = result.get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_result = step.get("result")
        if not isinstance(step_result, dict):
            continue
        for key in ("structured", "json", "payload", "final_text", "text", "content"):
            value = step_result.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value

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
    exec_result = await cortex_client(plan_request)
    raw_payload = extract_finalize_reflection_payload(exec_result)
    if isinstance(raw_payload, dict):
        raw_payload.setdefault("correlation_id", correlation_id)
        raw_payload.setdefault("thought_event_id", thought.event_id)
        raw_payload.setdefault("substrate_appraisal_id", substrate_appraisal.molecule_id)
        raw_payload.setdefault("draft_hash", substrate_appraisal.draft_hash)
    reflection = parse_finalize_reflection_payload(raw_payload)
    cortex_trace_id = exec_result.get("trace_id") or exec_result.get("cortex_trace_id")
    if isinstance(cortex_trace_id, str):
        return reflection, False, cortex_trace_id
    return reflection, False, None


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
        "voice_contract": contract_dump,
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
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
    final_text = result.get("final_text")
    if isinstance(final_text, str) and final_text.strip():
        return final_text.strip()

    steps = result.get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_result = step.get("result")
        if not isinstance(step_result, dict):
            continue
        for key in ("final_text", "text", "content"):
            value = step_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

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
    channel: str = OUTCOME_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessTurnOutcomeMoleculeV1:
    receipts = list(grammar_receipts or [])
    grammar_event_ids = [
        receipt.grammar_event_id for receipt in receipts if receipt.grammar_event_id
    ]
    surprise_resolved = (
        reflection.alignment_verdict == "aligned"
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
    )
    if publish_fn is not None:
        await publish_fn(molecule, channel=channel)
        logger.info(
            "harness_turn_outcome_published corr=%s channel=%s surprise_resolved=%s grammar_events=%d",
            correlation_id,
            channel,
            surprise_resolved,
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
            "harness_turn_outcome_published corr=%s channel=%s surprise_resolved=%s grammar_events=%d",
            correlation_id,
            channel,
            surprise_resolved,
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
) -> HarnessFinalizeChainResult:
    """Orchestrate finalize beats 5a → 5b → 5c → 6b."""
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

    verdict_molecule = await emit_verdict_molecule(
        correlation_id=correlation_id,
        reflection=reflection,
        cortex_trace_id=cortex_trace_id,
        publish_fn=verdict_publish_fn,
        bus=bus,
    )
    verdict_molecule_id = _verdict_molecule_id(verdict_molecule)

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


async def emit_post_turn_closure(
    *,
    correlation_id: str,
    outcome_molecule: HarnessTurnOutcomeMoleculeV1,
    verdict_molecule_id: str,
    grammar_event_ids: list[str] | None = None,
    channel: str = POST_TURN_CLOSURE_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessPostTurnClosureV1:
    closure = HarnessPostTurnClosureV1(
        correlation_id=correlation_id,
        outcome_molecule_id=_outcome_molecule_id(outcome_molecule),
        verdict_molecule_id=verdict_molecule_id,
        grammar_event_ids=list(grammar_event_ids or outcome_molecule.grammar_event_ids),
        surprise_unresolved=not outcome_molecule.surprise_resolved,
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
