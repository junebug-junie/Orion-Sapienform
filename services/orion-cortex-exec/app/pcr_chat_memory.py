from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.memory.recall_skip_gate import RecallSkipGateResult, recall_skip_gate
from orion.memory.retrieval_intent import derive_retrieval_intent
from orion.schemas.cortex.schemas import StepExecutionResult
from orion.schemas.recall_pcr import PcrChatMemoryV1

from .executor import _last_user_message, run_recall_step
from .recall_utils import hub_chat_lane_from_ctx
from .settings import Settings, settings

logger = logging.getLogger("orion.cortex.pcr_chat_memory")

CONTINUITY_PROFILE = "chat.continuity.v1"
PROFILE_FOR_INTENT: dict[str, str] = {
    "relational": "chat.belief.relational.v1",
    "semantic": "chat.belief.semantic.v1",
    "procedural": "chat.belief.procedural.v1",
    "open_loop": "chat.belief.open_loop.v1",
    "contradiction": "chat.belief.contradiction.v1",
}


def _extract_turn_change_appraisal(ctx: Dict[str, Any]) -> dict[str, Any] | None:
    direct = ctx.get("turn_change_appraisal")
    if isinstance(direct, dict):
        return direct
    for container_key in ("spark_meta", "metadata"):
        container = ctx.get(container_key)
        if not isinstance(container, dict):
            continue
        appraisal = container.get("turn_change_appraisal")
        if isinstance(appraisal, dict):
            return appraisal
        nested_spark = container.get("spark_meta")
        if isinstance(nested_spark, dict):
            nested_appraisal = nested_spark.get("turn_change_appraisal")
            if isinstance(nested_appraisal, dict):
                return nested_appraisal
    return None


def _has_repair_grammar_signal(ctx: Dict[str, Any]) -> bool:
    if ctx.get("has_repair_grammar_signal") is True:
        return True
    grammar_events = ctx.get("grammar_events")
    if isinstance(grammar_events, list):
        for event in grammar_events:
            if not isinstance(event, dict):
                continue
            atom = event.get("atom")
            if isinstance(atom, dict) and atom.get("semantic_role") == "repair_signal":
                return True
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    if metadata.get("substrate_effect_summary") or metadata.get("repair_pressure_contract"):
        return True
    return False


def _apply_pcr_to_ctx(ctx: Dict[str, Any], pcr: PcrChatMemoryV1) -> None:
    ctx["pcr_memory"] = pcr
    ctx["continuity_digest"] = pcr.continuity_digest
    ctx["belief_digest"] = pcr.belief_digest
    ctx["memory_digest"] = pcr.memory_digest
    debug = ctx.setdefault("debug", {})
    if isinstance(debug, dict):
        debug["pcr"] = {
            "phase": pcr.phase,
            "retrieval_intent": pcr.retrieval_intent,
            "skip_reasons": list(pcr.skip_reasons),
            "continuity_digest_chars": len(pcr.continuity_digest or ""),
            "belief_digest_chars": len(pcr.belief_digest or ""),
            "recall_debug": dict(pcr.recall_debug),
        }


def _pcr_from_ctx(ctx: Dict[str, Any]) -> PcrChatMemoryV1 | None:
    raw = ctx.get("pcr_memory")
    if isinstance(raw, PcrChatMemoryV1):
        return raw
    return None


def _skip_gate_from_ctx(ctx: Dict[str, Any]) -> RecallSkipGateResult:
    pcr = _pcr_from_ctx(ctx)
    if pcr is not None and pcr.phase == "skip":
        return RecallSkipGateResult(skip=True, reasons=list(pcr.skip_reasons))
    return RecallSkipGateResult(skip=False, reasons=[])


def _attention_frame_from_ctx(ctx: Dict[str, Any]) -> dict[str, Any] | None:
    for key in ("chat_attention_frame", "attention_frame"):
        frame = ctx.get(key)
        if isinstance(frame, dict):
            return frame
    return None


def _stance_brief_from_ctx(ctx: Dict[str, Any]) -> dict[str, Any]:
    brief = ctx.get("chat_stance_brief")
    return brief if isinstance(brief, dict) else {}


def _merge_memory_digest(continuity_digest: str, belief_digest: str) -> str:
    parts = [part.strip() for part in (continuity_digest, belief_digest) if part and part.strip()]
    return "\n\n".join(parts)


def _empty_pcr(*, phase: str, retrieval_intent: str | None, skip_reasons: list[str]) -> PcrChatMemoryV1:
    return PcrChatMemoryV1(
        phase=phase,  # type: ignore[arg-type]
        retrieval_intent=retrieval_intent,  # type: ignore[arg-type]
        continuity_digest="",
        belief_digest="",
        memory_digest="",
        skip_reasons=list(skip_reasons),
        recall_debug={},
    )


async def run_pcr_phase0_and_1(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    exec_settings: Settings | None = None,
) -> Tuple[PcrChatMemoryV1, StepExecutionResult | None, Dict[str, Any]]:
    """Phase 0 skip gate + optional phase 1 continuity recall for chat_general."""
    cfg = exec_settings or settings
    user_message = _last_user_message(ctx) or str(ctx.get("user_message") or "")

    if cfg.chat_pcr_skip_on_low_info:
        gate = recall_skip_gate(
            user_message=user_message,
            appraisal=_extract_turn_change_appraisal(ctx),
            has_repair_grammar_signal=_has_repair_grammar_signal(ctx),
        )
    else:
        from orion.memory.recall_skip_gate import RecallSkipGateResult

        gate = RecallSkipGateResult(skip=False, reasons=[])

    if gate.skip:
        pcr = _empty_pcr(phase="skip", retrieval_intent="none", skip_reasons=gate.reasons)
        _apply_pcr_to_ctx(ctx, pcr)
        logger.info(
            "pcr_phase0_skip corr_id=%s reasons=%s",
            correlation_id,
            gate.reasons,
        )
        return pcr, None, {"pcr_phase": "skip", "skip_reasons": gate.reasons}

    recall_step, recall_debug, continuity_digest = await run_recall_step(
        bus,
        source=source,
        ctx=ctx,
        correlation_id=correlation_id,
        recall_cfg=recall_cfg,
        recall_profile=CONTINUITY_PROFILE,
        step_name="pcr_continuity_recall",
        step_order=-1,
        recall_phase="continuity",
        retrieval_intent="continuity",
    )
    continuity_text = (continuity_digest or "").strip()
    pcr = PcrChatMemoryV1(
        phase="continuity",
        retrieval_intent="continuity",
        continuity_digest=continuity_text,
        belief_digest="",
        memory_digest=continuity_text,
        skip_reasons=[],
        recall_debug=dict(recall_debug) if isinstance(recall_debug, dict) else {},
    )
    _apply_pcr_to_ctx(ctx, pcr)
    logger.info(
        "pcr_phase1_continuity corr_id=%s profile=%s items=%s digest_chars=%s",
        correlation_id,
        CONTINUITY_PROFILE,
        recall_debug.get("count") if isinstance(recall_debug, dict) else 0,
        len(continuity_text),
    )
    return pcr, recall_step, recall_debug if isinstance(recall_debug, dict) else {}


async def run_pcr_phase3(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    exec_settings: Settings | None = None,
) -> Tuple[PcrChatMemoryV1, StepExecutionResult | None, Dict[str, Any]]:
    """Phase 3 purposeful recall after stance synthesis for chat_general."""
    cfg = exec_settings or settings
    prior = _pcr_from_ctx(ctx)
    continuity_text = (ctx.get("continuity_digest") or (prior.continuity_digest if prior else "") or "").strip()
    skip_gate = _skip_gate_from_ctx(ctx)
    stance_brief = _stance_brief_from_ctx(ctx)
    attention_frame = _attention_frame_from_ctx(ctx)
    user_message = _last_user_message(ctx) or str(ctx.get("user_message") or "")

    intent, rule_id = derive_retrieval_intent(
        skip_gate=skip_gate,
        stance_brief=stance_brief,
        attention_frame=attention_frame,
        appraisal=_extract_turn_change_appraisal(ctx),
        hub_chat_lane=hub_chat_lane_from_ctx(ctx),
        user_message=user_message,
    )

    if intent in ("none", "continuity"):
        phase = "skip" if intent == "none" else "continuity"
        pcr = PcrChatMemoryV1(
            phase=phase,  # type: ignore[arg-type]
            retrieval_intent=intent,  # type: ignore[arg-type]
            continuity_digest=continuity_text,
            belief_digest="",
            memory_digest=continuity_text,
            skip_reasons=list(prior.skip_reasons) if prior and prior.phase == "skip" else [],
            recall_debug={"pcr_phase": "phase3_skipped", "retrieval_intent": intent, "rule_id": rule_id},
        )
        _apply_pcr_to_ctx(ctx, pcr)
        logger.info(
            "pcr_phase3_skip corr_id=%s intent=%s rule_id=%s",
            correlation_id,
            intent,
            rule_id,
        )
        return pcr, None, {"pcr_phase": "phase3_skipped", "retrieval_intent": intent, "rule_id": rule_id}

    profile = PROFILE_FOR_INTENT[str(intent)]
    task_mode = str(stance_brief.get("task_mode") or "").strip()
    recall_step, recall_debug, belief_digest = await run_recall_step(
        bus,
        source=source,
        ctx=ctx,
        correlation_id=correlation_id,
        recall_cfg=recall_cfg,
        recall_profile=profile,
        step_name="pcr_belief_recall",
        step_order=-1,
        recall_phase="purposeful",
        retrieval_intent=str(intent),
        task_hints={"task_mode": task_mode, "rule_id": rule_id},
    )
    belief_text = (belief_digest or "").strip()
    memory_digest = _merge_memory_digest(continuity_text, belief_text)
    pcr = PcrChatMemoryV1(
        phase="purposeful",
        retrieval_intent=intent,  # type: ignore[arg-type]
        continuity_digest=continuity_text,
        belief_digest=belief_text,
        memory_digest=memory_digest,
        skip_reasons=[],
        recall_debug=dict(recall_debug) if isinstance(recall_debug, dict) else {},
    )
    _apply_pcr_to_ctx(ctx, pcr)
    logger.info(
        "pcr_phase3_purposeful corr_id=%s intent=%s profile=%s rule_id=%s digest_chars=%s",
        correlation_id,
        intent,
        profile,
        rule_id,
        len(belief_text),
    )
    return pcr, recall_step, recall_debug if isinstance(recall_debug, dict) else {}
