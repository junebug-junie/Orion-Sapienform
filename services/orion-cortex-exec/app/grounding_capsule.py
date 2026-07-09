from __future__ import annotations

import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.thought import GroundingCapsuleV1
from orion.thought.json_extract import extract_first_json_object_text

from .pcr_chat_memory import pcr_phase01_complete, run_pcr_phase0_and_1, run_pcr_phase3
from .settings import Settings, settings

logger = logging.getLogger("orion.cortex.grounding_capsule")


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if x]


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def stance_slice_brief_from_step_text(text: str) -> Dict[str, Any]:
    """Minimal stance brief (task_mode/conversation_frame) parsed from stance JSON.

    Feeds run_pcr_phase3's retrieval-intent derivation, which reads
    ctx['chat_stance_brief']. Tolerant of markdown wrappers and non-JSON.
    """
    import json

    raw = (text or "").strip()
    if not raw:
        return {}
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        blob = extract_first_json_object_text(raw)
        if blob:
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return {}
    sl = parsed.get("stance_harness_slice")
    if not isinstance(sl, dict):
        return {}
    return {
        "task_mode": str(sl.get("task_mode") or "").strip(),
        "conversation_frame": str(sl.get("conversation_frame") or "").strip(),
    }


def build_grounding_capsule(ctx: Dict[str, Any], *, pcr_ran: bool) -> GroundingCapsuleV1:
    """Assemble the capsule from identity summaries + PCR digests already in ctx."""
    return GroundingCapsuleV1(
        identity_summary=_str_list(ctx.get("orion_identity_summary")),
        relationship_summary=_str_list(ctx.get("juniper_relationship_summary")),
        response_policy_summary=_str_list(ctx.get("response_policy_summary")),
        continuity_digest=_clean(ctx.get("continuity_digest")),
        belief_digest=_clean(ctx.get("belief_digest")),
        memory_digest=_clean(ctx.get("memory_digest")),
        provenance={
            "identity_source": str(ctx.get("identity_kernel_source") or "unknown"),
            "pcr_ran": bool(pcr_ran),
        },
    )


async def assemble_stance_grounding(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    stance_step_text: str,
    exec_settings: Settings | None = None,
) -> GroundingCapsuleV1 | None:
    """Run PCR phase-3 for the unified turn, then assemble the grounding capsule.

    Returns None when the flag is off. On PCR failure the capsule still ships
    with identity only (graceful degradation) — the turn never blocks on recall.
    """
    cfg = exec_settings or settings
    if not cfg.orion_unified_grounding_enabled:
        return None

    pcr_ran = False
    try:
        ctx["chat_stance_brief"] = stance_slice_brief_from_step_text(stance_step_text)
        if cfg.chat_pcr_enabled:
            if not pcr_phase01_complete(ctx):
                await run_pcr_phase0_and_1(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=recall_cfg,
                    exec_settings=cfg,
                )
            _pcr, _step, _debug = await run_pcr_phase3(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                exec_settings=cfg,
            )
            pcr_ran = True
    except Exception:
        logger.warning(
            "unified_grounding_pcr_failed corr=%s (shipping identity-only capsule)",
            correlation_id,
            exc_info=True,
        )
        pcr_ran = False

    capsule = build_grounding_capsule(ctx, pcr_ran=pcr_ran)
    logger.info(
        "unified_grounding_capsule_ready corr=%s identity=%s relationship=%s policy=%s pcr_ran=%s memory_chars=%s",
        correlation_id,
        len(capsule.identity_summary),
        len(capsule.relationship_summary),
        len(capsule.response_policy_summary),
        pcr_ran,
        len(capsule.memory_digest or ""),
    )
    return capsule
