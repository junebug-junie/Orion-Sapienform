from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.context_provenance import CONTEXT_PROVENANCE_REGISTRY
from orion.memory.pages.anchor import CTX_KEY_ANCHOR as CORE_PAGE_ANCHOR_CTX_KEY, anchor_page_from_payload
from orion.memory.pages.juniper import CTX_KEY_JUNIPER as CORE_PAGE_JUNIPER_CTX_KEY, core_page_from_payload
from orion.memory.pages.origin import CTX_KEY_ORIGIN as CORE_PAGE_ORIGIN_CTX_KEY
from orion.memory.pages.origin import core_page_from_payload as origin_page_from_payload
from orion.schemas.thought import CorePageV1, GroundingCapsuleV1
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


def _has_content(value: Any) -> bool:
    """True unless the value carries no evidence at all (None, or an empty
    dict/list/tuple/set/str). A key present in ctx only via a fallback like
    ``ctx["biometrics"] = biometrics_result or {}`` degrades to an empty
    container when the live computation didn't run or failed -- classifying
    that as "live_runtime_projection" would be exactly the false-liveness
    claim this registry exists to prevent. Falsy scalars (0, 0.0, False) are
    deliberately NOT excluded here -- e.g. a real eventfulness score of 0.0 is
    a genuine live result, not evidence of absence.
    """
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str)) and len(value) == 0:
        return False
    return True


def context_provenance_for_ctx(ctx: Dict[str, Any]) -> Dict[str, str]:
    """Map every ctx key present this turn (with actual content) to its
    registered source kind.

    Keys with no registry entry (unclassified, not plumbing-exempt) are
    omitted rather than guessed — see CONTEXT_PROVENANCE_REGISTRY's coverage
    test for the gate that catches new ctx keys shipping unclassified.

    Cached on ctx itself: this and the chat_stance.py provenance hazard both
    scan the full registry against ctx once per turn on the live default
    path (ORION_UNIFIED_GROUNDING_ENABLED=True) -- caching avoids doing that
    scan twice for the same ctx.
    """
    cached = ctx.get("_context_provenance_cache")
    if isinstance(cached, dict):
        return cached
    computed = {
        key: entry.source_kind
        for key, entry in CONTEXT_PROVENANCE_REGISTRY.items()
        if key in ctx and _has_content(ctx.get(key))
    }
    ctx["_context_provenance_cache"] = computed
    return computed


def core_pages_for_ctx(ctx: Dict[str, Any], *, enabled: bool) -> list[CorePageV1]:
    """The core pages ctx carries, anchor first: Juniper's fixed page:anchor lines, Orion's
    page:juniper, then Orion's page:origin (issue #3462). Any may be missing. Empty when the flag is
    off or no row was hydrated, so the prefix falls back to the identity-yaml bullets."""
    if not enabled:
        return []
    pages = [
        anchor_page_from_payload(ctx.get(CORE_PAGE_ANCHOR_CTX_KEY)),
        core_page_from_payload(ctx.get(CORE_PAGE_JUNIPER_CTX_KEY)),
        origin_page_from_payload(ctx.get(CORE_PAGE_ORIGIN_CTX_KEY)),
    ]
    return [p for p in pages if p is not None]


_CORE_PAGE_CTX_KEYS = (CORE_PAGE_ANCHOR_CTX_KEY, CORE_PAGE_JUNIPER_CTX_KEY, CORE_PAGE_ORIGIN_CTX_KEY)


def hydrate_core_pages(ctx: Dict[str, Any]) -> None:
    """Pull ONLY the core-page felt-state lanes (page:anchor, page:juniper, page:origin; cached 5 min;
    fail-open, never raises). Lanes already in ctx are not read again."""
    missing = tuple(k for k in _CORE_PAGE_CTX_KEYS if ctx.get(k) is None)
    if not missing:
        return
    from app.substrate_felt_state_reader import hydrate_felt_state_ctx

    hydrate_felt_state_ctx(ctx, lanes=missing)
    if any(ctx.get(k) is not None for k in missing):
        # The provenance map may already be cached from earlier in the turn; recompute so the
        # pages' keys are listed in CONTEXT PROVENANCE.
        ctx.pop("_context_provenance_cache", None)


def build_grounding_capsule(
    ctx: Dict[str, Any], *, pcr_ran: bool, core_pages_enabled: bool = False
) -> GroundingCapsuleV1:
    """Assemble the capsule from identity summaries + PCR digests already in ctx."""
    return GroundingCapsuleV1(
        core_pages=core_pages_for_ctx(ctx, enabled=core_pages_enabled),
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
        context_provenance=context_provenance_for_ctx(ctx),
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

    core_pages_on = bool(getattr(cfg, "orion_core_pages_enabled", False))
    if core_pages_on:
        try:
            await asyncio.to_thread(hydrate_core_pages, ctx)
        except Exception:  # noqa: BLE001 -- a page is never worth a failed turn
            logger.warning("core_page_hydrate_failed corr=%s", correlation_id, exc_info=True)
    capsule = build_grounding_capsule(ctx, pcr_ran=pcr_ran, core_pages_enabled=core_pages_on)
    logger.info(
        "unified_grounding_capsule_ready corr=%s identity=%s relationship=%s policy=%s pcr_ran=%s memory_chars=%s "
        "core_pages=%s",
        correlation_id,
        len(capsule.identity_summary),
        len(capsule.relationship_summary),
        len(capsule.response_policy_summary),
        pcr_ran,
        len(capsule.memory_digest or ""),
        ",".join(f"{p.concept_id}@v{p.version}" for p in capsule.core_pages) or "-",
    )
    return capsule
