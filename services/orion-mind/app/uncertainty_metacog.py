"""Advisory metacog trigger when LLM language surface appears unstable."""

from __future__ import annotations

import asyncio
import logging
from threading import Thread
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry
from .settings import settings

logger = logging.getLogger("orion-mind.uncertainty_metacog")


def should_emit_llm_surface_instability(unc: dict[str, Any]) -> tuple[bool, str]:
    if not unc.get("available"):
        return False, "unavailable"
    tokens = int(unc.get("token_count_observed") or 0)
    if tokens <= 0:
        return False, "no_tokens"
    low_lp = int(unc.get("low_logprob_token_count") or 0)
    unstable = int(unc.get("unstable_span_count") or 0)
    margin = unc.get("mean_top1_margin")
    if unstable >= 1:
        return True, "unstable_span"
    if isinstance(margin, (int, float)) and margin < 0.75:
        return True, "low_mean_margin"
    if low_lp / tokens > 0.15:
        return True, "high_low_logprob_ratio"
    return False, "stable"


def _run_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _publish_metacog_trigger_async(
    trigger: MetacogTriggerV1,
    *,
    context: MindLLMRequestContext | None,
) -> None:
    if not settings.ORION_BUS_ENABLED:
        logger.info("metacog trigger skipped: ORION_BUS_ENABLED=false")
        return
    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    await bus.connect()
    try:
        corr = context.envelope_correlation_id() if context else uuid4()
        service_ref = ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        )
        trace = context.trace_baggage() if context else {}
        env = BaseEnvelope(
            kind="orion.metacog.trigger.v1",
            source=service_ref,
            correlation_id=corr,
            trace=trace,
            causality_chain=list(context.causality_chain or []) if context else [],
            payload=trigger.model_dump(mode="json"),
        )
        await bus.publish(settings.MIND_METACOG_TRIGGER_CHANNEL, env)
        logger.info(
            "mind_llm_surface_instability_metacog_trigger_published "
            "correlation_id=%s mind_run_id=%s phase=%s trigger_kind=%s reason=%s channel=%s",
            str(corr),
            context.mind_run_id if context else None,
            context.phase_name if context else None,
            trigger.trigger_kind,
            trigger.reason,
            settings.MIND_METACOG_TRIGGER_CHANNEL,
        )
    finally:
        await bus.close()


def maybe_publish_llm_surface_instability_trigger(
    telemetry: MindPhaseTelemetry,
    *,
    context: MindLLMRequestContext | None = None,
) -> None:
    if not settings.MIND_LLM_UNCERTAINTY_METACOG_ENABLED:
        return
    unc = telemetry.llm_uncertainty
    if not isinstance(unc, dict):
        return
    should_emit, detail = should_emit_llm_surface_instability(unc)
    if not should_emit:
        logger.debug(
            "mind_llm_surface_instability_metacog_skipped detail=%s mind_run_id=%s",
            detail,
            context.mind_run_id if context else None,
        )
        return
    tokens = max(int(unc.get("token_count_observed") or 0), 1)
    low_lp = int(unc.get("low_logprob_token_count") or 0)
    trigger = MetacogTriggerV1(
        trigger_kind="llm_surface_instability",
        reason="language_surface_unstable",
        pressure=min(1.0, low_lp / tokens),
        upstream={
            "llm_uncertainty": unc,
            "phase": telemetry.phase_name,
            "instability_detail": detail,
        },
    )
    try:
        _run_blocking(_publish_metacog_trigger_async(trigger, context=context))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "mind_llm_surface_instability_metacog_publish_failed mind_run_id=%s err=%s",
            context.mind_run_id if context else None,
            exc,
        )
