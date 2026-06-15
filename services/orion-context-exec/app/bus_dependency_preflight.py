"""Fast bus consumer readiness checks for investigation_v2 dependency preflight."""

from __future__ import annotations

import asyncio

from orion.bus.consumer_readiness import BusConsumerReadinessResult, check_bus_consumer_readiness
from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.context_exec import SourceResult, SourceStatus

from .settings import settings

_SERVICE_DISPLAY = {
    "recall": "RecallService",
    "llm-gateway": "LLMGatewayService",
}


async def _check_intake_ready(
    bus: OrionBusAsync,
    *,
    intake_channel: str,
    service_name: str,
    timeout_sec: float,
) -> BusConsumerReadinessResult:
    try:
        return await asyncio.wait_for(
            check_bus_consumer_readiness(
                bus.redis,
                intake_channel=intake_channel,
                service_name=service_name,
                check_heartbeat=False,
            ),
            timeout=float(timeout_sec),
        )
    except TimeoutError:
        return BusConsumerReadinessResult(
            ok=False,
            bus_consumer_ready=False,
            intake_channel=intake_channel,
            subscriber_count=0,
            dependency_status="unavailable",
            error=f"readiness check timed out after {timeout_sec}s",
        )


async def check_recall_bus_ready(bus: OrionBusAsync, *, timeout_sec: float) -> BusConsumerReadinessResult:
    return await _check_intake_ready(
        bus,
        intake_channel=settings.channel_recall_intake,
        service_name="recall",
        timeout_sec=timeout_sec,
    )


async def check_llm_gateway_bus_ready(bus: OrionBusAsync, *, timeout_sec: float) -> BusConsumerReadinessResult:
    return await _check_intake_ready(
        bus,
        intake_channel=settings.channel_llm_intake,
        service_name="llm-gateway",
        timeout_sec=timeout_sec,
    )


def unavailable_source_result(source: str, readiness: BusConsumerReadinessResult) -> SourceResult:
    display = _SERVICE_DISPLAY.get(source, source)
    error = f"{display} bus consumer not ready: subscriber_count={readiness.subscriber_count}"
    if readiness.error and readiness.subscriber_count <= 0:
        error = f"{display} bus consumer not ready: subscriber_count={readiness.subscriber_count}"
    return SourceResult(
        source=source,
        status=SourceStatus.unavailable,
        summary=f"{source} dependency unavailable (bus preflight)",
        error=error,
        metadata={
            "intake_channel": readiness.intake_channel,
            "subscriber_count": readiness.subscriber_count,
            "preflight": True,
            "readiness_error": readiness.error,
        },
    )
