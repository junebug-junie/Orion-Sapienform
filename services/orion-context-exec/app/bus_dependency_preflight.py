"""Fast bus consumer readiness checks for investigation_v2 dependency preflight."""

from __future__ import annotations

import asyncio
from typing import Any

from orion.bus.consumer_readiness import BusConsumerReadinessResult, check_bus_consumer_readiness
from orion.core.bus.async_service import OrionBusAsync
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


def dependency_health_entry(
    readiness: BusConsumerReadinessResult,
    *,
    redis_ping_ok: bool,
) -> dict[str, Any]:
    return {
        "bus_consumer_ready": readiness.bus_consumer_ready,
        "intake_channel": readiness.intake_channel,
        "subscriber_count": readiness.subscriber_count,
        "redis_ping_ok": redis_ping_ok,
        "heartbeat_fresh": readiness.heartbeat_fresh,
        "rpc_smoke_ok": readiness.rpc_smoke_ok,
        "status": readiness.dependency_status,
    }


def _unavailable_dependency_entry(intake_channel: str, *, redis_ping_ok: bool, error: str) -> dict[str, Any]:
    return {
        "bus_consumer_ready": False,
        "intake_channel": intake_channel,
        "subscriber_count": 0,
        "redis_ping_ok": redis_ping_ok,
        "heartbeat_fresh": None,
        "rpc_smoke_ok": None,
        "status": "unavailable",
        "error": error,
    }


async def _redis_ping_ok(redis) -> bool:
    try:
        pong = await redis.ping()
        return pong is True or pong == b"PONG" or str(pong).upper() == "PONG"
    except Exception:
        return False


async def collect_bus_dependencies_health(
    bus: OrionBusAsync | None,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    """Structured bus dependency readiness for context-exec /health."""
    if not settings.orion_bus_enabled:
        return {
            "bus_enabled": False,
            "bus_connected": False,
            "bus_consumer_ready": None,
            "dependencies": {},
        }

    recall_channel = settings.channel_recall_intake
    llm_channel = settings.channel_llm_intake
    if bus is None or not getattr(bus, "enabled", False) or getattr(bus, "redis", None) is None:
        return {
            "bus_enabled": True,
            "bus_connected": False,
            "bus_consumer_ready": False,
            "dependencies": {
                "recall": _unavailable_dependency_entry(
                    recall_channel,
                    redis_ping_ok=False,
                    error="bus not connected",
                ),
                "llm_gateway": _unavailable_dependency_entry(
                    llm_channel,
                    redis_ping_ok=False,
                    error="bus not connected",
                ),
            },
        }

    ping_ok = await _redis_ping_ok(bus.redis)
    recall_ready, llm_ready = await asyncio.gather(
        check_recall_bus_ready(bus, timeout_sec=timeout_sec),
        check_llm_gateway_bus_ready(bus, timeout_sec=timeout_sec),
    )
    dependencies = {
        "recall": dependency_health_entry(recall_ready, redis_ping_ok=ping_ok),
        "llm_gateway": dependency_health_entry(llm_ready, redis_ping_ok=ping_ok),
    }
    return {
        "bus_enabled": True,
        "bus_connected": True,
        "bus_consumer_ready": recall_ready.ok and llm_ready.ok,
        "dependencies": dependencies,
    }


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
