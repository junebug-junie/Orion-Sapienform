from __future__ import annotations

import logging
from uuid import uuid4

import httpx

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1

from .settings import settings

logger = logging.getLogger("orion-self-experiments.context_exec_client")

_SOURCE = ServiceRef(name=settings.service_name, version=settings.service_version)


async def dispatch_context_exec(request: ContextExecRequestV1) -> ContextExecRunV1:
    transport = (settings.self_experiments_context_exec_dispatch_transport or "bus").strip().lower()
    timeout = float(settings.self_experiments_context_exec_timeout_seconds)
    if transport == "http":
        return await _dispatch_http(request, timeout=timeout)
    return await _dispatch_bus(request, timeout=timeout)


async def _dispatch_http(request: ContextExecRequestV1, *, timeout: float) -> ContextExecRunV1:
    url = f"{settings.self_experiments_context_exec_url.rstrip('/')}/context-exec/run"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=request.model_dump(mode="json"))
        resp.raise_for_status()
        return ContextExecRunV1.model_validate(resp.json())


async def _dispatch_bus(request: ContextExecRequestV1, *, timeout: float) -> ContextExecRunV1:
    correlation_id = request.correlation_id or request.request_id or str(uuid4())
    reply_channel = f"orion:self_experiments:context_exec:reply:{uuid4()}"
    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    if not bus.enabled:
        raise RuntimeError("orion_bus_disabled")
    await bus.connect()
    try:
        env = BaseEnvelope(
            kind="context.exec.request.v1",
            source=_SOURCE,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=request.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(
            settings.self_experiments_context_exec_request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"context_exec_decode_failed:{decoded.error}")
        return ContextExecRunV1.model_validate(decoded.envelope.payload)
    finally:
        await bus.close()
