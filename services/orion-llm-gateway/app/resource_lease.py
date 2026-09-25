"""Durable-run lease validation (bus ``options.resource_lease``, HTTP ``X-Orion-Resource-Lease``),
and parsing of the stage-4 GPU pool lease ref (bus ``options.gpu_lease``, HTTP ``X-Orion-Gpu-Lease``).

The two coexist until stage 4.6 deletes the durable half. A GPU lease ref is not validated here:
the pool is its fencing authority, reached by ``attach`` in pool_placement (a stale or unknown
hold makes attach refuse, and the call gets ``gpu_pool_unavailable``).

Until stage 4 durable-runs still issues these leases, so the gateway keeps checking them with the
broker. They are an ADMISSION TOKEN only: where the call runs is always a GPU pool grant
(pool_placement.py). The lease's ``backend_key`` is therefore compared with itself, not with the
granted URL -- the pool may legitimately place the call on a different role than the one durable
admission saw in ``GET /routes``. Lane and broker generation are still enforced.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Awaitable, Optional, TypeVar

from orion.llm.resource_lease import (
    GPU_LEASE_HEADER, GPU_LEASE_OPTION, LEASE_HEADER, ResourceLeaseRejected, decode_gpu_lease_header,
    decode_lease_header, validate_resource_lease,
)
from orion.schemas.gpu_pool import GpuLeaseRefV1

from .settings import settings

T = TypeVar("T")


class LeaseGuard:
    def __init__(self, lease: Any, *, lane: str) -> None:
        self.lease = lease
        self.lane = lane
        self.backend_key = str(lease.get("backend_key") or "") if isinstance(lease, dict) else ""
        self.enabled = bool(settings.llm_gateway_lease_validation_enabled and lease is not None)

    @classmethod
    def from_headers(cls, headers: Any, *, lane: str) -> "LeaseGuard":
        value = headers.get(LEASE_HEADER)
        lease = decode_lease_header(value) if value is not None else None
        return cls(lease, lane=lane)

    async def check(self) -> None:
        if self.enabled:
            await validate_resource_lease(
                self.lease, lane=self.lane, backend_key=self.backend_key,
                validation_url=settings.llm_gateway_lease_validation_url,
                timeout_sec=settings.llm_gateway_lease_validation_timeout_sec,
            )

    async def run(self, operation: Awaitable[T], *, validate_result: bool = True) -> T:
        if not self.enabled:
            return await operation
        task = asyncio.ensure_future(operation)
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=settings.llm_gateway_lease_check_interval_sec)
                if not done:
                    await self.check()
            result = await task
            if validate_result:
                await self.check()
            return result
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def chunks(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        iterator = chunks.__aiter__()
        checked_at = time.monotonic()
        while True:
            try:
                # Also checks idle streams, rather than waiting for their next byte.
                chunk = await self.run(anext(iterator), validate_result=False)
            except StopAsyncIteration:
                await self.check()
                return
            if time.monotonic() - checked_at >= settings.llm_gateway_lease_check_interval_sec:
                await self.check()
                checked_at = time.monotonic()
            yield chunk


def gpu_lease_from_options(options: Any) -> Optional[GpuLeaseRefV1]:
    """``options.gpu_lease`` (a GpuLeaseRefV1 dict) or None. Malformed -> ResourceLeaseRejected,
    never None: a call that meant to run under a hold must not silently take a lease of its own."""
    value = (options or {}).get(GPU_LEASE_OPTION) if isinstance(options, dict) else None
    if value is None:
        return None
    try:
        return GpuLeaseRefV1.model_validate(value)
    except ValueError as exc:
        raise ResourceLeaseRejected("malformed_gpu_lease") from exc


def gpu_lease_from_headers(headers: Any) -> Optional[GpuLeaseRefV1]:
    value = headers.get(GPU_LEASE_HEADER)
    return decode_gpu_lease_header(value) if value is not None else None


def lease_error(reason: str) -> dict[str, Any]:
    return {"error": {"type": "resource_lease_rejected", "message": reason}}
