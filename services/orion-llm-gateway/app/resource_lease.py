"""Optional fencing around existing bus and FCC HTTP generation paths."""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Awaitable, TypeVar

from orion.llm.resource_lease import (
    LEASE_HEADER, ResourceLeaseRejected, decode_lease_header, validate_resource_lease,
)

from .settings import settings

T = TypeVar("T")


class LeaseGuard:
    def __init__(self, lease: Any, *, lane: str, backend_key: str) -> None:
        self.lease = lease
        self.lane = lane
        self.backend_key = backend_key
        self.enabled = bool(settings.llm_gateway_lease_validation_enabled and lease is not None)

    @classmethod
    def from_headers(cls, headers: Any, *, lane: str, backend_key: str) -> "LeaseGuard":
        value = headers.get(LEASE_HEADER)
        lease = decode_lease_header(value) if value is not None else None
        return cls(lease, lane=lane, backend_key=backend_key)

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


def lease_error(reason: str) -> dict[str, Any]:
    return {"error": {"type": "resource_lease_rejected", "message": reason}}
