"""Gateway capacity tickets backed by the durable admission authority.

The ticket belongs to real backend work. In particular, cancelling a bus handler
does not release a ticket while its blocking HTTP executor thread is still alive.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar
from uuid import uuid4

import httpx
from fastapi.responses import StreamingResponse
from orion.schemas.resource_admission import (
    CapacityAcquireResultV1, CapacityAcquireV1, CapacityPermitV1,
    CapacityReleaseResultV1, CapacityRenewResultV1,
)

from .settings import settings

logger = logging.getLogger("orion-llm-gateway.capacity")
T = TypeVar("T")
_supervisors: set[asyncio.Task[Any]] = set()


class CapacityRejected(RuntimeError):
    """Capacity could not be acquired or its ownership could not be confirmed."""


class CapacityUnavailable(CapacityRejected):
    """Transport loss may hide an accepted acquire; retry its identical ID."""


def capacity_error(reason: str) -> dict[str, Any]:
    return {"error": {"type": "gateway_capacity_rejected", "message": reason}}


class CapacityStreamingResponse(StreamingResponse):
    """Close an opened upstream even if ASGI fails before iterating its body."""
    def __init__(self, *args: Any, cleanup: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


def _retain(task: asyncio.Task[Any]) -> None:
    _supervisors.add(task)
    def done(finished: asyncio.Task[Any]) -> None:
        _supervisors.discard(finished)
        if not finished.cancelled():
            finished.exception()  # detached handlers still retrieve terminal errors
    task.add_done_callback(done)


def stream_cleanup(upstream: Any, client: Any, permit: "CapacityPermit | None") -> Callable[[], Awaitable[None]]:
    """One retained cleanup task survives Starlette's repeated disconnect cancellation."""
    task: asyncio.Task[None] | None = None
    async def clean() -> None:
        try:
            await upstream.aclose()
        finally:
            try:
                await client.aclose()
            finally:
                if permit is not None:
                    await permit.close()
    async def close() -> None:
        nonlocal task
        if task is None:
            task = asyncio.create_task(clean(), name="capacity-stream-cleanup")
            _retain(task)
        await asyncio.shield(task)
    return close


class CapacityPermit:
    def __init__(self, *, lane: str, backend_key: str, correlation_id: str, budget_sec: float,
                 lease: dict[str, Any] | None = None) -> None:
        self.enabled = settings.llm_gateway_capacity_enabled
        self.request_id = str(uuid4())
        self.lane = lane
        self.backend_key = backend_key.rstrip("/")
        self.correlation_id = correlation_id
        self.lease = lease
        self.budget_sec = max(0.0, min(86400.0, budget_sec))
        self.deadline = time.monotonic() + self.budget_sec
        self.permit: dict[str, Any] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._lost = asyncio.Event()
        self._reason = "capacity_lost"
        self._expires_mono = 0.0
        self._interval = 1.0
        self._closed = False
        self._transferred = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    async def _post(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=settings.llm_gateway_lease_validation_timeout_sec) as client:
                response = await client.post(settings.llm_gateway_capacity_url.rstrip("/") + "/" + action,
                                             json=payload)
                response.raise_for_status()
                result = response.json()
            model = {"acquire": CapacityAcquireResultV1, "renew": CapacityRenewResultV1,
                     "release": CapacityReleaseResultV1}[action]
            return model.model_validate(result).model_dump(mode="json")
        except httpx.RequestError as exc:
            raise CapacityUnavailable("capacity_authority_unavailable") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise CapacityUnavailable("capacity_authority_unavailable") from exc
            raise CapacityRejected("capacity_authority_rejected") from exc
        except ValueError as exc:
            raise CapacityRejected("invalid_capacity_response") from exc

    def _accept(self, raw: Any) -> None:
        try:
            raw = CapacityPermitV1.model_validate(raw).model_dump(mode="json")
            if not isinstance(raw, dict) or not raw.get("permit_id"):
                raise ValueError("missing permit")
            if (raw.get("request_id") != self.request_id or raw.get("lane") != self.lane
                    or raw.get("backend_key") != self.backend_key
                    or raw.get("correlation_id") != self.correlation_id or raw.get("status") != "active"):
                raise ValueError("permit identity mismatch")
            if self.permit is not None and raw["permit_id"] != self.permit["permit_id"]:
                raise ValueError("permit changed")
            expected_lease = self.lease or {}
            if raw.get("lease_id") != expected_lease.get("lease_id") or raw.get("generation") != expected_lease.get("generation"):
                raise ValueError("permit lease mismatch")
            expiry = datetime.fromisoformat(str(raw["expires_at"]).replace("Z", "+00:00"))
            heartbeat = datetime.fromisoformat(str(raw["heartbeat_at"]).replace("Z", "+00:00"))
            if expiry.tzinfo is None or heartbeat.tzinfo is None:
                raise ValueError("permit timestamps require timezone")
            remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
            ttl = (expiry - heartbeat).total_seconds()
            if remaining <= 0 or ttl <= 0:
                raise ValueError("expired permit")
            self._expires_mono = time.monotonic() + remaining
            self._interval = min(15.0, ttl / 3.0, remaining / 3.0)
            self.permit = dict(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise CapacityRejected("invalid_capacity_permit") from exc

    def _lose(self, reason: str) -> None:
        self._reason = reason
        self._lost.set()

    async def acquire(self) -> "CapacityPermit":
        if not self.enabled:
            return self
        payload = CapacityAcquireV1(
            request_id=self.request_id, correlation_id=self.correlation_id,
            lane=self.lane, backend_key=self.backend_key,
            max_inflight=settings.llm_gateway_upstream_max_inflight,
            budget_sec=self.budget_sec, lease=self.lease,
        ).model_dump(mode="json") if self.remaining > 0 else None
        while self.remaining > 0:
            try:
                async with asyncio.timeout(self.remaining):
                    result = await self._post("acquire", payload)
            except TimeoutError as exc:
                raise CapacityRejected("capacity_wait_budget_exhausted") from exc
            except CapacityUnavailable:
                await asyncio.sleep(min(settings.llm_gateway_background_poll_interval_sec, self.remaining))
                continue
            if result.get("acquired") is True:
                self._accept(result.get("permit"))
                if self.remaining <= 0:
                    await self.close()
                    raise CapacityRejected("capacity_wait_budget_exhausted")
                self._heartbeat = asyncio.create_task(self._renew_loop(), name=f"capacity-{self.request_id}")
                return self
            reason = str(result.get("reason") or "capacity_busy")
            if reason not in {"durable_lease_active", "capacity_full", "owner_request_active", "durable_waiting"}:
                raise CapacityRejected(reason)
            await asyncio.sleep(min(settings.llm_gateway_background_poll_interval_sec, self.remaining))
        raise CapacityRejected("capacity_wait_budget_exhausted")

    async def _renew(self) -> None:
        async with self._lock:
            if self._closed or self.permit is None:
                raise CapacityRejected("capacity_released")
            result = await self._post("renew", {"request_id": self.request_id, "permit_id": self.permit["permit_id"]})
            if result.get("permit") is not None:
                self._accept(result["permit"])
            if result.get("valid") is not True:
                raise CapacityRejected(str(result.get("reason") or "capacity_lost"))

    async def _renew_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._interval)
            try:
                await self._renew()
            except CapacityRejected as exc:
                # Keep attempting renewal until the actual backend ends. An
                # authority outage invalidates the result, not the live thread.
                self._lose(str(exc))

    async def check(self) -> None:
        if not self.enabled:
            return
        if self._lost.is_set() or self._closed or self._expires_mono <= time.monotonic():
            raise CapacityRejected(self._reason)
        try:
            await self._renew()
        except CapacityRejected as exc:
            self._lose(str(exc))
            raise

    async def _race(self, operation: Awaitable[T]) -> T:
        if not self.enabled:
            return await operation
        task = asyncio.ensure_future(operation)
        lost = asyncio.create_task(self._lost.wait())
        try:
            if self._lost.is_set():
                raise CapacityRejected(self._reason)
            if self.remaining <= 0:
                raise CapacityRejected("capacity_budget_exhausted")
            done, _ = await asyncio.wait({task, lost}, timeout=self.remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise CapacityRejected("capacity_budget_exhausted")
            if lost in done:
                raise CapacityRejected(self._reason)
            return await task
        finally:
            for pending in (task, lost):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, lost, return_exceptions=True)

    async def run(self, operation: Awaitable[T], *, validate_result: bool = True) -> T:
        result = await self._race(operation)
        if validate_result:
            await self._race(self.check())
        return result

    async def chunks(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        iterator = chunks.__aiter__()
        while True:
            try:
                value = await self.run(anext(iterator), validate_result=False)
            except StopAsyncIteration:
                await self._race(self.check())
                return
            if self.enabled and self._expires_mono <= time.monotonic():
                raise CapacityRejected("capacity_expired")
            yield value

    async def run_blocking(self, submit: Callable[[], Future[T]]) -> T:
        """Transfer ownership to a supervisor which survives handler cancellation."""
        await self._race(self.check())
        future = submit()
        if not self.enabled:
            return await asyncio.wrap_future(future)
        self._transferred = True
        loop = asyncio.get_running_loop()
        async def finish() -> T:
            try:
                result = await asyncio.wrap_future(future)
                await self.check()
                return result
            finally:
                if future.done():
                    await self.close(force=True)
        supervisor = asyncio.create_task(finish(), name=f"capacity-backend-{self.request_id}")
        _retain(supervisor)
        # Install synchronously: shutdown can cancel a task before its coroutine
        # starts, in which case even that coroutine's finally block never runs.
        def release_when_done(_future) -> None:
            async def cleanup() -> None:
                await asyncio.gather(supervisor, return_exceptions=True)
                await self.close(force=True)
            def schedule() -> None:
                _retain(asyncio.create_task(cleanup()))
            try:
                loop.call_soon_threadsafe(schedule)
            except RuntimeError:
                logger.warning("capacity_release_waits_for_expiry request_id=%s", self.request_id)
        future.add_done_callback(release_when_done)
        return await self._race(asyncio.shield(supervisor))

    async def close(self, *, force: bool = False) -> None:
        if not self.enabled or self._closed or (self._transferred and not force):
            return
        self._closed = True
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        if self.permit is not None:
            try:
                await self._post("release", {"request_id": self.request_id, "permit_id": self.permit["permit_id"]})
            except CapacityRejected:
                logger.warning("capacity_release_unconfirmed request_id=%s permit_id=%s", self.request_id, self.permit["permit_id"])
