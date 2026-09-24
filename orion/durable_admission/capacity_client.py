"""Minimal client for the durable-runs Gateway capacity-permit authority
(`orion.durable_admission.capacity.PostgresCapacityStore`, fronted by
`services/orion-durable-runs`'s `/capacity/*` routes).

A lean subset of `services/orion-llm-gateway/app/capacity.py`'s
`CapacityPermit` -- that class also carries streaming-response cleanup, a
blocking-executor-transfer path, and a Gateway-specific borrowed-capacity
lease rule (`BURST_LLM_ROUTES`), none of which apply to a caller that just
needs "acquire before touching a shared physical resource, release after"
for one bounded operation (a single HTTP call, a single forward pass).
Every config value is a constructor argument on purpose -- this module has
no dependency on any one service's settings module, so any service can use
it without importing another service's app package.

First real use (docs/architecture/durable-gateway-capacity.md's own stated
intended pattern for "a standalone GPU-bound HTTP call"): `orion-thought`
and `orion-world-model` sharing circe's GPU2 (a physical card with zero
OS/driver-level arbitration between the two processes touching it) each
acquire a permit on the same `backend_key` before touching the GPU, so the
broker's existing `max_inflight` enforcement becomes real cross-service
mutual exclusion on the actual contested hardware.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from orion.schemas.resource_admission import (
    CapacityAcquireResultV1,
    CapacityAcquireV1,
    CapacityPermitV1,
    CapacityReleaseResultV1,
    CapacityRenewResultV1,
)

logger = logging.getLogger(__name__)

# Reasons the authority returns when the permit is merely contended, not
# refused -- the caller should keep polling until its own budget runs out.
_RETRYABLE_REASONS = {"durable_lease_active", "capacity_full", "owner_request_active", "durable_waiting"}


class CapacityRejected(RuntimeError):
    """Capacity could not be acquired, or its ownership could not be confirmed."""


class CapacityUnavailable(CapacityRejected):
    """Transport loss may hide an accepted acquire; safe to retry the same request_id."""


class GpuCapacityPermit:
    """Acquire-before / release-after guard for a shared physical resource.

    Not a stream-safe wrapper the way Gateway's `CapacityPermit` is -- this
    is for a single bounded operation: `acquire()`, do the work, `close()`.
    A background renew loop keeps the permit alive for operations that
    outlive the authority's own lease TTL (90s by default server-side,
    `DURABLE_RUNS_LEASE_SECONDS`), independent of the caller's own
    `budget_sec` (which only bounds how long `acquire()` will poll trying to
    get IN, not how long the permit is held once granted).
    """

    def __init__(
        self,
        *,
        capacity_url: str,
        lane: str,
        backend_key: str,
        correlation_id: str,
        max_inflight: int,
        budget_sec: float,
        poll_interval_sec: float = 1.0,
        http_timeout_sec: float = 5.0,
    ) -> None:
        self.capacity_url = capacity_url.rstrip("/")
        self.lane = lane
        self.backend_key = backend_key.rstrip("/")
        self.correlation_id = correlation_id
        self.max_inflight = max_inflight
        self.poll_interval_sec = poll_interval_sec
        self.http_timeout_sec = http_timeout_sec
        self.request_id = str(uuid4())
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

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def lost(self) -> bool:
        """True once a background renew has failed -- the permit is no
        longer held. This cannot cancel work already dispatched to CUDA;
        callers doing a long-running GPU operation should check this
        periodically and stop starting new work once it flips, rather than
        relying on it to interrupt anything in flight."""
        return self._lost.is_set()

    async def _post(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout_sec) as client:
                response = await client.post(f"{self.capacity_url}/{action}", json=payload)
                response.raise_for_status()
                result = response.json()
            model = {
                "acquire": CapacityAcquireResultV1,
                "renew": CapacityRenewResultV1,
                "release": CapacityReleaseResultV1,
            }[action]
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
            if (
                raw.get("request_id") != self.request_id
                or raw.get("lane") != self.lane
                or raw.get("backend_key") != self.backend_key
                or raw.get("correlation_id") != self.correlation_id
                or raw.get("status") != "active"
            ):
                raise ValueError("permit identity mismatch")
            if self.permit is not None and raw["permit_id"] != self.permit["permit_id"]:
                raise ValueError("permit changed")
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

    async def acquire(self) -> "GpuCapacityPermit":
        while self.remaining > 0:
            payload = CapacityAcquireV1(
                request_id=self.request_id,
                correlation_id=self.correlation_id,
                lane=self.lane,
                backend_key=self.backend_key,
                max_inflight=self.max_inflight,
                budget_sec=self.budget_sec,
            ).model_dump(mode="json")
            try:
                async with asyncio.timeout(self.remaining):
                    result = await self._post("acquire", payload)
            except TimeoutError as exc:
                raise CapacityRejected("capacity_wait_budget_exhausted") from exc
            except CapacityUnavailable:
                await asyncio.sleep(min(self.poll_interval_sec, self.remaining))
                continue
            if result.get("acquired") is True:
                try:
                    self._accept(result.get("permit"))
                except CapacityRejected:
                    # The authority already granted this permit -- _accept
                    # only rejected it locally (e.g. clock-skew "expired
                    # permit", or a duplicate-acquire race returning a
                    # different permit_id). Release the real server-side
                    # slot before propagating, or it sits held on the
                    # shared backend_key until its TTL expires, starving
                    # the other side of the mutex this whole class exists
                    # for (review finding, caught before this shipped).
                    raw = result.get("permit") or {}
                    permit_id = raw.get("permit_id")
                    if permit_id:
                        try:
                            await self._post(
                                "release",
                                {"request_id": self.request_id, "permit_id": permit_id},
                            )
                        except CapacityRejected:
                            logger.warning(
                                "gpu_capacity_release_unconfirmed_after_invalid_accept "
                                "request_id=%s permit_id=%s",
                                self.request_id,
                                permit_id,
                            )
                    raise
                self._heartbeat = asyncio.create_task(
                    self._renew_loop(), name=f"gpu-capacity-{self.request_id}"
                )
                return self
            reason = str(result.get("reason") or "capacity_busy")
            if reason not in _RETRYABLE_REASONS:
                raise CapacityRejected(reason)
            await asyncio.sleep(min(self.poll_interval_sec, self.remaining))
        raise CapacityRejected("capacity_wait_budget_exhausted")

    async def _renew(self) -> None:
        async with self._lock:
            if self._closed or self.permit is None:
                raise CapacityRejected("capacity_released")
            result = await self._post(
                "renew", {"request_id": self.request_id, "permit_id": self.permit["permit_id"]}
            )
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
                # Keep attempting renewal until the actual operation ends. An
                # authority outage invalidates the result, not GPU work
                # already dispatched.
                self._lose(str(exc))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        if self.permit is not None:
            try:
                await self._post(
                    "release", {"request_id": self.request_id, "permit_id": self.permit["permit_id"]}
                )
            except CapacityRejected:
                logger.warning(
                    "gpu_capacity_release_unconfirmed request_id=%s permit_id=%s",
                    self.request_id,
                    self.permit["permit_id"],
                )

    async def __aenter__(self) -> "GpuCapacityPermit":
        return await self.acquire()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
