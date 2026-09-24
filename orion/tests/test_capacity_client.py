"""orion.durable_admission.capacity_client.GpuCapacityPermit -- the shared
acquire-before/release-after guard orion-thought (diffusion) and
orion-world-model both use to fence circe's GPU2. Mirrors the mock-authority
pattern in services/orion-llm-gateway/tests/test_capacity.py, trimmed to
this module's own (smaller) surface: acquire/renew/close only, no streaming
or blocking-executor-transfer helpers.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.durable_admission import capacity_client as cc

CAPACITY_URL = "http://durable-runs:8121/capacity"


class FakeAuthority:
    """Emulates orion.durable_admission.capacity.PostgresCapacityStore's real
    per-backend_key max_inflight enforcement, closely enough for these tests:
    one active permit per (backend_key) up to the minimum max_inflight of
    every currently-active permit on that key, first-come-first-served."""

    def __init__(self) -> None:
        self.active: dict[str, dict] = {}
        # Server-side-only bookkeeping, never part of the public permit shape
        # (CapacityPermitV1 is extra="forbid" -- it has no max_inflight field).
        self._max_inflight: dict[str, int] = {}
        self.events: list[tuple[str, dict]] = []
        self.ttl = 60.0
        self.force_reason: str | None = None
        self.valid = True

    async def post(self, action: str, payload: dict) -> dict:
        self.events.append((action, dict(payload)))
        if action == "acquire":
            return self._acquire(payload)
        permit = self.active.get(payload["request_id"])
        if action == "renew":
            if permit is None:
                return {"valid": False, "permit": None}
            now = datetime.now(timezone.utc)
            permit.update(
                heartbeat_at=now.isoformat(),
                expires_at=(now + timedelta(seconds=self.ttl)).isoformat(),
            )
            return {"valid": self.valid, "permit": permit if self.valid else None}
        assert action == "release"
        self.active.pop(payload["request_id"], None)
        self._max_inflight.pop(payload["request_id"], None)
        return {"released": True}

    def _acquire(self, payload: dict) -> dict:
        request_id = payload["request_id"]
        existing = self.active.get(request_id)
        if existing:
            return {"acquired": True, "reason": "duplicate", "permit": existing}
        if self.force_reason is not None:
            return {"acquired": False, "reason": self.force_reason, "permit": None}
        same_backend_ids = [rid for rid, row in self.active.items() if row["backend_key"] == payload["backend_key"]]
        limit = min([payload["max_inflight"], *(self._max_inflight[rid] for rid in same_backend_ids)])
        if len(same_backend_ids) >= limit:
            return {"acquired": False, "reason": "capacity_full", "permit": None}
        now = datetime.now(timezone.utc)
        permit = {key: payload[key] for key in ("request_id", "correlation_id", "lane", "backend_key")}
        permit.update(
            permit_id=f"permit-{request_id}",
            lease_id=None,
            generation=None,
            granted_at=now.isoformat(),
            heartbeat_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.ttl)).isoformat(),
            status="active",
        )
        self.active[request_id] = permit
        self._max_inflight[request_id] = payload["max_inflight"]
        return {"acquired": True, "reason": "acquired", "permit": permit}


@pytest.fixture
def authority(monkeypatch):
    authority = FakeAuthority()

    async def post(self, action, payload):
        return await authority.post(action, payload)

    monkeypatch.setattr(cc.GpuCapacityPermit, "_post", post)
    return authority


def make_permit(*, lane="diffusion", backend_key="http://100.112.254.99:8014",
                 max_inflight=1, budget_sec=5.0, poll_interval_sec=0.001) -> cc.GpuCapacityPermit:
    return cc.GpuCapacityPermit(
        capacity_url=CAPACITY_URL, lane=lane, backend_key=backend_key,
        correlation_id="corr", max_inflight=max_inflight, budget_sec=budget_sec,
        poll_interval_sec=poll_interval_sec,
    )


@pytest.mark.asyncio
async def test_acquire_and_close_happy_path(authority):
    permit = await make_permit().acquire()
    try:
        assert permit.permit is not None
        assert permit.permit["lane"] == "diffusion"
    finally:
        await permit.close()
    assert not authority.active
    assert any(action == "release" for action, _ in authority.events)


@pytest.mark.asyncio
async def test_second_acquire_on_same_backend_key_waits_then_succeeds(authority):
    """The property this whole feature exists for: two different lanes
    sharing one backend_key get real mutual exclusion, not just per-lane."""
    first = await make_permit(lane="diffusion").acquire()
    second_permit = make_permit(lane="world-model", budget_sec=5.0)
    second_task = asyncio.create_task(second_permit.acquire())
    await asyncio.sleep(0.01)
    assert not second_task.done(), "second lane must not acquire while diffusion holds the shared key"
    await first.close()
    second = await asyncio.wait_for(second_task, 2)
    try:
        assert second.permit["lane"] == "world-model"
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_short_budget_fails_fast_when_contended(authority):
    """world-model's own precedence model: a short budget must not camp."""
    first = await make_permit(lane="diffusion").acquire()
    try:
        short = make_permit(lane="world-model", budget_sec=0.05, poll_interval_sec=0.01)
        with pytest.raises(cc.CapacityRejected, match="capacity_wait_budget_exhausted"):
            await short.acquire()
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_non_retryable_reason_raises_immediately(authority):
    authority.force_reason = "elastic_requires_owner"
    permit = make_permit(budget_sec=5.0)
    with pytest.raises(cc.CapacityRejected, match="elastic_requires_owner"):
        await permit.acquire()
    # Only one acquire call -- a non-retryable reason must not loop.
    assert len([e for e in authority.events if e[0] == "acquire"]) == 1


@pytest.mark.asyncio
async def test_renew_loop_flags_lost_on_invalid_renewal(authority):
    authority.ttl = 0.05
    permit = await make_permit(budget_sec=5.0).acquire()
    authority.valid = False
    try:
        await asyncio.wait_for(_wait_for(lambda: permit.lost), 2)
        assert permit.lost
    finally:
        await permit.close()


@pytest.mark.asyncio
async def test_unreachable_authority_raises_capacity_unavailable():
    """No monkeypatched authority -- a real transport failure against an
    unreachable address, not a scripted one, exercises the real httpx path."""
    permit = cc.GpuCapacityPermit(
        capacity_url="http://unreachable.invalid:1/capacity", lane="diffusion",
        backend_key="http://100.112.254.99:8014", correlation_id="corr",
        max_inflight=1, budget_sec=0.05, poll_interval_sec=0.01,
    )
    with pytest.raises(cc.CapacityUnavailable):
        await permit._post("acquire", {})


async def _wait_for(predicate, interval: float = 0.005) -> None:
    while not predicate():
        await asyncio.sleep(interval)
