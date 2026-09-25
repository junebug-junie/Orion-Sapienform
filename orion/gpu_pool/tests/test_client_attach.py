"""Stage 4.4 client side of ``attach`` and ``status`` (orion.gpu_pool.client).

A scripted bus with the real OrionCodec both ways: each lease RPC is recorded and answered from a
per-verb script, so the wire request the client builds is what gets asserted -- not a mock's view
of it. The pool side of both verbs is stage 4.3.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable

import pytest

from orion.core.bus.codec import OrionCodec
from orion.gpu_pool.client import (
    DURABLE_RUN_HOLDER_PREFIX, LeaseUnavailable, durable_run_holder, gpu_lease, lease_status, validate_hold_ref,
)
from orion.schemas.gpu_pool import GpuLeaseGrantV1, GpuLeaseRefV1, GpuLeaseReplyV1, GpuLeaseRequestV1

REF = GpuLeaseRefV1(lease_id="hold-1", generation=3, role="agent", holder="durable-runs:run-1")


def _grant(lease_id: str, generation: int = 1) -> GpuLeaseGrantV1:
    return GpuLeaseGrantV1(lease_id=lease_id, generation=generation, role="agent", cards=["gpu1"],
                           url="http://agent:8015", served_by="circe-worker-agent")


class ScriptedBus:
    def __init__(self, answer: Callable[[GpuLeaseRequestV1], GpuLeaseReplyV1]):
        self.codec = OrionCodec()
        self.requests: list[GpuLeaseRequestV1] = []
        self._answer = answer

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec=60.0, health_label=None):
        wire = self.codec.decode(self.codec.encode(envelope)).envelope
        req = GpuLeaseRequestV1.model_validate(wire.payload)
        self.requests.append(req)
        reply = envelope.model_copy(update={"payload": self._answer(req).model_dump(mode="json")})
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(reply)}

    @contextlib.asynccontextmanager
    async def subscribe(self, *channels, patterns=False):
        yield object()

    async def iter_messages(self, pubsub):
        await asyncio.Event().wait()
        yield  # pragma: no cover


def _pool(req: GpuLeaseRequestV1) -> GpuLeaseReplyV1:
    if req.verb == "attach":
        if req.hold_lease_id == REF.lease_id and req.hold_generation == REF.generation:
            return GpuLeaseReplyV1(status="granted", lease_id="child-1", grant=_grant("child-1"))
        return GpuLeaseReplyV1(status="unknown_lease", reason="stale_hold_generation")
    if req.verb == "status":
        return GpuLeaseReplyV1(status="granted", lease_id=req.lease_id, grant=_grant(req.lease_id, REF.generation))
    return GpuLeaseReplyV1(status="ok", lease_id=req.lease_id)


@pytest.mark.asyncio
async def test_hold_sends_attach_with_the_hold_fence_and_never_acquire():
    bus = ScriptedBus(_pool)
    async with gpu_lease(bus, work_class="agent", holder="http:anthropic", hold=REF,
                         turn_correlation_id="turn-1", deadline_sec=5) as lease:
        assert lease.lease_id == "child-1" and lease.grant.role == "agent"
    attach, release = bus.requests
    assert attach.verb == "attach"
    assert (attach.hold_lease_id, attach.hold_generation) == (REF.lease_id, REF.generation)
    assert attach.lease_id is None and attach.request_id and attach.work_class == "agent"
    assert attach.turn_correlation_id == "turn-1" and attach.kind == "request"
    assert release.verb == "release" and release.lease_id == "child-1" and release.outcome == "ok"
    assert not any(r.verb == "acquire" for r in bus.requests)


@pytest.mark.asyncio
async def test_refused_attach_raises_and_does_not_retry_as_acquire():
    bus = ScriptedBus(_pool)
    stale = REF.model_copy(update={"generation": 2})
    with pytest.raises(LeaseUnavailable):
        async with gpu_lease(bus, work_class="agent", holder="http:anthropic", hold=stale, deadline_sec=5):
            pytest.fail("must not be granted")
    assert {r.verb for r in bus.requests} <= {"attach", "cancel"}


@pytest.mark.asyncio
async def test_no_hold_is_still_a_plain_acquire():
    bus = ScriptedBus(lambda req: GpuLeaseReplyV1(status="granted", lease_id="l", grant=_grant("l"))
                      if req.verb == "acquire" else GpuLeaseReplyV1(status="ok"))
    async with gpu_lease(bus, work_class="agent", holder="x", deadline_sec=5):
        pass
    assert bus.requests[0].verb == "acquire" and bus.requests[0].hold_lease_id is None


@pytest.mark.asyncio
async def test_status_is_a_side_effect_free_read_of_one_lease():
    bus = ScriptedBus(_pool)
    reply = await lease_status(bus, "hold-1", source="orion-hub")
    assert reply.status == "granted"
    assert [(r.verb, r.lease_id) for r in bus.requests] == [("status", "hold-1")]


@pytest.mark.asyncio
async def test_validate_hold_ref_accepts_a_live_hold_at_its_generation():
    await validate_hold_ref(ScriptedBus(_pool), REF, source="orion-hub", expected_holder=durable_run_holder("run-1"))


@pytest.mark.asyncio
async def test_validate_hold_ref_accepts_a_recalled_hold_inside_its_grace():
    bus = ScriptedBus(lambda req: GpuLeaseReplyV1(status="recall", lease_id=req.lease_id,
                                                  grant=_grant(req.lease_id, REF.generation)))
    await validate_hold_ref(bus, REF, source="orion-hub")


@pytest.mark.parametrize("reply,reason", [
    (GpuLeaseReplyV1(status="unknown_lease"), "gpu_lease_unknown_lease"),
    (GpuLeaseReplyV1(status="queued", lease_id="hold-1"), "gpu_lease_queued"),
    (GpuLeaseReplyV1(status="granted", lease_id="hold-1", grant=_grant("hold-1", 2)), "gpu_lease_stale_generation"),
    (GpuLeaseReplyV1(status="granted", lease_id="hold-1"), "gpu_lease_stale_generation"),
    (GpuLeaseReplyV1(status="unavailable", reason="verb_not_supported:status"),
     "gpu_lease_unavailable:verb_not_supported:status"),
])
@pytest.mark.asyncio
async def test_validate_hold_ref_fails_closed(reply, reason):
    with pytest.raises(LeaseUnavailable) as err:
        await validate_hold_ref(ScriptedBus(lambda req: reply), REF, source="orion-hub")
    assert err.value.reason == reason


@pytest.mark.asyncio
async def test_validate_hold_ref_refuses_another_runs_hold_without_asking_the_pool():
    bus = ScriptedBus(_pool)
    with pytest.raises(LeaseUnavailable, match="holder_mismatch"):
        await validate_hold_ref(bus, REF, source="orion-hub", expected_holder=durable_run_holder("run-2"))
    assert bus.requests == []


@pytest.mark.asyncio
async def test_validate_hold_ref_unreachable_pool_fails_closed():
    class Down(ScriptedBus):
        async def rpc_request(self, *a: Any, **kw: Any):
            raise asyncio.TimeoutError()

    with pytest.raises(LeaseUnavailable, match="validation_unavailable"):
        await validate_hold_ref(Down(_pool), REF, source="orion-hub")


def test_durable_run_holder_matches_the_spec_shape():
    assert durable_run_holder("abc") == "durable-runs:abc"
    assert DURABLE_RUN_HOLDER_PREFIX == "durable-runs:"
