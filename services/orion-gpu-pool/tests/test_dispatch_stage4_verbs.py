"""Consumer-first for stage 4.1: the pool must accept the widened GpuLeaseRequestV1 and answer the
verbs it has no engine for yet (attach, status) without touching any lease.

Before 4.1 the dispatcher's final ``else`` was cancel, so once the schema admitted ``status`` a
read of a durable run's hold would have *ended* that hold. This pins that it cannot."""
from __future__ import annotations

import pytest

import app.main as main_mod
from orion.schemas.gpu_pool import GpuLeaseReplyV1, GpuLeaseRequestV1


class RecordingRuntime:
    def __init__(self):
        self.calls: list[tuple] = []

    async def acquire(self, req):
        self.calls.append(("acquire", req.request_id))
        return GpuLeaseReplyV1(status="queued", lease_id="new")

    async def heartbeat(self, lease_id):
        self.calls.append(("heartbeat", lease_id))
        return GpuLeaseReplyV1(status="granted", lease_id=lease_id)

    async def release(self, lease_id, outcome="ok", detail=None):
        self.calls.append(("release", lease_id, outcome))
        return GpuLeaseReplyV1(status="ok", lease_id=lease_id)

    async def cancel(self, lease_id):
        self.calls.append(("cancel", lease_id))
        return GpuLeaseReplyV1(status="ok", lease_id=lease_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("req", [
    GpuLeaseRequestV1(verb="status", lease_id="hold-1"),
    GpuLeaseRequestV1(verb="attach", request_id="c1", holder="gateway", work_class="agent",
                      parent_lease_id="hold-1", parent_generation=1),
])
async def test_unsupported_stage4_verbs_touch_nothing(req):
    rt = RecordingRuntime()
    out = await main_mod.dispatch_lease(rt, req)
    assert rt.calls == []
    assert out.status == "unavailable" and out.reason == f"verb_not_supported:{req.verb}"


@pytest.mark.asyncio
@pytest.mark.parametrize("req,call", [
    (GpuLeaseRequestV1(verb="acquire", request_id="r1", work_class="agent"), ("acquire", "r1")),
    (GpuLeaseRequestV1(verb="heartbeat", lease_id="l1"), ("heartbeat", "l1")),
    (GpuLeaseRequestV1(verb="release", lease_id="l1", outcome="timeout"), ("release", "l1", "timeout")),
    (GpuLeaseRequestV1(verb="cancel", lease_id="l1"), ("cancel", "l1")),
])
async def test_existing_verbs_dispatch_unchanged(req, call):
    rt = RecordingRuntime()
    await main_mod.dispatch_lease(rt, req)
    assert rt.calls == [call]


@pytest.mark.asyncio
async def test_missing_lease_id_is_unknown_lease():
    rt = RecordingRuntime()
    out = await main_mod.dispatch_lease(rt, GpuLeaseRequestV1(verb="heartbeat"))
    assert out.status == "unknown_lease" and rt.calls == []
