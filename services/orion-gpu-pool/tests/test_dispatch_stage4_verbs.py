"""Stage 4 verbs reach their own handlers and nothing else.

Before 4.1 the dispatcher's final ``else`` was cancel, so once the schema admitted ``status`` a
read of a durable run's hold would have *ended* that hold. 4.1 answered attach/status
``verb_not_supported``; 4.3 builds them. This pins that each goes to its own handler only."""
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

    async def status(self, lease_id):
        self.calls.append(("status", lease_id))
        return GpuLeaseReplyV1(status="granted", lease_id=lease_id)

    async def attach(self, req):
        self.calls.append(("attach", req.hold_lease_id, req.hold_generation))
        return GpuLeaseReplyV1(status="queued", lease_id="child")


@pytest.mark.asyncio
@pytest.mark.parametrize("req,call", [
    (GpuLeaseRequestV1(verb="status", lease_id="hold-1"), ("status", "hold-1")),
    (GpuLeaseRequestV1(verb="attach", request_id="c1", holder="gateway", work_class="agent",
                       hold_lease_id="hold-1", hold_generation=1), ("attach", "hold-1", 1)),
])
async def test_stage4_verbs_reach_only_their_own_handler(req, call):
    rt = RecordingRuntime()
    await main_mod.dispatch_lease(rt, req)
    assert rt.calls == [call]   # never release/cancel


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
