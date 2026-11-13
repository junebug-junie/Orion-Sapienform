"""Hub consumer for "reflect on yourself" (2026-10-30 fold of self-study into curiosity).

cortex-orch's self_review workflow publishes a CuriositySelfInquiryRequestV1 on
orion:curiosity:self_inquiry:request; Hub answers with exactly one forced self-inquiry tick
(the same call as POST /api/self-inquiry/run-now). Bad payloads never start a run.
"""

from __future__ import annotations

import asyncio

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.curiosity_self_inquiry_request import (
    SELF_INQUIRY_REQUEST_KIND,
    CuriositySelfInquiryRequestV1,
)
from scripts.curiosity_self_inquiry_request import handle_self_inquiry_request


class _Bus:
    def __init__(self) -> None:
        self.codec = OrionCodec()


class _Loop:
    def __init__(self, reason=None) -> None:
        self.calls: list[bool] = []
        self._reason = reason

    async def tick_self_inquiry(self, *, force: bool = False, now=None, force_reasons=None):
        self.calls.append(force)
        self.force_reasons = force_reasons
        return self._reason


def _msg(bus: _Bus, payload: dict) -> dict:
    env = BaseEnvelope(
        kind=SELF_INQUIRY_REQUEST_KIND,
        source=ServiceRef(name="orion-cortex-orch"),
        payload=payload,
    )
    return {"data": bus.codec.encode(env)}


def test_a_valid_request_takes_exactly_one_forced_self_inquiry_tick() -> None:
    bus, loop = _Bus(), _Loop()
    request = CuriositySelfInquiryRequestV1(requested_by="workflow:self_review", correlation_id="c-1")
    reason = asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, request.model_dump(mode="json"))))
    assert loop.calls == [True]
    assert reason is None


def test_the_handler_passes_only_cooldown_and_cap_as_forceable_and_reports_the_block() -> None:
    """The real gate behaviour (strain still holds a chat-requested run) is pinned in
    test_curiosity_self_inquiry.py; here: the handler narrows force and returns the block."""
    bus, loop = _Bus(), _Loop(reason="disabled")
    request = CuriositySelfInquiryRequestV1(requested_by="workflow:self_review")
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, request.model_dump(mode="json")))) == "disabled"
    assert loop.calls == [True]
    assert loop.force_reasons == frozenset({"cooldown", "daily_cap"})


def test_the_listener_loop_is_cancelled_with_its_children() -> None:
    from contextlib import asynccontextmanager

    from scripts.curiosity_self_inquiry_request import self_inquiry_request_loop

    started = asyncio.Event()

    class _SubBus(_Bus):
        @asynccontextmanager
        async def subscribe(self, channel):
            yield object()

        async def iter_messages(self, pubsub):
            started.set()
            while True:
                await asyncio.sleep(3600)
                yield {}

    async def scenario():
        task = asyncio.create_task(self_inquiry_request_loop(_SubBus(), _Loop()))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled() or task.done()

    asyncio.run(scenario())


def test_an_invalid_payload_starts_nothing() -> None:
    bus, loop = _Bus(), _Loop()
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, {"requested_by": ""}))) == "invalid"
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, {"unexpected": 1}))) == "invalid"
    assert asyncio.run(handle_self_inquiry_request(bus, loop, {"data": b"not an envelope"})) == "invalid"
    assert loop.calls == []
