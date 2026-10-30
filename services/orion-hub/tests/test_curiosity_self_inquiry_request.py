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

    async def tick_self_inquiry(self, *, force: bool = False, now=None):
        self.calls.append(force)
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


def test_the_ticks_own_gate_still_decides() -> None:
    """force skips cooldown and cap only; a disabled line or a health gate still blocks,
    and the handler reports that block instead of pretending a run started."""
    bus, loop = _Bus(), _Loop(reason="disabled")
    request = CuriositySelfInquiryRequestV1(requested_by="workflow:self_review")
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, request.model_dump(mode="json")))) == "disabled"
    assert loop.calls == [True]


def test_an_invalid_payload_starts_nothing() -> None:
    bus, loop = _Bus(), _Loop()
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, {"requested_by": ""}))) == "invalid"
    assert asyncio.run(handle_self_inquiry_request(bus, loop, _msg(bus, {"unexpected": 1}))) == "invalid"
    assert asyncio.run(handle_self_inquiry_request(bus, loop, {"data": b"not an envelope"})) == "invalid"
    assert loop.calls == []
