"""`--publish` on the curiosity-supervisor readings report: opt-in, one bus
event per reading, best-effort (one publish failure doesn't drop the rest).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.curiosity_supervisor import READING_CHANNEL, READING_KIND, HopReadingV1
from scripts.report_curiosity_supervisor_readings import _publish_readings


class _FakeBus:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.published: list[tuple[str, object]] = []
        self.fail_on = fail_on or set()

    async def publish(self, channel, envelope):
        # BaseEnvelope parses correlation_id into a real UUID -- compare on
        # .hex, the same form _reading()'s callers pass in.
        if envelope.correlation_id.hex in self.fail_on:
            raise RuntimeError("simulated bus failure")
        self.published.append((channel, envelope))


def _reading(reading_id: str, n: int) -> HopReadingV1:
    return HopReadingV1(
        reading_id=reading_id,
        hop_run_id="r1",
        hop_n=n,
        kind="test",
        reading_confidence=0.5,
        reasoning="x",
    )


@pytest.mark.asyncio
async def test_publishes_one_event_per_reading_on_the_reading_channel():
    bus = _FakeBus()
    id_a, id_b = uuid4().hex, uuid4().hex
    readings = [_reading(id_a, 1), _reading(id_b, 2)]
    source = ServiceRef(name="test", node="x", version="0.0.1")
    published = await _publish_readings(bus, readings, source=source)
    assert published == 2
    assert [c for c, _ in bus.published] == [READING_CHANNEL, READING_CHANNEL]
    envelope = bus.published[0][1]
    assert envelope.kind == READING_KIND
    assert envelope.correlation_id.hex == id_a
    assert envelope.payload["hop_n"] == 1


@pytest.mark.asyncio
async def test_one_publish_failure_does_not_drop_the_rest():
    id_a, id_b, id_c = uuid4().hex, uuid4().hex, uuid4().hex
    bus = _FakeBus(fail_on={id_b})
    readings = [_reading(id_a, 1), _reading(id_b, 2), _reading(id_c, 3)]
    source = ServiceRef(name="test", node="x", version="0.0.1")
    published = await _publish_readings(bus, readings, source=source)
    assert published == 2
    assert [e.correlation_id.hex for _, e in bus.published] == [id_a, id_c]
