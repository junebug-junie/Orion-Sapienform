"""Regression test for the bus_synaptic transport trigger notifier.

Prior to this fix, _handle_metacog_trigger built a HubNotificationEvent
using a stale field vocabulary (kind/summary/body/timestamp/dismissible/
priority) that never matched the real schema (notification_id/created_at/
severity/event_kind/source_service/title/body_text), and published the raw
dict instead of a BaseEnvelope -- so every real trigger raised a 5-field
pydantic ValidationError and was silently swallowed by the handler's own
try/except, live in production. This test constructs a real trigger
payload and asserts the notifier both publishes without raising and
produces a wire payload that HubNotificationEvent.model_validate() accepts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import DecodeResult
from orion.schemas.notify import HubNotificationEvent

from scripts.bus_synaptic_trigger_notifier import BusSynapticTriggerNotifier


def _trigger_decode_result() -> DecodeResult:
    envelope = BaseEnvelope(
        kind="metacog.trigger.v1",
        source=ServiceRef(name="orion-equilibrium-service", version="0.1.0", node="athena"),
        payload={
            "trigger_kind": "transport",
            "reason": "bus_synaptic anomaly detected",
            "upstream": {
                "evidence_source": "bus_synaptic_prediction_error",
                "error": 0.87,
            },
        },
    )
    return DecodeResult(envelope=envelope, raw={}, ok=True)


@pytest.mark.asyncio
async def test_bus_synaptic_trigger_publishes_valid_notification_envelope() -> None:
    notifier = BusSynapticTriggerNotifier(notify_channel="orion:notify:in_app")
    notifier._bus = MagicMock()
    notifier._bus.codec.decode = MagicMock(return_value=_trigger_decode_result())
    notifier._bus.publish = AsyncMock()

    await notifier._handle_metacog_trigger({"data": b"irrelevant, decode is mocked"})

    notifier._bus.publish.assert_awaited_once()
    channel, published = notifier._bus.publish.await_args.args
    assert channel == "orion:notify:in_app"
    assert isinstance(published, BaseEnvelope)

    # The actual regression check: this used to raise ValidationError with
    # 5 missing fields for every real trigger.
    event = HubNotificationEvent.model_validate(published.payload)
    assert event.event_kind == "bus_synaptic.transport_trigger"
    assert event.severity == "warning"
    assert event.source_service == "orion-hub"
    assert "0.87" in (event.body_text or "")


@pytest.mark.asyncio
async def test_bus_synaptic_trigger_ignores_non_transport_triggers() -> None:
    notifier = BusSynapticTriggerNotifier(notify_channel="orion:notify:in_app")
    notifier._bus = MagicMock()
    envelope = BaseEnvelope(
        kind="metacog.trigger.v1",
        source=ServiceRef(name="orion-equilibrium-service", version="0.1.0", node="athena"),
        payload={"trigger_kind": "reasoning", "upstream": {}},
    )
    notifier._bus.codec.decode = MagicMock(return_value=DecodeResult(envelope=envelope, raw={}, ok=True))
    notifier._bus.publish = AsyncMock()

    await notifier._handle_metacog_trigger({"data": b"irrelevant, decode is mocked"})

    notifier._bus.publish.assert_not_awaited()
