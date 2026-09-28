from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.zwave_client import METER_CC, METER_W_PROPERTY_KEY, ZWaveJSClient
from orion.schemas.telemetry.home_cooling import CoolingObservedStateV1

T0 = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
W_KEY = f"{METER_CC}-0-value-{METER_W_PROPERTY_KEY}"


def test_state_carries_explicit_stale_flag():
    assert CoolingObservedStateV1().stale is None
    assert CoolingObservedStateV1(stale=True).stale is True


def _client() -> ZWaveJSClient:
    return ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)


@pytest.mark.asyncio
async def test_successful_poll_marks_fresh_and_resets_failures():
    client = _client()
    client.consecutive_poll_failures = 4
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    assert await client.refresh_meter_watts(2) == 885.7
    assert client.last_fresh_at(2) == T0
    assert client.consecutive_poll_failures == 0


@pytest.mark.asyncio
async def test_failed_poll_is_counted_and_never_marks_fresh(caplog):
    client = _client()
    client._values_by_node[2] = {W_KEY: {"commandClass": METER_CC, "property": "value",
                                         "propertyKey": METER_W_PROPERTY_KEY, "value": 850.0}}
    client._request = AsyncMock(side_effect=TimeoutError("no answer"))
    with caplog.at_level("WARNING"):
        assert await client.refresh_meter_watts(2) is None
    assert client.last_fresh_at(2) is None
    assert client.consecutive_poll_failures == 1
    assert "poll_value" in caplog.text
    assert client.get_values(2)[W_KEY]["value"] == 850.0  # cache untouched, but not fresh


@pytest.mark.asyncio
async def test_unsuccessful_result_is_a_failure():
    client = _client()
    client._request = AsyncMock(return_value={"success": False, "errorCode": "node_dead"})
    assert await client.refresh_meter_watts(2) is None
    assert client.last_fresh_at(2) is None
    assert client.consecutive_poll_failures == 1


def test_pushed_watts_event_marks_fresh():
    client = _client()
    client._handle_message({"type": "event", "event": {
        "source": "node", "event": "value updated", "nodeId": 2,
        "args": {"commandClass": METER_CC, "endpoint": 0, "property": "value",
                 "propertyKey": METER_W_PROPERTY_KEY, "newValue": 870.0}}})
    assert client.last_fresh_at(2) == T0


def test_pushed_non_watts_event_does_not_mark_fresh():
    client = _client()
    client._handle_message({"type": "event", "event": {
        "source": "node", "event": "value updated", "nodeId": 2,
        "args": {"commandClass": 37, "endpoint": 0, "property": "currentValue", "newValue": True}}})
    assert client.last_fresh_at(2) is None


def test_not_connected_without_socket():
    assert _client().connected is False
