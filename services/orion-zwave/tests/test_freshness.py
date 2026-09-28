from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

import app.main as zwave_main
from app.main import build_cooling_sample
from app.settings import Settings
from app.zwave_client import (
    METER_A_PROPERTY_KEY,
    METER_CC,
    METER_V_PROPERTY_KEY,
    METER_W_PROPERTY_KEY,
    ZWaveJSClient,
)
from orion.schemas.telemetry.home_cooling import CoolingObservedStateV1

T0 = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
W_KEY = f"{METER_CC}-0-value-{METER_W_PROPERTY_KEY}"
V_KEY = f"{METER_CC}-0-value-{METER_V_PROPERTY_KEY}"
A_KEY = f"{METER_CC}-0-value-{METER_A_PROPERTY_KEY}"


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


def _sample(last_fresh_at, now=T0 + timedelta(seconds=30)):
    return build_cooling_sample(
        node_id=2, controller_ready=True, device_online=True,
        watts=850.0, volts=121.0, amps=7.0, switch_on=True,
        device_path="/dev/zwave", product="Shelly Wave Plug",
        now=now, last_fresh_at=last_fresh_at, stale_after_sec=120.0,
    )


def test_fresh_sample_keeps_readings_and_reports_age():
    s = _sample(T0)
    assert s.state.stale is False
    assert s.measurements.cooling_watts == 850.0
    assert s.state.switch_on is True
    assert s.provenance.sample_age_sec == 30.0


def test_stale_sample_omits_every_cached_reading():
    s = _sample(T0, now=T0 + timedelta(seconds=121))
    assert s.state.stale is True
    assert s.measurements.cooling_watts is None
    assert s.measurements.cooling_volts is None
    assert s.measurements.cooling_amps is None
    assert s.state.switch_on is None
    assert s.provenance.sample_age_sec == 121.0


def test_never_fresh_is_stale_with_unknown_age():
    s = _sample(None)
    assert s.state.stale is True
    assert s.measurements.cooling_watts is None
    assert s.provenance.sample_age_sec is None


@pytest.mark.asyncio
async def test_poll_once_after_silence_publishes_stale_not_cache():
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)
    client._values_by_node[2] = {W_KEY: {"commandClass": METER_CC, "property": "value",
                                         "propertyKey": METER_W_PROPERTY_KEY, "value": 850.0}}
    client._last_fresh_at[2] = T0
    client._request = AsyncMock(side_effect=TimeoutError("plug gone"))
    settings = Settings(ZWAVE_NODE_ID=2, COOLING_STALE_AFTER_SEC=120.0)
    sample = await zwave_main.poll_once(client, settings, now_fn=lambda: T0 + timedelta(seconds=300))
    assert sample.state.stale is True
    assert sample.measurements.cooling_watts is None
    assert zwave_main.cooling_status()["cooling_sensor"] == "stale"
    assert zwave_main.cooling_status()["consecutive_poll_failures"] == 1


@pytest.mark.asyncio
async def test_poll_once_fresh_poll_publishes_reading():
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: T0)
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    settings = Settings(ZWAVE_NODE_ID=2, COOLING_STALE_AFTER_SEC=120.0)
    sample = await zwave_main.poll_once(client, settings, now_fn=lambda: T0 + timedelta(seconds=1))
    assert sample.state.stale is False
    assert sample.measurements.cooling_watts == 885.7
    assert zwave_main.cooling_status()["cooling_sensor"] == "fresh"


def test_heartbeat_carries_cooling_status():
    chassis = zwave_main.build_heartbeat_chassis(Settings())
    assert chassis._heartbeat_details is not None
    assert "cooling_sensor" in chassis._heartbeat_details()


def test_stale_window_setting_default():
    assert Settings(_env_file=None).COOLING_STALE_AFTER_SEC == 120.0


# --- Fix round 1 -------------------------------------------------------------


def _meter_event(property_key: int, value: float, node_id: int = 2) -> dict:
    return {"type": "event", "event": {
        "source": "node", "event": "value updated", "nodeId": node_id,
        "args": {"commandClass": METER_CC, "endpoint": 0, "property": "value",
                 "propertyKey": property_key, "newValue": value}}}


def _cached_meter(property_key: int, value: float) -> dict:
    return {"commandClass": METER_CC, "endpoint": 0, "property": "value",
            "propertyKey": property_key, "value": value}


def _settings() -> Settings:
    return Settings(_env_file=None, ZWAVE_NODE_ID=2, COOLING_STALE_AFTER_SEC=120.0)


@pytest.mark.asyncio
async def test_fresh_watts_with_stale_cached_volts_omits_volts():
    client = _client()
    client._values_by_node[2] = {V_KEY: _cached_meter(METER_V_PROPERTY_KEY, 121.0),
                                 A_KEY: _cached_meter(METER_A_PROPERTY_KEY, 7.0)}
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    sample = await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0)
    assert sample.state.stale is False
    assert sample.measurements.cooling_watts == 885.7
    assert sample.measurements.cooling_volts is None
    assert sample.measurements.cooling_amps is None


@pytest.mark.asyncio
async def test_pushed_volts_and_amps_events_are_included():
    client = _client()
    client._handle_message(_meter_event(METER_V_PROPERTY_KEY, 121.0))
    client._handle_message(_meter_event(METER_A_PROPERTY_KEY, 7.0))
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    sample = await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0 + timedelta(seconds=5))
    assert sample.measurements.cooling_watts == 885.7
    assert sample.measurements.cooling_volts == 121.0
    assert sample.measurements.cooling_amps == 7.0


@pytest.mark.asyncio
async def test_pushed_volts_older_than_window_are_omitted_even_when_watts_fresh():
    clock = [T0]
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: clock[0])
    client._handle_message(_meter_event(METER_V_PROPERTY_KEY, 121.0))
    clock[0] = T0 + timedelta(seconds=200)
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    sample = await zwave_main.poll_once(client, _settings(), now_fn=lambda: clock[0])
    assert sample.measurements.cooling_watts == 885.7
    assert sample.measurements.cooling_volts is None


def test_bootstrap_snapshot_marks_nothing_fresh():
    client = _client()
    client._ingest_listening_result({"success": True, "result": {"state": {"nodes": [{
        "nodeId": 2, "ready": True, "label": "Shelly Wave Plug",
        "values": [_cached_meter(METER_W_PROPERTY_KEY, 850.0),
                   _cached_meter(METER_V_PROPERTY_KEY, 121.0)],
    }]}}})
    assert client.last_fresh_at(2) is None
    assert client.fresh_values(2, now=T0, max_age_sec=120.0) == {}
    assert client.get_values(2)[V_KEY]["value"] == 121.0


def test_pushed_volts_do_not_count_as_watts_freshness():
    client = _client()
    client._handle_message(_meter_event(METER_V_PROPERTY_KEY, 121.0))
    assert client.last_fresh_at(2) is None
    assert V_KEY in client.fresh_values(2, now=T0, max_age_sec=120.0)


@pytest.mark.asyncio
async def test_close_forgets_controller_ready_and_device_online():
    client = _client()
    client._controller_ready = True
    client._device_online[2] = True
    await client.close()
    assert client.controller_ready is False
    assert client.device_online(2) is False


@pytest.mark.asyncio
async def test_close_drains_queued_messages_so_they_cannot_replay():
    client = _client()
    client._handle_message(_meter_event(METER_W_PROPERTY_KEY, 860.0))
    client._inbound.put_nowait(_meter_event(METER_W_PROPERTY_KEY, 870.0))
    assert client.last_fresh_at(2) == T0
    await client.close()
    assert client._inbound.empty()
    # Neither the live-era per-value stamp nor the queued event may leave a value looking fresh.
    assert client.fresh_values(2, now=T0, max_age_sec=120.0) == {}


@pytest.mark.asyncio
async def test_close_forgets_value_freshness_but_keeps_last_answer_time():
    client = _client()
    client._handle_message(_meter_event(METER_W_PROPERTY_KEY, 870.0))
    client._handle_message(_meter_event(METER_V_PROPERTY_KEY, 121.0))
    assert client.fresh_values(2, now=T0, max_age_sec=120.0)
    await client.close()
    assert client.fresh_values(2, now=T0, max_age_sec=120.0) == {}
    assert client.last_fresh_at(2) == T0


def test_recent_answer_without_fresh_watts_is_stale():
    s = build_cooling_sample(
        node_id=2, controller_ready=True, device_online=True,
        watts=None, volts=121.0, amps=7.0, switch_on=True,
        device_path="/dev/zwave", product="Shelly Wave Plug",
        now=T0 + timedelta(seconds=5), last_fresh_at=T0, stale_after_sec=120.0,
    )
    assert s.state.stale is True
    assert s.measurements.cooling_volts is None
    assert s.state.switch_on is None
    assert s.provenance.sample_age_sec == 5.0


@pytest.mark.asyncio
async def test_reconnect_snapshot_after_close_is_not_published_as_fresh():
    client = _client()
    client._handle_message(_meter_event(METER_W_PROPERTY_KEY, 870.0))
    await client.close()
    client._ingest_listening_result({"success": True, "result": {"state": {"nodes": [{
        "nodeId": 2, "ready": True, "values": [_cached_meter(METER_W_PROPERTY_KEY, 850.0)],
    }]}}})
    client._request = AsyncMock(side_effect=TimeoutError("plug gone"))
    sample = await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0 + timedelta(seconds=5))
    assert sample.state.stale is True
    assert sample.measurements.cooling_watts is None
    # Outage age survives the reconnect so the Hub can say "stale since HH:MM".
    assert sample.provenance.sample_age_sec == 5.0


@pytest.mark.asyncio
async def test_request_without_socket_raises_connection_error():
    with pytest.raises(ConnectionError, match="not connected"):
        await _client()._request("node.poll_value")


@pytest.mark.asyncio
async def test_poll_cycle_reconnects_and_still_publishes_stale_when_connect_fails(caplog, monkeypatch):
    client = _client()
    calls: list[str] = []
    client.close = AsyncMock(side_effect=lambda: calls.append("close"))

    async def _refuse() -> None:
        calls.append("connect")
        raise OSError("connection refused")

    client.connect = _refuse
    publish = AsyncMock()
    monkeypatch.setattr(zwave_main, "_publish_sample", publish)
    with caplog.at_level("WARNING"):
        await zwave_main.run_poll_cycle(client, object(), _settings())
    assert calls == ["close", "connect"]
    assert "connection refused" in caplog.text
    connect_logs = [r for r in caplog.records if "zwave_js_connect_failed" in r.getMessage()]
    assert connect_logs and connect_logs[0].levelname == "WARNING" and connect_logs[0].exc_info is None
    publish.assert_awaited_once()
    sample = publish.await_args.args[2]
    assert sample.state.stale is True
    assert sample.measurements.cooling_watts is None


@pytest.mark.asyncio
async def test_poll_cycle_skips_reconnect_while_connected(monkeypatch):
    # run_poll_cycle reads the real clock, so the client must stamp on it too.
    client = ZWaveJSClient("ws://test", node_id=2)
    client.close = AsyncMock()
    client.connect = AsyncMock()
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    publish = AsyncMock()
    monkeypatch.setattr(zwave_main, "_publish_sample", publish)
    with patch.object(ZWaveJSClient, "connected", new_callable=PropertyMock, return_value=True):
        await zwave_main.run_poll_cycle(client, object(), _settings())
    client.close.assert_not_awaited()
    client.connect.assert_not_awaited()
    publish.assert_awaited_once()
    sample = publish.await_args.args[2]
    assert sample.state.stale is False
    assert sample.measurements.cooling_watts == 885.7


@pytest.mark.asyncio
async def test_poll_cycle_bounds_hanging_reconnect_and_still_publishes_stale(caplog, monkeypatch):
    client = _client()
    closes: list[str] = []
    client.close = AsyncMock(side_effect=lambda: closes.append("close"))
    connect_cancelled = asyncio.Event()

    async def _hang() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            connect_cancelled.set()
            raise

    client.connect = _hang
    publish = AsyncMock()
    monkeypatch.setattr(zwave_main, "_publish_sample", publish)
    monkeypatch.setattr(zwave_main, "RECONNECT_TIMEOUT_SEC", 0.05)
    with caplog.at_level("WARNING"):
        await asyncio.wait_for(zwave_main.run_poll_cycle(client, object(), _settings()), timeout=2)
    assert connect_cancelled.is_set()
    assert closes == ["close", "close"]  # pre-connect reset, then cleanup after the timeout
    assert "zwave_js_connect_failed" in caplog.text
    publish.assert_awaited_once()
    sample = publish.await_args.args[2]
    assert sample.state.stale is True
    assert sample.measurements.cooling_watts is None


@pytest.mark.asyncio
async def test_poll_once_ignores_second_stale_watts_like_entry():
    client = _client()
    # A snapshot-only watts-like label (never polled/pushed) sits ahead of the polled key.
    client._values_by_node[2] = {
        "50-0-value-power": {"commandClass": METER_CC, "endpoint": 0, "property": "value",
                             "propertyKeyName": "Power (W)", "value": 12.0},
    }
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    sample = await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0)
    assert sample.state.stale is False
    assert sample.measurements.cooling_watts == 885.7


@pytest.mark.asyncio
async def test_switch_state_kept_while_watts_fresh_dropped_when_stale():
    clock = [T0]
    client = ZWaveJSClient("ws://test", node_id=2, now_fn=lambda: clock[0])
    client._values_by_node[2] = {"37-0-currentValue": {"commandClass": 37, "endpoint": 0,
                                                       "property": "currentValue", "value": True}}
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    fresh = await zwave_main.poll_once(client, _settings(), now_fn=lambda: clock[0])
    assert fresh.state.switch_on is True
    clock[0] = T0 + timedelta(seconds=300)
    client._request = AsyncMock(side_effect=TimeoutError("plug gone"))
    stale = await zwave_main.poll_once(client, _settings(), now_fn=lambda: clock[0])
    assert stale.state.stale is True
    assert stale.state.switch_on is None


@pytest.mark.asyncio
async def test_poll_once_failure_sets_heartbeat_status_to_error(monkeypatch):
    client = _client()
    client._request = AsyncMock(return_value={"success": True, "result": {"value": 885.7}})
    await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0)
    assert zwave_main.cooling_status()["cooling_sensor"] == "fresh"

    def _boom(**_kwargs):
        raise ValueError("sample build failed")

    monkeypatch.setattr(zwave_main, "build_cooling_sample", _boom)
    client._request = AsyncMock(side_effect=TimeoutError("plug gone"))
    with pytest.raises(ValueError):
        await zwave_main.poll_once(client, _settings(), now_fn=lambda: T0)
    status = zwave_main.cooling_status()
    assert status["cooling_sensor"] == "error"
    assert status["consecutive_poll_failures"] == 1
    assert status["zwave_connected"] is False
