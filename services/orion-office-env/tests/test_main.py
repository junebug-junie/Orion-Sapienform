from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.main as office_main
from app.serial_reader import FrameStore, parse_line
from app.settings import get_settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly

NOW = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)
GOOD = b'{"schema":"orion.office_env_frame.v1","seq":3,"uptime_ms":5000,"environment":{"temp_c":22.0,"humidity_pct":28.0}}\n'


def test_heartbeat_chassis_uses_office_settings() -> None:
    s = get_settings()
    chassis = office_main.build_heartbeat_chassis(s)
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == "orion-office-env"
    assert chassis.cfg.node_name == s.INSTANCE_ID == "pan-1"
    assert chassis.cfg.health_channel == "orion:system:health"


def test_make_sample_updates_heartbeat_status() -> None:
    store = FrameStore()
    store.put(parse_line(GOOD), NOW)
    s = office_main.make_sample(store, get_settings(), now_fn=lambda: NOW)
    assert s.state.stale is False
    assert office_main.office_status()["sensor"] == "fresh"
    assert office_main.office_status()["frames_ok"] == 1


def test_make_sample_without_frame_reports_stale_status() -> None:
    s = office_main.make_sample(FrameStore(), get_settings(), now_fn=lambda: NOW)
    assert s.state.stale is True
    assert office_main.office_status()["sensor"] == "stale"


@pytest.mark.asyncio
async def test_publish_sample_sends_registered_kind_on_configured_channel() -> None:
    store = FrameStore()
    store.put(parse_line(GOOD), NOW)
    settings = get_settings()
    sample = office_main.make_sample(store, settings, now_fn=lambda: NOW)
    bus = MagicMock()
    bus.enabled = True
    bus.publish = AsyncMock()
    await office_main.publish_sample(bus, settings, sample)
    channel, envelope = bus.publish.await_args.args
    assert channel == "orion:office:env:sample"
    assert envelope.kind == "office.env.sample.v1"
    assert envelope.payload["measurements"]["office_temp_c"] == 22.0


CLOSET = b'{"schema":"orion.office_env_frame.v1","seq":4,"uptime_ms":5000,"co2":{"co2_ppm":612,"temp_c":29.1,"humidity_pct":23.0}}\n'


@pytest.mark.asyncio
async def test_closet_room_publishes_closet_kind_on_closet_channel() -> None:
    store = FrameStore()
    store.put(parse_line(CLOSET), NOW)
    settings = get_settings().model_copy(update={"ENV_ROOM": "closet", "SERVICE_NAME": "orion-closet-env"})
    sample = office_main.make_sample(store, settings, now_fn=lambda: NOW)
    assert office_main.office_status()["room"] == "closet"
    bus = MagicMock()
    bus.enabled = True
    bus.publish = AsyncMock()
    await office_main.publish_sample(bus, settings, sample)
    channel, envelope = bus.publish.await_args.args
    assert channel == "orion:closet:env:sample"
    assert envelope.kind == "closet.env.sample.v1"
    assert envelope.source.name == "orion-closet-env"
    assert envelope.payload["sensor"]["id"] == "scd40-closet"
    assert envelope.payload["measurements"] == {
        "closet_co2_ppm": 612, "closet_temp_c": 29.1, "closet_humidity_pct": 23.0,
    }


@pytest.mark.asyncio
async def test_office_payload_stays_valid_for_a_pre_pm_light_consumer() -> None:
    """A sample with no PM/lux readings must not carry the new keys at all, so a sql-writer
    whose extra="forbid" model predates them still accepts it (deploy-order safety)."""
    store = FrameStore()
    store.put(parse_line(GOOD), NOW)
    settings = get_settings()
    bus = MagicMock()
    bus.enabled = True
    bus.publish = AsyncMock()
    await office_main.publish_sample(bus, settings, office_main.make_sample(store, settings, now_fn=lambda: NOW))
    payload = bus.publish.await_args.args[1].payload
    new_keys = {"office_pm1_ug_m3", "office_pm25_ug_m3", "office_pm10_ug_m3", "office_lux", "office_sound_dba"}
    assert not new_keys & payload["measurements"].keys()
    assert "light_saturated" not in payload["state"]
    assert payload["measurements"] == {"office_temp_c": 22.0, "office_humidity_pct": 28.0}
