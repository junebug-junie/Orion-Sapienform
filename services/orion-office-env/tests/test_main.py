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
    assert office_main.office_status()["office_sensor"] == "fresh"
    assert office_main.office_status()["frames_ok"] == 1


def test_make_sample_without_frame_reports_stale_status() -> None:
    s = office_main.make_sample(FrameStore(), get_settings(), now_fn=lambda: NOW)
    assert s.state.stale is True
    assert office_main.office_status()["office_sensor"] == "stale"


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
