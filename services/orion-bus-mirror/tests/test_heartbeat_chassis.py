"""Regression coverage for bus-mirror's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-bus-mirror had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service). This test does not open a real
bus connection or touch sqlite/Falkor -- it verifies (a) `build_heartbeat_chassis()` wires
ChassisConfig from the mirror's own settings, and (b) `mirror_bus()` starts and stops that
chassis around its subscribe loop.

Note: this service is currently disabled in production (see README.md "Status") after a
2026-07-24 unbounded-disk incident -- this patch is purely additive code, it does not
re-enable the container.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.main import build_heartbeat_chassis, mirror_bus
from app.settings import settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_mirror_settings() -> None:
    chassis = build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.SERVICE_NAME
    assert chassis.cfg.service_version == settings.SERVICE_VERSION
    assert chassis.cfg.node_name == settings.NODE_NAME
    assert chassis.cfg.bus_url == settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


class _StopMirror(Exception):
    pass


@pytest.mark.asyncio
async def test_mirror_bus_starts_and_stops_heartbeat_chassis(tmp_path, monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_bus = AsyncMock()
    # bus.subscribe() is an @asynccontextmanager -- calling it returns an async context
    # manager directly (not a coroutine to await), so a plain MagicMock side_effect raises
    # synchronously at the `async with bus.subscribe(...)` call, before entry.
    fake_bus.subscribe = MagicMock(side_effect=RuntimeError("stop before subscribing"))

    monkeypatch.setattr(settings, "MIRROR_PARQUET_DIR", str(tmp_path / "parquet"))
    monkeypatch.setattr(settings, "MIRROR_SQLITE_PATH", str(tmp_path / "mirror.sqlite"))
    monkeypatch.setattr(settings, "MIRROR_GRAPH_ENABLED", False)

    with patch("app.main.OrionBusAsync", return_value=fake_bus), patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        with pytest.raises(RuntimeError, match="stop before subscribing"):
            await mirror_bus()

    fake_chassis.start_background.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()
    fake_bus.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_mirror_bus_survives_heartbeat_start_failure(tmp_path, monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real mirror subscribe loop from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    fake_bus = AsyncMock()
    fake_bus.subscribe = MagicMock(side_effect=RuntimeError("reached subscribe"))

    monkeypatch.setattr(settings, "MIRROR_PARQUET_DIR", str(tmp_path / "parquet"))
    monkeypatch.setattr(settings, "MIRROR_SQLITE_PATH", str(tmp_path / "mirror.sqlite"))
    monkeypatch.setattr(settings, "MIRROR_GRAPH_ENABLED", False)

    with patch("app.main.OrionBusAsync", return_value=fake_bus), patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        with pytest.raises(RuntimeError, match="reached subscribe"):
            await mirror_bus()

    fake_chassis.stop.assert_not_awaited()
    fake_bus.close.assert_awaited_once()
