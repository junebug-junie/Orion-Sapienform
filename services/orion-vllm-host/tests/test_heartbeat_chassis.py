"""Regression coverage for vllm-host's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-vllm-host had no bus connection at all before this patch (confirmed via
`grep -rl "BaseChassis\\|SystemHealthV1"` returning nothing for this service) -- its `main()`
was a purely synchronous subprocess launcher (`subprocess.run` on the vLLM OpenAI server). This
test does not launch a real subprocess or open a real bus connection -- it verifies (a)
`build_heartbeat_chassis()` wires ChassisConfig from the service's own settings, and (b)
`_main_async()` starts and stops that chassis around the (mocked) blocking launch flow.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as vllm_main
from app.settings import settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_vllm_settings() -> None:
    chassis = vllm_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.service_name
    assert chassis.cfg.service_version == settings.service_version
    assert chassis.cfg.node_name == settings.node_name
    assert chassis.cfg.bus_url == settings.orion_bus_url
    assert chassis.cfg.bus_enabled == settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_main_async_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(vllm_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(vllm_main, "run_vllm_server_blocking", lambda: None)

    await vllm_main._main_async()

    fake_chassis.start_background.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_async_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real vLLM launch from running."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(vllm_main, "build_heartbeat_chassis", lambda: fake_chassis)

    launched = []
    monkeypatch.setattr(vllm_main, "run_vllm_server_blocking", lambda: launched.append(True))

    await vllm_main._main_async()

    assert launched == [True]
    fake_chassis.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_async_stops_heartbeat_even_if_server_launch_raises(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(vllm_main, "build_heartbeat_chassis", lambda: fake_chassis)

    def _boom():
        raise RuntimeError("vllm server crashed")

    monkeypatch.setattr(vllm_main, "run_vllm_server_blocking", _boom)

    with pytest.raises(RuntimeError, match="vllm server crashed"):
        await vllm_main._main_async()

    fake_chassis.start_background.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()
