"""Regression coverage for substrate-runtime's bus-native SystemHealthV1 heartbeat wiring.

Part of the pilot-5 rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-substrate-runtime had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service) -- it does have a pre-existing HTTP
/health route, which per the design doc is a separate, parallel mechanism this patch does not
touch. `worker` and the finalize/closure/goal-context listeners are mocked out here so this test
exercises only the new heartbeat wiring in app/main.py's lifespan, without opening a real bus
connection or touching worker.py's own task list (see
test_worker_independent_reducers.py::test_start_spawns_independent_reducer_poll_tasks, which
asserts an exact count of worker-owned tasks -- the new heartbeat chassis is deliberately kept
outside that list).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import HeartbeatOnly

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_main(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")
    monkeypatch.setenv(
        "NODE_CATALOG_PATH",
        str(REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"),
    )
    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main

    return main


def test_build_heartbeat_chassis_uses_runtime_settings(monkeypatch) -> None:
    main = _import_main(monkeypatch)
    import app.settings as settings_mod

    chassis = main.build_heartbeat_chassis()
    s = settings_mod.get_settings()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.service_name
    assert chassis.cfg.service_version == s.service_version
    assert chassis.cfg.node_name == s.node_name
    assert chassis.cfg.bus_url == s.orion_bus_url
    assert chassis.cfg.bus_enabled == s.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == s.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(main.worker, "start", AsyncMock())
    monkeypatch.setattr(main.worker, "stop", AsyncMock())
    monkeypatch.setattr(main.worker, "_bus", None)  # skip the finalize/closure/goal-context listeners

    app = FastAPI()
    async with main.lifespan(app):
        assert main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert main.heartbeat_chassis is None
    main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real substrate worker from starting."""
    main = _import_main(monkeypatch)

    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(main.worker, "start", AsyncMock())
    monkeypatch.setattr(main.worker, "stop", AsyncMock())
    monkeypatch.setattr(main.worker, "_bus", None)

    app = FastAPI()
    async with main.lifespan(app):
        assert main.heartbeat_chassis is None
        main.worker.start.assert_awaited_once()
