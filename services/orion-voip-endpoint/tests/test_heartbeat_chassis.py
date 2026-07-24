"""Regression coverage for voip-endpoint's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-voip-endpoint had no bus-native heartbeat before
this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service) and no
tests/ directory at all -- this is the first test file for this service.

Structurally the most divergent of the 12-service batch: `app/main.py:run()` is a synchronous
entrypoint that bootstraps Asterisk/TFTP then calls a blocking `uvicorn.run(app, ...)`, not an
async FastAPI `lifespan`. The heartbeat is wired via `@app.on_event("startup")`/`("shutdown")`
on the FastAPI app `create_app()` returns in `app/http_api.py` -- uvicorn still drives those
events the same way it would drive `lifespan`, regardless of the synchronous outer `run()` call.
`app/settings.py`'s `Settings` uses `env_prefix = "VOIP_"` for its VOIP_* fields, so the new
service_name/service_version/node_name/orion_bus_url/heartbeat_interval_sec fields use an
explicit `alias=` to opt out of that prefix and match this repo's general
SERVICE_NAME/SERVICE_VERSION/NODE_NAME/ORION_BUS_URL/HEARTBEAT_INTERVAL_SEC convention.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.http_api import build_heartbeat_chassis, create_app
from app.settings import Settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def _make_settings(**overrides) -> Settings:
    base = dict(
        lan_host_ip="192.168.1.66",
        tailscale_host_ip="100.92.216.81",
        phone_mac="AC:44:F2:1C:7D:D5",
    )
    base.update(overrides)
    return Settings(**base)


def test_build_heartbeat_chassis_uses_voip_settings() -> None:
    settings = _make_settings()
    chassis = build_heartbeat_chassis(settings)
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.service_name
    assert chassis.cfg.service_version == settings.service_version
    assert chassis.cfg.node_name == settings.node_name
    assert chassis.cfg.bus_url == settings.orion_bus_url
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


def test_settings_identity_fields_bypass_voip_prefix() -> None:
    """service_name/service_version/node_name/orion_bus_url/heartbeat_interval_sec must read
    from the bare (non-VOIP_-prefixed) env var names, matching every other service in this
    rollout -- not VOIP_SERVICE_NAME etc. Construction here uses the fields' explicit
    aliases (SERVICE_NAME etc.), matching how pydantic-settings actually populates them from
    real env vars -- unlike lan_host_ip/phone_mac above, these fields opt out of the
    env_prefix via `alias=`, so BaseSettings (extra="forbid" by default here) does not accept
    the bare field name as a constructor kwarg, only the alias."""
    settings = _make_settings(
        SERVICE_NAME="orion-voip-endpoint",
        NODE_NAME="athena",
        ORION_BUS_URL="redis://100.92.216.81:6379/0",
        HEARTBEAT_INTERVAL_SEC=15.0,
    )
    assert settings.service_name == "orion-voip-endpoint"
    assert settings.node_name == "athena"
    assert settings.orion_bus_url == "redis://100.92.216.81:6379/0"
    assert settings.heartbeat_interval_sec == 15.0


def test_settings_reads_bare_env_vars_bypassing_voip_prefix(monkeypatch) -> None:
    """End-to-end confirmation (not just constructor kwargs) that the alias= fields actually
    read from real, bare env vars at Settings() instantiation time -- the scenario that
    matters in production (docker-compose passes these as plain env vars, not Python kwargs)."""
    monkeypatch.setenv("VOIP_LAN_HOST_IP", "192.168.1.66")
    monkeypatch.setenv("VOIP_PHONE_MAC", "AC:44:F2:1C:7D:D5")
    monkeypatch.setenv("SERVICE_NAME", "voip-endpoint")
    monkeypatch.setenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SEC", "7")

    settings = Settings()
    assert settings.service_name == "voip-endpoint"
    assert settings.orion_bus_url == "redis://100.92.216.81:6379/0"
    assert settings.heartbeat_interval_sec == 7.0


def test_startup_and_shutdown_events_manage_heartbeat_chassis(monkeypatch) -> None:
    settings = _make_settings()
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(
        "app.http_api.build_heartbeat_chassis", lambda _settings: fake_chassis
    )

    app = create_app(
        settings,
        bus_publish=lambda *a, **k: None,
        do_echo_call=lambda: {"status": "ok"},
        do_page_echo=lambda: {"status": "ok"},
    )

    with TestClient(app):
        assert app.state.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert app.state.heartbeat_chassis is None


def test_startup_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of the app from starting."""
    settings = _make_settings()
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(
        "app.http_api.build_heartbeat_chassis", lambda _settings: fake_chassis
    )

    app = create_app(
        settings,
        bus_publish=lambda *a, **k: None,
        do_echo_call=lambda: {"status": "ok"},
        do_page_echo=lambda: {"status": "ok"},
    )

    with TestClient(app) as client:
        assert app.state.heartbeat_chassis is None
        # The rest of the app (unrelated routes) must still be reachable.
        response = client.get("/openapi.json")
        assert response.status_code == 200
