from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.chdir(Path(__file__).resolve().parents[1])

from app.main import app


@pytest.fixture
def client() -> TestClient:
    with patch("app.main.service.start", new=AsyncMock()), patch("app.main.service.stop", new=AsyncMock()):
        with TestClient(app) as test_client:
            yield test_client


def test_ready_503_when_bus_not_started(client: TestClient) -> None:
    with patch("app.main.service") as mock_service:
        mock_service.bus = MagicMock()
        mock_service.bus.enabled = False
        mock_service.settings.pad_rpc_request_channel = "orion:pad:rpc:request"
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


def test_ready_200_when_consumer_ready(client: TestClient) -> None:
    fake_redis = MagicMock()
    rpc_task = MagicMock()
    rpc_task.done.return_value = False
    with patch("app.main.service") as mock_service, patch(
        "app.main.check_bus_consumer_readiness",
        new=AsyncMock(
            return_value=MagicMock(
                ok=True,
                bus_consumer_ready=True,
                subscriber_count=1,
                intake_channel="orion:pad:rpc:request",
                dependency_status="available",
                error=None,
                heartbeat_fresh=True,
                rpc_smoke_ok=True,
            )
        ),
    ), patch("app.main.bus_consumer_readiness_v1") as mock_v1:
        mock_service.bus = MagicMock(enabled=True, redis=fake_redis)
        mock_service.settings.pad_rpc_request_channel = "orion:pad:rpc:request"
        mock_service.settings.app_name = "landing-pad"
        mock_service.settings.health_channel = "orion:system:health"
        mock_service.settings.heartbeat_interval_sec = 10
        mock_service.rpc._task = rpc_task
        mock_v1.return_value = MagicMock(ok=True, model_dump=lambda mode: {"ok": True})
        resp = client.get("/ready")
    assert resp.status_code == 200
