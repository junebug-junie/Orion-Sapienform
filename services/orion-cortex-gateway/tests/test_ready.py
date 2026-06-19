from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ready_503_when_intake_bus_not_connected(client: TestClient) -> None:
    with patch("app.main.bus_client") as mock_client, patch("app.main.settings") as mock_settings:
        mock_client._intake_bus = None
        mock_settings.channel_gateway_request = "orion:cortex:gateway:request"
        mock_settings.service_name = "cortex-gateway"
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


def test_ready_200_when_consumer_ready(client: TestClient) -> None:
    fake_redis = MagicMock()
    intake_bus = MagicMock(enabled=True, redis=fake_redis)
    with patch("app.main.bus_client") as mock_client, patch("app.main.settings") as mock_settings, patch(
        "app.main.check_bus_consumer_readiness",
        new=AsyncMock(
            return_value=MagicMock(
                ok=True,
                bus_consumer_ready=True,
                subscriber_count=1,
                intake_channel="orion:cortex:gateway:request",
                dependency_status="available",
                error=None,
            )
        ),
    ), patch("app.main.bus_consumer_readiness_v1") as mock_v1:
        mock_client._intake_bus = intake_bus
        mock_settings.channel_gateway_request = "orion:cortex:gateway:request"
        mock_settings.service_name = "cortex-gateway"
        mock_v1.return_value = MagicMock(ok=True, model_dump=lambda mode: {"ok": True})
        resp = client.get("/ready")
    assert resp.status_code == 200
