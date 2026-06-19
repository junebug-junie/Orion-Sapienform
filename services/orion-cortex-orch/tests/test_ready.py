from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.health_http import create_health_app


def test_ready_503_without_redis() -> None:
    app = create_health_app(
        redis_getter=lambda: None,
        intake_channel="orion:cortex:request",
        service_name="cortex-orch",
        service_version="0.2.0",
    )
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_200_when_consumer_ready() -> None:
    fake_redis = MagicMock()
    app = create_health_app(
        redis_getter=lambda: fake_redis,
        intake_channel="orion:cortex:request",
        service_name="cortex-orch",
        service_version="0.2.0",
    )
    with patch(
        "app.health_http.check_bus_consumer_readiness",
        new=AsyncMock(return_value=MagicMock(ok=True, bus_consumer_ready=True, subscriber_count=1)),
    ), patch("app.health_http.bus_consumer_readiness_v1") as mock_v1:
        mock_v1.return_value = MagicMock(ok=True, model_dump=lambda mode: {"ok": True})
        client = TestClient(app)
        resp = client.get("/ready")
    assert resp.status_code == 200
