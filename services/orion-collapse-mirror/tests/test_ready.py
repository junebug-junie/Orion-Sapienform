from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from orion.bus.consumer_readiness import BusConsumerReadinessResult

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_app(monkeypatch: pytest.MonkeyPatch):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(SERVICE_DIR)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(SERVICE_DIR))

    rabbit = MagicMock()
    rabbit.bus.redis = object()
    hunter = MagicMock()
    hunter.bus.redis = rabbit.bus.redis

    async def _start_services(_stop_event):
        return rabbit, hunter

    monkeypatch.setattr("app.bus_runtime.start_services", _start_services)
    from app.main import app  # noqa: WPS433

    return app, rabbit, hunter


def test_ready_reports_unavailable_when_redis_missing(monkeypatch: pytest.MonkeyPatch):
    app, rabbit, hunter = _load_app(monkeypatch)
    hunter.bus.redis = None
    rabbit.bus.redis = None

    with TestClient(app) as tc:
        resp = tc.get("/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["bus_consumer_ready"] is False


def test_ready_ok_when_intake_and_exec_subscribers_present(monkeypatch: pytest.MonkeyPatch):
    app, _rabbit, _hunter = _load_app(monkeypatch)
    ready = BusConsumerReadinessResult(
        ok=True,
        bus_consumer_ready=True,
        intake_channel="orion:collapse:intake",
        subscriber_count=1,
        dependency_status="available",
    )

    with (
        patch(
            "app.main.check_bus_consumer_readiness",
            new=AsyncMock(side_effect=[ready, ready]),
        ),
        patch(
            "app.main.redis_pubsub_numsub",
            new=AsyncMock(
                return_value={
                    "orion:collapse:intake": 1,
                    "orion:exec:request:CollapseMirrorService": 1,
                    "orion:collapse:sql-write": 2,
                }
            ),
        ),
        TestClient(app) as tc,
    ):
        resp = tc.get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["bus_consumer_ready"] is True
    assert body["intake"]["subscriber_count"] == 1
    assert body["exec"]["subscriber_count"] == 1


def test_ready_503_when_exec_subscriber_missing(monkeypatch: pytest.MonkeyPatch):
    app, _rabbit, _hunter = _load_app(monkeypatch)
    intake_ready = BusConsumerReadinessResult(
        ok=True,
        bus_consumer_ready=True,
        intake_channel="orion:collapse:intake",
        subscriber_count=1,
        dependency_status="available",
    )
    exec_missing = BusConsumerReadinessResult(
        ok=False,
        bus_consumer_ready=False,
        intake_channel="orion:exec:request:CollapseMirrorService",
        subscriber_count=0,
        dependency_status="unavailable",
        error="no subscribers on intake channel: orion:exec:request:CollapseMirrorService",
    )

    with (
        patch(
            "app.main.check_bus_consumer_readiness",
            new=AsyncMock(side_effect=[intake_ready, exec_missing]),
        ),
        patch(
            "app.main.redis_pubsub_numsub",
            new=AsyncMock(
                return_value={
                    "orion:collapse:intake": 1,
                    "orion:exec:request:CollapseMirrorService": 0,
                    "orion:collapse:sql-write": 2,
                }
            ),
        ),
        TestClient(app) as tc,
    ):
        resp = tc.get("/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["exec"]["subscriber_count"] == 0
