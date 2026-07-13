from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from orion.schemas.chat_projection import ChatSessionProjectionV1, ChatTurnStateV1
from orion.schemas.route_projection import (
    RouteArbitrationProjectionV1,
    RouteArbitrationRunStateV1,
)
from orion.substrate.chat_loop.constants import CHAT_SESSION_PROJECTION_ID
from orion.substrate.route_loop.constants import ROUTE_ARBITRATION_PROJECTION_ID

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


@pytest.mark.asyncio
async def test_chat_session_endpoint_returns_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    now = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    proj = ChatSessionProjectionV1(
        projection_id=CHAT_SESSION_PROJECTION_ID,
        generated_at=now,
        turns={
            "turn-1": ChatTurnStateV1(
                trace_id="trace-1",
                turn_id="turn-1",
                session_id="session-1",
                node_id="athena",
                observed_at=now,
                word_count=12,
                last_updated_at=now,
            )
        },
        total_turn_count=1,
        sessions=["session-1"],
    )
    store = MagicMock()
    store.load_chat_session_projection.return_value = proj
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/chat_session")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["projection"]["turns"]["turn-1"]["session_id"] == "session-1"
    store.load_chat_session_projection.assert_called_once_with(CHAT_SESSION_PROJECTION_ID)


@pytest.mark.asyncio
async def test_chat_session_endpoint_no_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    store = MagicMock()
    store.load_chat_session_projection.return_value = None
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/chat_session")

    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "reason": "no_projection"}


@pytest.mark.asyncio
async def test_route_arbitration_endpoint_returns_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    now = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    proj = RouteArbitrationProjectionV1(
        projection_id=ROUTE_ARBITRATION_PROJECTION_ID,
        generated_at=now,
        runs={
            "trace-1": RouteArbitrationRunStateV1(
                trace_id="trace-1",
                correlation_id="corr-1",
                session_id="session-1",
                turn_id="turn-1",
                node_id="athena",
                lane="mind",
                lane_reason="high_salience",
                mind_requested=True,
                output_mode="stream",
                last_updated_at=now,
            )
        },
    )
    store = MagicMock()
    store.load_route_arbitration.return_value = proj
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/route_arbitration")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["projection"]["runs"]["trace-1"]["lane"] == "mind"
    store.load_route_arbitration.assert_called_once_with(ROUTE_ARBITRATION_PROJECTION_ID)


@pytest.mark.asyncio
async def test_route_arbitration_endpoint_no_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    store = MagicMock()
    store.load_route_arbitration.return_value = None
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/route_arbitration")

    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "reason": "no_projection"}
