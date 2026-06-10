from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app import recall_tools
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


@pytest.mark.asyncio
async def test_recall_query_disabled_returns_empty():
    result = await recall_tools.recall_query(None, query="Denver", profile="assist.light.v1")
    assert result.hits == []
    assert result.query == "Denver"


@pytest.mark.asyncio
async def test_recall_query_bus_rpc_parses_bundle(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_real_recall_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "orion_bus_enabled", True)

    bundle_payload = {
        "bundle": {
            "rendered": "Denver mention",
            "items": [
                {
                    "id": "mem-1",
                    "source": "vector",
                    "snippet": "User said Denver",
                    "score": 0.9,
                }
            ],
            "stats": {"latency_ms": 3, "profile": "assist.light.v1"},
        }
    }
    reply_env = BaseEnvelope(
        kind="recall.reply.v1",
        source=ServiceRef(name="recall", version="1"),
        correlation_id=uuid4(),
        payload=bundle_payload,
    )
    bus = SimpleNamespace(
        codec=SimpleNamespace(
            decode=lambda _raw: SimpleNamespace(ok=True, envelope=reply_env, error=None)
        ),
        rpc_request=AsyncMock(return_value={"data": b"x"}),
    )
    result = await recall_tools.recall_query(bus, query="Denver", profile="assist.light.v1", limit=5)
    assert len(result.hits) == 1
    assert result.hits[0]["snippet"] == "User said Denver"
    bus.rpc_request.assert_awaited_once()
